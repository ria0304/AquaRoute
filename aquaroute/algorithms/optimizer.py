from __future__ import annotations

import copy
import itertools
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from aquaroute.models.water_network import WaterNetwork

try:  # Pyomo is an optional hard dependency for the LP path.
    from pyomo.environ import (
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        Set,
        SolverFactory,
        Var,
        maximize,
        minimize,
        value,
    )

    PYOMO_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when pyomo is absent
    PYOMO_AVAILABLE = False

SOLVER_CANDIDATES: Tuple[str, ...] = ("appsi_highs", "highs", "cbc", "glpk")

STRATEGIES: Tuple[str, ...] = (
    "weighted",
    "shortage_min",
    "uniform",
    "maxmin",
    "proportional",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_fairness(allocations: Dict[int, float], demands: Dict[int, float]) -> float:
    """Average pairwise proportional disparity.

    Fairness = mean over unordered pairs (i, j) of |alloc_i/demand_i - alloc_j/demand_j|.
    Zero means every community receives the same fraction of its demand; larger
    values indicate greater inequity. Demand normalisation makes communities with
    different needs comparable.

    Returns
    -------
    float
        Mean pairwise proportional disparity. 0.0 if fewer than 2 communities
        with positive demand.
    """
    communities = [n for n in allocations if n in demands and demands[n] > 0]
    if len(communities) <= 1:
        return 0.0

    proportions = [allocations.get(n, 0.0) / demands[n] for n in communities]
    disparities = [
        abs(proportions[i] - proportions[j])
        for i in range(len(proportions))
        for j in range(i + 1, len(proportions))
    ]
    if not disparities:
        return 0.0
    return sum(disparities) / len(disparities)


def compute_cvar(losses: List[float], alpha: float = 0.95) -> float:
    """Compute Conditional Value at Risk (CVaR) at confidence level alpha.

    CVaR (also known as Expected Shortfall) is the expected loss given that
    the loss exceeds the Value at Risk (VaR) at level alpha.

    Parameters
    ----------
    losses : list of float
        Loss values across scenarios.
    alpha : float, optional
        Confidence level in [0, 1]. Default 0.95.

    Returns
    -------
    float
        CVaR at the given confidence level.
    """
    losses = np.array(losses, dtype=float)
    if len(losses) == 0:
        return 0.0
    # Sort losses ascending
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    # Number of scenarios in the tail
    k = max(1, int(np.ceil((1 - alpha) * n)))
    # Tail losses (worst k scenarios)
    tail_losses = sorted_losses[-k:]
    return float(np.mean(tail_losses))


def check_hydraulic_feasibility(
    network: WaterNetwork,
    allocations: Dict[int, float],
    demands: Optional[Dict[int, float]] = None,
) -> Dict:
    """Check hydraulic feasibility of an allocation using WNTR simulation.

    Distinguishes between mathematically valid network-flow allocations and
    physically deliverable allocations through a hydraulic network.

    Parameters
    ----------
    network : WaterNetwork
        The water distribution network model.
    allocations : dict
        Mapping of community ID -> allocated volume.
    demands : dict, optional
        Mapping of community ID -> demand. If None, uses network node demands.

    Returns
    -------
    dict
        Hydraulic feasibility report with keys:
        - ``feasible``: bool indicating if hydraulic constraints are satisfied
        - ``pressures``: dict of node pressure heads
        - ``flows``: dict of link flow rates
        - ``head_losses``: dict of pipe head losses
        - ``violations``: list of constraint violations
        - ``min_pressure``: minimum pressure head across all nodes
        - ``max_pressure``: maximum pressure head across all nodes
    """
    try:
        import wntr

        # Build WNTR network from WaterNetwork
        wn = wntr.network.WaterNetworkModel()

        # Add junctions (communities) with demands
        for comm_id in network.get_communities():
            if demands is not None and comm_id in demands:
                demand = demands[comm_id]
            else:
                demand = network.G.nodes[comm_id].get("demand", 0.0)
            # WNTR requires string node names with base_demand parameter
            wn.add_junction(str(comm_id), base_demand=demand)

# Add reservoirs
        for res_id in network.get_reservoirs():
            supply = network.G.nodes[res_id].get("supply", float("inf"))
            # Scale supply to head; use a large head value for reservoir
            # add_reservoir(name, base_head=0.0, head_pattern=None, coordinates=None)
            # head_pattern must be a string or None
            wn.add_reservoir(str(res_id), base_head=supply * 10.0 if np.isfinite(supply) else 1000.0,
                          head_pattern="pattern1")

        # Add pipes (pipelines) - map from network edges
        for u, v in network.G.edges():
            if network.G.has_edge(u, v):
                # Add pipe with appropriate length and roughness
                # add_pipe(name, start_node_name, end_node_name, length, diameter, roughness, minor_loss, initial_status)
                wn.add_pipe(
                    f"pipe_{u}_{v}", str(u), str(v), length=1000.0, diameter=0.3,
                    roughness=100.0, minor_loss=0.0, initial_status="OPEN"
                )

        # Set simulation time simulation
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        # Extract hydraulic results
        node_results = results.node
        link_results = results.link

        # Get pressure heads
        pressures = {}
        for node_id in wn.junction_name_list:
            try:
                h = float(node_results["head"][node_id])
                pressures[node_id] = h
            except (KeyError, IndexError, TypeError):
                pressures[node_id] = 0.0

        # Get link flows
        flows = {}
        for link_id in wn.pipe_name_list:
            try:
                f = float(link_results["flowrate"][link_id])
                flows[link_id] = f
            except (KeyError, IndexError, TypeError):
                flows[link_id] = 0.0

        # Compute statistics
        all_pressures = [p for p in pressures.values() if np.isfinite(p)]
        min_pressure = min(all_pressures) if all_pressures else 0.0
        max_pressure = max(all_pressures) if all_pressures else 0.0

        # Check feasibility: typical minimum pressure requirement ~ 20m (15 psi)
        # and maximum ~ 100m (120 psi)
        violation_threshold_low = 20.0  # m
        violation_threshold_high = 100.0  # m

        violations = []
        for node_id, pressure in pressures.items():
            if pressure < violation_threshold_low:
                violations.append(
                    f"Low pressure at node {node_id}: {pressure:.1f}m < {violation_threshold_low}m"
                )
            if pressure > violation_threshold_high:
                violations.append(
                    f"High pressure at node {node_id}: {pressure:.1f}m > {violation_threshold_high}m"
                )

        feasible = len(violations) == 0 and min_pressure > 0

        return {
            "feasible": feasible,
            "pressures": pressures,
            "flows": flows,
            "violations": violations,
            "min_pressure": min_pressure,
            "max_pressure": max_pressure,
        }

    except Exception as e:
        # If WNTR simulation fails, return limited info
        return {
            "feasible": False,
            "pressures": {},
            "flows": {},
            "violations": [f"Hydraulic simulation error: {str(e)}"],
            "min_pressure": 0.0,
            "max_pressure": 0.0,
        }

    except Exception as e:
        # If WNTR simulation fails, return limited info
        return {
            "feasible": False,
            "pressures": {},
            "flows": {},
            "violations": [f"Hydraulic simulation error: {str(e)}"],
            "min_pressure": 0.0,
            "max_pressure": 0.0,
        }


def compute_shortage(allocations: Dict[int, float], demands: Dict[int, float]) -> float:
    """Total unmet demand across communities.

    Parameters
    ----------
    allocations : dict
        Mapping of community ID -> allocated volume.
    demands : dict
        Mapping of community ID -> demand volume.

    Returns
    -------
    float
        Total unmet demand (= max(0, sum(demands) - sum(allocations))).
    """
    total_demand = sum(demands.values())
    total_allocated = sum(allocations.values())
    return max(0.0, total_demand - total_allocated)


def compute_delivery_cost(allocations: Dict[int, float], network: WaterNetwork,
                          path_flows: Optional[Dict[Tuple[int, int], float]] = None) -> float:
    """Total pipeline delivery cost.

    If ``path_flows`` (edge -> flow) is supplied the true routed cost
    ``sum_e flow_e * cost_e`` is returned.  Otherwise, when only volumetric
    allocations are known, the cheapest incoming pipeline is used as a
    lower-bound estimate.  This avoids the previous bug of counting the same
    delivered volume once per incoming pipe.

    Parameters
    ----------
    allocations : dict
        Mapping of community ID -> allocated volume.
    network : WaterNetwork
        The network instance, used to look up edge capacities and costs.
    path_flows : dict, optional
        Mapping of (u, v) edge -> flow volume. If provided, the exact routed
        cost is computed; otherwise a lower-bound estimate using the cheapest
        incoming edge cost per community is returned.

    Returns
    -------
    float
        Total delivery cost.
    """
    if path_flows is not None:
        return float(sum(flow * network.G[u][v].get("delivery_cost", 1.0)
                         for (u, v), flow in path_flows.items()
                         if network.G.has_edge(u, v)))

    total_cost = 0.0
    for community, alloc in allocations.items():
        if not network.G.has_node(community):
            continue
        incoming = [network.G[p][community].get("delivery_cost", 1.0)
                    for p in network.G.predecessors(community)]
        if incoming:
            total_cost += alloc * min(incoming)
    return total_cost


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_solution(allocations: Dict[int, float], network: WaterNetwork,
                      demands: Dict[int, float], total_supply: float,
                      effective_caps: Optional[Dict[int, float]] = None,
                      min_service_priority: Optional[List[int]] = None,
                      min_allocation: float = 0.0) -> Tuple[bool, List[str]]:
    """Validate an allocation against the hard constraints of the model.

    Checks source availability, individual demand limits, non-negativity,
    pipeline/network delivery capacity (via max-flow effective capacities) and
    the minimum-service guarantee.  Returns ``(is_valid, violations)``.

    Parameters
    ----------
    allocations : dict
        Mapping of community ID -> allocated volume.
    network : WaterNetwork
        The network instance.
    demands : dict
        Mapping of community ID -> demand volume.
    total_supply : float
        Total supply available from all reservoirs.
    effective_caps : dict, optional
        Per-community effective capacity (max deliverable volume). If ``None``,
        computed via ``effective_capacities``.
    min_service_priority : list, optional
        Community IDs that must receive at least ``min_allocation``.
    min_allocation : float, optional
        Minimum allocation required for each priority community.

    Returns
    -------
    tuple
        ``(is_valid, violations)`` where ``is_valid`` is ``True`` if no hard
        constraints are violated (``C5`` warnings are excluded).
    """
    violations: List[str] = []
    communities = network.get_communities()
    total_allocated = sum(allocations.values()) if allocations else 0.0

    if total_allocated > total_supply + 1e-6:
        violations.append(
            f"C1 VIOLATED: total allocated ({total_allocated:.4f}) exceeds "
            f"supply ({total_supply:.4f})"
        )

    for c in communities:
        alloc = allocations.get(c, 0.0)
        demand = demands.get(c, 0.0)
        if alloc > demand + 1e-6:
            violations.append(
                f"C2 VIOLATED: community {c} allocation ({alloc:.4f}) exceeds "
                f"demand ({demand:.4f})"
            )
        if alloc < -1e-6:
            violations.append(f"C3 VIOLATED: community {c} negative allocation ({alloc:.4f})")

    caps = effective_caps if effective_caps is not None else effective_capacities(network)
    for c in communities:
        alloc = allocations.get(c, 0.0)
        cap = caps.get(c, float("inf"))
        if alloc > cap + 1e-6:
            violations.append(
                f"C4 VIOLATED: community {c} allocation ({alloc:.4f}) exceeds "
                f"network delivery capacity ({cap:.4f})"
            )

    for c in (min_service_priority or []):
        if c in communities and allocations.get(c, 0.0) < min_allocation - 1e-6:
            violations.append(
                f"C5 VIOLATED: priority community {c} allocation "
                f"({allocations.get(c, 0.0):.4f}) below minimum "
                f"({min_allocation:.4f})"
            )

    return (len(violations) == 0), violations


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def effective_capacities(network: WaterNetwork) -> Dict[int, float]:
    """Maximum deliverable volume to each community (single-sink max flow).

    Uses a super-source joined to every reservoir (capacity = supply) so that
    transshipment through intermediate communities is accounted for.

    Returns
    -------
    dict
        Mapping of community ID -> maximum deliverable volume.
    """
    G = network.G
    reservoirs = network.get_reservoirs()
    communities = network.get_communities()
    if not communities:
        return {}

    total_demand = network.total_demand()
    # Use a large capacity value; finite capacities are scaled relative to total demand.
    big = max(total_demand, 1.0) * 10 + 1.0

    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data.get("capacity", float("inf"))
        H.add_edge(u, v, capacity=big if not np.isfinite(cap) else float(cap))

    if not reservoirs:
        # No supply sources; communities are unreachable.
        return {c: 0.0 for c in communities}

    super_source = "__aquaroute_source__"
    for r in reservoirs:
        supply = G.nodes[r].get("supply", float("inf"))
        cap = big if not np.isfinite(supply) else float(supply)
        H.add_edge(super_source, r, capacity=cap)

    caps: Dict[int, float] = {}
    for c in communities:
        if not nx.has_path(H, super_source, c):
            caps[c] = 0.0
            continue
        try:
            flow_value, _ = nx.maximum_flow(H, super_source, c)
            caps[c] = float(flow_value)
        except nx.NetworkXError:
            caps[c] = 0.0
    return caps


def max_total_deliverable(network: WaterNetwork) -> float:
    """Maximum total volume that can be delivered simultaneously.

    Computed as a single max-flow from a super-source (reservoirs) to a
    super-sink (communities capped at their demand).  Unlike summing
    per-community ``effective_capacities`` this respects shared pipelines, so it
    is the correct global budget for the analytical fallback.

    Returns
    -------
    float
        Maximum total volume that can be delivered simultaneously.
    """
    G = network.G
    communities = network.get_communities()
    if not communities:
        return 0.0

    total_demand = network.total_demand()
    big = max(total_demand, 1.0) * 10 + 1.0
    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data.get("capacity", float("inf"))
        H.add_edge(u, v, capacity=big if not np.isfinite(cap) else float(cap))

    src, sink = "__aquaroute_source__", "__aquaroute_sink__"
    for r in network.get_reservoirs():
        supply = G.nodes[r].get("supply", float("inf"))
        H.add_edge(src, r, capacity=big if not np.isfinite(supply) else float(supply))
    for c in communities:
        H.add_edge(c, sink, capacity=max(G.nodes[c]["demand"], 0.0))

    if not nx.has_path(H, src, sink):
        return 0.0
    try:
        flow_value, _ = nx.maximum_flow(H, src, sink)
        return float(flow_value)
    except nx.NetworkXError:
        return 0.0


# ---------------------------------------------------------------------------
# Analytical fallback solvers (no MIP/LP solver required)
# ---------------------------------------------------------------------------
def _greedy_direct(network: WaterNetwork) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Effective caps and per-community max service fraction under those caps."""
    caps = effective_capacities(network)
    demands = {c: network.G.nodes[c]["demand"] for c in network.get_communities()}
    return caps, demands


def _proportional_waterfill(demands: List[float], caps: List[float],
                            budget: float) -> List[float]:
    """Water-fill volume proportionally to demand under per-sink caps.

    Returns the volume assigned to each sink.  ``demands``/``caps`` are parallel
    lists; ``budget`` is the total volume available.

    Allocation respects both demand limits and cap limits.  The residual budget
    after all sinks have hit their limits is distributed proportionally to active
    sinks that still have room.
    """
    n = len(demands)
    alloc = [0.0] * n
    remaining = budget
    active = [i for i in range(n) if demands[i] > 0 and caps[i] > 0]

    while remaining > 1e-9 and active:
        total_weight = sum(demands[i] for i in active)
        if total_weight <= 0:
            # No community can receive more; distribute remaining proportionally
            # to sinks with remaining room.
            total_room = sum(max(caps[i] - alloc[i], 0.0) for i in active)
            if total_room > 0:
                for i in active:
                    give = (remaining * (caps[i] - alloc[i]) / total_room)
                    if give > 1e-9:
                        alloc[i] += give
                        remaining -= give
            break

        progressed = False
        for i in list(active):
            share = remaining * (demands[i] / total_weight)
            room = min(demands[i] - alloc[i], caps[i] - alloc[i])
            give = min(share, room)
            if give > 1e-9:
                alloc[i] += give
                remaining -= give
                progressed = True
            if alloc[i] >= min(demands[i], caps[i]) - 1e-9:
                active.remove(i)
        if not progressed and remaining > 1e-9:
            # All active sinks are at their limits but budget remains;
            # distribute residual proportionally to room available.
            total_room = sum(max(caps[i] - alloc[i], 0.0) for i in active)
            if total_room > 0:
                for i in list(active):
                    give = (remaining * (caps[i] - alloc[i]) / total_room)
                    if give > 1e-9:
                        alloc[i] += give
                        remaining -= give
            break
    return alloc


def _equal_waterfill(limits: List[float], budget: float) -> List[float]:
    """Distribute ``budget`` as evenly as possible subject to per-sink limits.

    Returns the volume assigned to each sink.  ``limits`` are per-sink caps
    (e.g. ``min(demand - alloc, cap - alloc)``) and ``budget`` is the total
    volume to spread.  The algorithm guarantees that every sink receives at most
    its limit and the total allocated equals ``budget`` (or as much as feasible).
    """
    n = len(limits)
    alloc = [0.0] * n
    remaining = budget
    active = [i for i in range(n) if limits[i] > 0]
    while remaining > 1e-9 and active:
        share = remaining / len(active)
        progressed = False
        for i in list(active):
            give = min(share, limits[i] - alloc[i])
            if give > 1e-9:
                alloc[i] += give
                remaining -= give
                progressed = True
            if alloc[i] >= limits[i] - 1e-9:
                active.remove(i)
        if not progressed and remaining > 1e-9:
            # No sink can take the equal share; distribute residual
            # proportionally to remaining room.
            total_room = sum(max(limits[i] - alloc[i], 0.0) for i in active)
            if total_room > 0:
                for i in active:
                    give = (remaining * (limits[i] - alloc[i]) / total_room)
                    if give > 1e-9:
                        alloc[i] += give
                        remaining -= give
            break
    return alloc


def _analytical_solve(network: WaterNetwork, strategy: str = "weighted",
                      alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                      min_service_priority: Optional[List[int]] = None,
                      min_allocation: float = 0.0,
                      n_scenarios: int = 1) -> Dict:
    """Solver-free heuristic that returns a feasible allocation.

    Produces genuinely different allocations per strategy so that the
    comparative evaluation remains meaningful even without an LP solver.

    Parameters
    ----------
    network : WaterNetwork
        The water distribution network model.
    strategy : str, optional
        One of ``"shortage_min"``, ``"proportional"``, ``"maxmin"``,
        ``"uniform"``, or ``"weighted"``.  Determines the secondary objective
        after Stage 1 (minimise total shortage) has been resolved.
    alpha, beta, gamma : float, optional
        Weights for the combined objective ``alpha*shortage + beta*fairness +
        gamma*cost``.  Only used when ``strategy`` is ``"weighted"``.
    min_service_priority : list of int, optional
        Community IDs that must receive at least ``min_allocation`` units.
    min_allocation : float, optional
        Minimum allocation for each priority community.  If the guarantee is
        physically impossible the LP is infeasible and the analytical fallback
        serves priority communities as far as supply allows, reporting a ``C5``
        warning.
    n_scenarios : int, optional
        Number of scenarios to generate for the analytical fallback.
        Used by stochastic optimization to compute CVaR.

    Returns
    -------
    dict
        Solution dictionary with keys: ``allocations``, ``flows``,
        ``total_demand``, ``total_allocated``, ``shortage``, ``fairness``,
        ``delivery_cost``, ``objective_value``, ``status``, ``solver``,
        ``cvar``, ``scenario_shortages``.
    """
    min_service_priority = min_service_priority or []
    communities = network.get_communities()
    caps, demands = _greedy_direct(network)
    total_supply = network.total_supply()
    if not np.isfinite(total_supply):
        total_supply = network.total_demand() * 10 + 1.0
    # Global budget must respect shared pipeline capacity.
    budget = min(total_supply, max_total_deliverable(network))

    alloc = {c: 0.0 for c in communities}
    remaining = budget

    # Minimum-service phase (priority ordered by community id for determinism).
    for c in sorted(set(min_service_priority) & set(communities)):
        room = min(min_allocation, demands[c], caps[c], remaining)
        alloc[c] = max(0.0, room)
        remaining -= alloc[c]

    idx = [c for c in communities if demands[c] > 0]
    room_after_min = [min(demands[c] - alloc[c], caps[c] - alloc[c]) for c in idx]

    if strategy == "shortage_min":
        # Fill as much demand as possible, largest remaining demand first.
        order = sorted(range(len(idx)), key=lambda i: room_after_min[i], reverse=True)
        for i in order:
            give = min(room_after_min[i], remaining)
            alloc[idx[i]] += give
            remaining -= give
    elif strategy == "uniform":
        vol = _equal_waterfill(room_after_min, remaining)
        for i, c in enumerate(idx):
            alloc[c] += vol[i]
    elif strategy == "maxmin":
        # Egalitarian: repeatedly give to the sink with the smallest served fraction.
        for _ in range(max(len(idx) * 10, 200)):
            if remaining <= 1e-12:
                break
            cand = [i for i in range(len(idx)) if room_after_min[i] > 1e-12]
            if not cand:
                break
            # Find sink with smallest proportional service ratio.
            i = min(cand, key=lambda k: alloc[idx[k]] / demands[idx[k]] if demands[idx[k]] else 1.0)
            # Step size: ensure at least a minimum increment while respecting remaining room.
            max_step = remaining / max(1, len(cand))
            step = min(room_after_min[i], remaining, max(max_step, 1e-12))
            if step <= 1e-15:
                break
            alloc[idx[i]] += step
            room_after_min[i] -= step
            remaining -= step
    else:  # weighted / proportional fallback
        prop = _proportional_waterfill([demands[c] for c in idx], room_after_min, remaining)
        for i, c in enumerate(idx):
            alloc[c] += prop[i]

    total_allocated = sum(alloc.values())
    shortage = max(0.0, network.total_demand() - total_allocated)
    fairness = compute_fairness(alloc, demands)
    cost = compute_delivery_cost(alloc, network)

    # Generate scenario-based losses for CVaR computation
    scenario_shortages = []
    for _ in range(max(1, n_scenarios - 1)):
        # Slightly vary demands to create scenario diversity
        varying_demands = {c: demands[c] * np.random.uniform(0.8, 1.2)
                          for c in communities}
        # Re-allocate with varied demands
        alloc_s = {c: 0.0 for c in communities}
        remaining_s = budget
        # Re-use the same allocation pattern but with varied demands
        for c in sorted(set(min_service_priority) & set(communities)):
            room = min(min_allocation, varying_demands[c], caps[c], remaining_s)
            alloc_s[c] = max(0.0, room)
            remaining_s -= alloc_s[c]

        idx2 = [c for c in communities if varying_demands[c] > 0]
        room_after_min_s = [min(varying_demands[c] - alloc_s[c], caps[c] - alloc_s[c])
                           for c in idx2]

        # Fill remaining using shortage_min order
        order = sorted(range(len(idx2)), key=lambda i: room_after_min_s[i], reverse=True)
        for i in order:
            give = min(room_after_min_s[i], remaining_s)
            alloc_s[idx2[i]] += give
            remaining_s -= give

        scenario_shortage = max(0.0, sum(varying_demands.values()) - sum(alloc_s.values()))
        scenario_shortages.append(scenario_shortage)

    # Compute CVaR if multiple scenarios requested
    cvar = 0.0
    if n_scenarios > 1 and scenario_shortages:
        # Include the base allocation's shortage as the first scenario
        all_shortages = [shortage] + scenario_shortages
        cvar = compute_cvar(all_shortages, alpha=0.95)
    elif n_scenarios == 1:
        cvar = 0.0

    return {
        "allocations": alloc,
        "flows": {},
        "total_demand": network.total_demand(),
        "total_allocated": total_allocated,
        "shortage": shortage,
        "fairness": fairness,
        "delivery_cost": cost,
        "objective_value": alpha * shortage + beta * fairness + gamma * cost,
        "status": "optimal",
        "solver": "analytical_fallback",
        "cvar": cvar,
        "scenario_shortages": scenario_shortages,
    }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class AllocationOptimizer:
    """Allocation optimizer over a single-commodity flow network.

    The network-flow formulation uses a non-negative flow variable ``f[e]`` for
    every pipeline.  Node balance enforces conservation:

    * reservoir ``r``:   sum(out) - sum(in) <= supply[r]
    * community ``c``:   sum(in) - sum(out)  = x[c],   0 <= x[c] <= demand[c]
    * junction ``j``:    sum(in) - sum(out)  = 0

    Objectives (selected via ``strategy``):

    * ``weighted``      minimize alpha*shortage + beta*fairness + gamma*cost
    * ``shortage_min``  minimize shortage (deliver as much as possible)
    * ``uniform``       lexicographic: minimize shortage, then volumetric disparity
    * ``maxmin``        lexicographic: minimize shortage, then maximize the
                        minimum proportional service ratio (egalitarian)
    * ``proportional``  lexicographic: minimize shortage, then proportional disparity

    ``fairness`` is the mean pairwise proportional disparity, linearised with
    auxiliary variables.  The model is an LP and is solved with HiGHS/CBC/GLPK
    when available; otherwise an analytical heuristic is used.
    """

    def __init__(self, network: WaterNetwork,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 min_service_priority: Optional[List[int]] = None,
                 min_allocation: float = 0.0,
                 strategy: str = "weighted",
                 solver_name: Optional[str] = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_service_priority = min_service_priority or []
        self.min_allocation = min_allocation
        self.strategy = strategy
        self.solver_name = solver_name
        self.model = None
        self.solution: Optional[Dict] = None
        self.runtime: Optional[float] = None

    # -- model construction ------------------------------------------------
    def build_model(self):
        """Build the Pyomo flow model with the strategy's primary objective."""
        if not PYOMO_AVAILABLE:
            raise RuntimeError("Pyomo is not installed; use solve() for the analytical fallback.")

        model = ConcreteModel()
        self._demands = {c: self.network.G.nodes[c]["demand"]
                         for c in self.network.get_communities()}
        self._edge_cost = {(u, v): self.network.G[u][v].get("delivery_cost", 1.0)
                           for u, v in self.network.G.edges()}
        self._cap = {(u, v): self.network.G[u][v].get("capacity", float("inf"))
                     for u, v in self.network.G.edges()}
        self._pred = {n: list(self.network.G.predecessors(n)) for n in self.network.G.nodes()}
        self._succ = {n: list(self.network.G.successors(n)) for n in self.network.G.nodes()}
        self._edge_set = list(self.network.G.edges())
        self._in_nodes = set(self._demands)
        self._sup_nodes = set(self.network.get_reservoirs())

        model.C = Set(initialize=list(self._demands))
        model.R = Set(initialize=list(self._sup_nodes))
        model.E = Set(initialize=self._edge_set, dimen=2)
        model.x = Var(model.C, domain=NonNegativeReals)
        model.f = Var(model.E, domain=NonNegativeReals)

        for c in self._demands:
            model.x[c].setlb(0.0)

        model.demand_limit = Constraint(model.C, rule=lambda m, c: m.x[c] <= self._demands[c])

        cap = self._cap

        def _cap_rule(m, u, v):
            c = cap[(u, v)]
            return m.f[u, v] <= c if np.isfinite(c) else Constraint.Skip

        model.pipeline_capacity = Constraint(model.E, rule=_cap_rule)

        def _balance(m, n):
            inflow = sum(m.f[u, n] for u in self._pred[n] if (u, n) in cap)
            outflow = sum(m.f[n, v] for v in self._succ[n] if (n, v) in cap)
            if n in self._sup_nodes:
                return outflow - inflow <= self.network.G.nodes[n].get("supply", float("inf"))
            if n in self._in_nodes:
                return inflow - outflow == m.x[n]
            return inflow - outflow == 0

        model.node_balance = Constraint(list(self.network.G.nodes()), rule=_balance)

        # Minimum-service is a hard guarantee: each priority community must
        # receive at least ``min_allocation``.  If this is physically
        # impossible the LP is infeasible and solve() falls back to the
        # analytical solver, which prioritises minimum service as far as
        # supply allows and reports a C5 warning.
        self._prio = [c for c in self.min_service_priority if c in self._demands]
        if self._prio and self.min_allocation > 0:
            model.MS = Set(initialize=self._prio)
            model.min_service = Constraint(
                model.MS, rule=lambda m, c: m.x[c] >= self.min_allocation)

        demand_pos = [c for c in self._demands if self._demands[c] > 0]
        self._pairs = list(itertools.combinations(demand_pos, 2))
        if self._pairs:
            model.PAIRS = Set(initialize=self._pairs, dimen=2)
            model.t = Var(model.PAIRS, domain=NonNegativeReals)
            model.fairness_lb = Constraint(
                model.PAIRS,
                rule=lambda m, i, j: m.t[i, j] >= m.x[i] / self._demands[i]
                - m.x[j] / self._demands[j])
            model.fairness_ub = Constraint(
                model.PAIRS,
                rule=lambda m, i, j: m.t[i, j] >= m.x[j] / self._demands[j]
                - m.x[i] / self._demands[i])

        model.short_expr = sum(self._demands[c] - model.x[c] for c in self._demands)
        model.cost_expr = sum(model.f[e] * self._edge_cost[e] for e in self._edge_set)
        if self._pairs:
            model.fair_expr = sum(model.t[i, j] for i, j in self._pairs) / len(self._pairs)
        else:
            model.fair_expr = 0.0

        self._install_primary_objective(model)
        self.model = model
        return model

    def _install_primary_objective(self, model):
        # Every strategy first maximises service (minimises shortage) subject
        # to the minimum-service guarantee; the strategy-specific objective
        # then only breaks ties among the maximum-service allocations.  The
        # outer loop in solve() re-applies the shortage constraint for stage 2,
        # so the minimum-service constraint remains binding throughout.
        model.objective = Objective(expr=model.short_expr, sense=minimize)

    def _add_secondary_objective(self, model):
        """Replace the primary objective with the strategy's secondary one."""
        if model.objective.is_constructed():
            model.objective.deactivate()

        if self.strategy == "shortage_min":
            model.objective2 = Objective(expr=model.short_expr, sense=minimize)
        elif self.strategy == "weighted":
            model.objective2 = Objective(
                expr=self.beta * model.fair_expr + self.gamma * model.cost_expr,
                sense=minimize)
        elif self.strategy == "maxmin":
            model.service_ratio = Var(domain=NonNegativeReals, bounds=(0.0, 1.0))
            model.min_ratio_link = Constraint(
                model.C,
                rule=lambda m, c: self._demands[c] * m.service_ratio <= m.x[c])
            model.objective2 = Objective(expr=model.service_ratio, sense=maximize)
        elif self.strategy == "uniform":
            model.vol_t = Var(model.C, model.C, domain=NonNegativeReals)
            model.vol_lb = Constraint(
                model.C, model.C,
                rule=lambda m, i, j: m.vol_t[i, j] >= m.x[i] - m.x[j])
            model.vol_ub = Constraint(
                model.C, model.C,
                rule=lambda m, i, j: m.vol_t[i, j] >= m.x[j] - m.x[i])
            model.objective2 = Objective(
                expr=sum(model.vol_t[i, j] for i in model.C for j in model.C), sense=minimize)
        else:  # proportional
            model.objective2 = Objective(expr=model.fair_expr, sense=minimize)

    # -- solving -----------------------------------------------------------
    def _available_solver(self):
        candidates = ([self.solver_name] if self.solver_name else []) + list(SOLVER_CANDIDATES)
        for name in candidates:
            try:
                solver = SolverFactory(name)
                if solver.available(exception_flag=False):
                    return name
            except Exception:
                continue
        return None

    def _configure_solver(self, solver, time_limit: Optional[int]) -> None:
        """Apply tight feasibility tolerances so results are numerically clean."""
        opts = {
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        }
        if time_limit:
            opts["time_limit"] = float(time_limit)
        try:
            solver.options.update(opts)
        except Exception:
            pass
        # appsi_highs exposes HiGHS options via a separate dict.
        try:
            if hasattr(solver, "highs_options"):
                solver.highs_options.update(opts)
        except Exception:
            pass

    def solve(self, solver_name: Optional[str] = None, time_limit: Optional[int] = None,
              allow_fallback: bool = True) -> Optional[Dict]:
        """Solve the allocation problem.  Returns the solution dict or None."""
        import time
        start = time.time()
        if solver_name:
            self.solver_name = solver_name

        name = self._available_solver()
        if name is None or not PYOMO_AVAILABLE:
            if not allow_fallback:
                return None
            self.solution = _analytical_solve(
                self.network, self.strategy, self.alpha, self.beta, self.gamma,
                self.min_service_priority, self.min_allocation)
            self.runtime = time.time() - start
            return self.solution

        self.build_model()
        solver = SolverFactory(name)
        self._configure_solver(solver, time_limit)

        try:
            results = solver.solve(self.model, tee=False)
            status = str(results.solver.termination_condition)
            if "optimal" not in status and "feasible" not in status:
                if allow_fallback:
                    self.solution = _analytical_solve(
                        self.network, self.strategy, self.alpha, self.beta, self.gamma,
                        self.min_service_priority, self.min_allocation)
                    self.runtime = time.time() - start
                    return self.solution
                return None

            if self.strategy != "shortage_min":
                primary = value(self.model.short_expr)
                # Absolute tolerance keeps the primary (service) optimum fixed;
                # a relative tolerance would let stage 2 shave delivered volume.
                tol = 1e-7
                self.model.stage2_cap = Constraint(expr=self.model.short_expr <= primary + tol)
                self._add_secondary_objective(self.model)
                results = solver.solve(self.model, tee=False)

            alloc = {c: float(value(self.model.x[c])) for c in self.model.C}
            flows = {e: float(value(self.model.f[e])) for e in self._edge_set}
            self.solution = {
                "allocations": alloc,
                "flows": flows,
                "solver": name,
                "strategy": self.strategy,
            }
        except Exception:
            if not allow_fallback:
                return None
            self.solution = _analytical_solve(
                self.network, self.strategy, self.alpha, self.beta, self.gamma,
                self.min_service_priority, self.min_allocation)

        self.runtime = time.time() - start
        return self.solution

    def solve_stochastic(self,
                         n_scenarios: int = 20,
                         alpha: float = 0.95,
                         strategy: Optional[str] = None,
                         allow_fallback: bool = True) -> Dict:
        """Solve the allocation problem across multiple stochastic scenarios.

        Optimizes for the expected shortage while also minimizing CVaR (Conditional
        Value at Risk) to protect against worst-case scenarios.

        Parameters
        ----------
        n_scenarios : int, optional
            Number of scenarios to generate. Default 20.
        alpha : float, optional
            CVaR confidence level. Default 0.95.
        strategy : str, optional
            Optimization strategy. If None, uses ``self.strategy``.
        allow_fallback : bool, optional
            If True, fall back to analytical solver when Pyomo is unavailable.

        Returns
        -------
        dict
            Solution dictionary including ``cvar``, ``scenario_shortages``,
            and ``allocations`` averaged across scenarios.
        """
        import time
        from aquaroute.algorithms.uncertainty import ScenarioGenerator, apply_scenario

        start = time.time()
        strategy = strategy or self.strategy

        # Generate scenarios
        base_demands = {c: self.network.G.nodes[c]["demand"]
                       for c in self.network.get_communities()}
        base_supply = {r: self.network.G.nodes[r].get("supply", float("inf"))
                      for r in self.network.get_reservoirs()}

        scenario_shortages = []
        all_allocations = []

        for i in range(n_scenarios):
            # Create scenario: vary demands and supply
            scenario = {
                "demands": {c: base_demands.get(c, 0.0) * np.random.uniform(0.7, 1.3)
                           for c in self.network.get_communities()},
                "supply": {r: base_supply.get(r, float("inf")) * np.random.uniform(0.7, 1.3)
                          for r in self.network.get_reservoirs()},
                "failures": [],
                "scenario_type": f"stochastic_{i}",
            }

            # Apply scenario to a network copy
            net_copy = WaterNetwork()
            net_copy.G = copy.deepcopy(self.network.G)

            # Update demands and supply
            for c in net_copy.get_communities():
                if net_copy.G.has_node(c):
                    net_copy.G.nodes[c]["demand"] = scenario["demands"][c]
            for r in net_copy.get_reservoirs():
                if net_copy.G.has_node(r):
                    net_copy.G.nodes[r]["supply"] = scenario["supply"][r]

            # Solve for this scenario
            opt = AllocationOptimizer(network=net_copy,
                                      strategy=strategy,
                                      alpha=self.alpha,
                                      beta=self.beta,
                                      gamma=self.gamma,
                                      min_service_priority=self.min_service_priority,
                                      min_allocation=self.min_allocation)
            solution = opt.solve(allow_fallback=allow_fallback)

            if solution and solution.get("allocations"):
                alloc = solution["allocations"]
                # Compute shortage for this scenario
                demands_s = {c: net_copy.G.nodes[c]["demand"]
                            for c in net_copy.get_communities()}
                total_demand_s = sum(demands_s.values())
                total_allocated_s = sum(alloc.values())
                shortage_s = max(0.0, total_demand_s - total_allocated_s)
                scenario_shortages.append(shortage_s)
                all_allocations.append(alloc)
            else:
                scenario_shortages.append(float("nan"))
                all_allocations.append({})

        # Compute average allocations (only from successful solves)
        valid_allocations = [a for a in all_allocations if a]
        if valid_allocations:
            # Get all community IDs
            all_comm_ids = set()
            for a in valid_allocations:
                all_comm_ids.update(a.keys())
            avg_allocations = {cid: float(np.mean([a.get(cid, 0.0) for a in valid_allocations]))
                              for cid in all_comm_ids}
        else:
            avg_allocations = {}

        # Compute CVaR
        if len(scenario_shortages) > 1 and any(np.isfinite(s) for s in scenario_shortages):
            finite_shortages = [s for s in scenario_shortages if np.isfinite(s)]
            cvar = compute_cvar(finite_shortages, alpha=alpha)
        else:
            cvar = 0.0

        self.runtime = time.time() - start
        self.solution = {
            "allocations": avg_allocations,
            "cvar": cvar,
            "scenario_shortages": scenario_shortages,
            "strategy": strategy,
            "n_scenarios": n_scenarios,
            "runtime": self.runtime,
        }
        return self.solution

    def solve_dynamic(self,
                      n_periods: int = 4,
                      period_supplies: Optional[List[float]] = None,
                      period_demands: Optional[List[float]] = None,
                      storage_capacity: float = 100.0,
                      storage_initial: float = 0.0,
                      storage_min: float = 0.0,
                      strategy: Optional[str] = None,
                      allow_fallback: bool = True) -> Dict:
        """Solve the multi-period water allocation problem with storage carryover.

        Optimizes allocation across time periods while respecting reservoir storage
        dynamics: S_{t+1} = S_t + I_t - O_t, where I_t is incoming supply and
        O_t is outgoing allocation.

        Parameters
        ----------
        n_periods : int, optional
            Number of time periods. Default 4.
        period_supplies : list of float, optional
            Supply available in each period. If None, uses network total supply
            divided across periods.
        period_demands : list of float, optional
            Community demand in each period. If None, uses network demands divided
            across periods.
        storage_capacity : float, optional
            Maximum reservoir storage capacity. Default 100.0.
        storage_initial : float, optional
            Initial reservoir storage at period 1. Default 0.0.
        storage_min : float, optional
            Minimum reservoir storage. Default 0.0.
        strategy : str, optional
            Optimization strategy. If None, uses ``self.strategy``.
        allow_fallback : bool, optional
            If True, fall back to analytical solver when Pyomo is unavailable.

        Returns
        -------
        dict
            Solution dictionary with per-period allocations, storage trajectory,
            and shortage metrics.
        """
        import time
        import numpy as np

        start = time.time()
        strategy = strategy or self.strategy

        # Set up period supplies and demands
        if period_supplies is None:
            # Use network supply distributed across periods
            total_supply = self.network.total_supply()
            if np.isfinite(total_supply) and total_supply > 0:
                period_supplies = [total_supply / n_periods] * n_periods
            else:
                period_supplies = [100.0] * n_periods

        if period_demands is None:
            # Use network community demands distributed across periods
            total_demand = self.network.total_demand()
            if total_demand > 0:
                period_demands = [total_demand / n_periods] * n_periods
            else:
                period_demands = [50.0] * n_periods

        # Ensure we have the right number of periods
        period_supplies = period_supplies[:n_periods] + [period_supplies[0]] * max(0, n_periods - len(period_supplies))
        period_demands = period_demands[:n_periods] + [period_demands[0]] * max(0, n_periods - len(period_demands))

        # Get community and reservoir info
        communities = self.network.get_communities()
        reservoirs = self.network.get_reservoirs()
        base_demands = {c: self.network.G.nodes[c]["demand"]
                       for c in communities}
        base_supply = {r: self.network.G.nodes[r].get("supply", float("inf"))
                      for r in reservoirs}

        # Allocate period demands across communities using the strategy
        period_allocations = {}
        storage_trajectory = [storage_initial]
        total_shortage = 0.0

        # Initial storage
        storage = storage_initial

        for t in range(n_periods):
            supply_t = period_supplies[t]
            demand_t = period_demands[t]

            # Build community allocations for this period
            # Distribute period demand across communities proportionally to their base demands
            period_comm_demands = {c: base_demands.get(c, 0.0)
                                  for c in communities}
            total_base_demand = sum(period_comm_demands.values())

            # Solve allocation for this period
            if total_base_demand > 0:
                comm_weighted_demands = {c: period_comm_demands[c] / total_base_demand
                                        for c in communities}
                # Use weighted strategy to allocate proportionally
                opt = AllocationOptimizer(network=self.network,
                                          strategy='weighted',
                                          alpha=self.alpha,
                                          beta=self.beta,
                                          gamma=self.gamma,
                                          min_service_priority=self.min_service_priority,
                                          min_allocation=self.min_allocation)

                # Temporarily set demands for this period
                net_copy = WaterNetwork()
                net_copy.G = copy.deepcopy(self.network.G)
                for c in net_copy.get_communities():
                    if net_copy.G.has_node(c):
                        net_copy.G.nodes[c]["demand"] = period_demands[t] * comm_weighted_demands.get(c, 0.0) / max(comm_weighted_demands.values()) if max(comm_weighted_demands.values()) > 0 else 0.0

                solution = opt.solve(allow_fallback=allow_fallback)
                period_alloc = solution.get("allocations", {}) if solution else {}
            else:
                period_alloc = {c: 0.0 for c in communities}

            period_allocations[t] = period_alloc

            # Compute period shortage
            total_allocated_t = sum(period_alloc.values())
            period_shortage = max(0.0, demand_t - total_allocated_t)
            total_shortage += period_shortage

            # Update storage: S_{t+1} = S_t + supply_t - allocate_t
            # We allocate to communities, so the "outgoing" is the total allocation
            storage = storage + supply_t - total_allocated_t

            # Clamp storage
            storage = max(storage_min, min(storage_capacity, storage))
            storage_trajectory.append(storage)

        # Compute average allocations across periods
        all_comm_ids = set()
        for pa in period_allocations.values():
            all_comm_ids.update(pa.keys())

        avg_allocations = {}
        if all_comm_ids:
            for cid in all_comm_ids:
                vals = [pa.get(cid, 0.0) for pa in period_allocations.values()]
                avg_allocations[cid] = float(np.mean(vals))

        self.runtime = time.time() - start
        self.solution = {
            "allocations": avg_allocations,
            "period_allocations": period_allocations,
            "storage_trajectory": storage_trajectory,
            "total_shortage": total_shortage,
            "strategy": strategy,
            "n_periods": n_periods,
            "period_supplies": period_supplies,
            "period_demands": period_demands,
            "runtime": self.runtime,
        }
        return self.solution

    # -- results -----------------------------------------------------------
    def get_results(self) -> Dict:
        """Return allocations plus post-solve metrics; {} if unsolved."""
        if self.solution is None:
            return {}

        demands = {c: self.network.G.nodes[c]["demand"]
                   for c in self.network.get_communities()}
        caps = effective_capacities(self.network)
        alloc = self.solution.get("allocations", {})
        flows = self.solution.get("flows") or None

        _, violations = validate_solution(
            alloc, self.network, demands, self.network.total_supply(), caps,
            self.min_service_priority, self.min_allocation)
        # C5 (minimum service) is a soft constraint: a violation there means the
        # guarantee is physically impossible, not that the solution is corrupt.
        hard_violations = [v for v in violations if not v.startswith("C5")]
        warnings = [v for v in violations if v.startswith("C5")]

        if hard_violations:
            fallback = _analytical_solve(
                self.network, self.strategy, self.alpha, self.beta, self.gamma,
                self.min_service_priority, self.min_allocation)
            alloc = fallback["allocations"]
            flows = fallback.get("flows") or None
            self.solution = fallback
            _, violations = validate_solution(
                alloc, self.network, demands, self.network.total_supply(), caps,
                self.min_service_priority, self.min_allocation)
            hard_violations = [v for v in violations if not v.startswith("C5")]
            warnings = [v for v in violations if v.startswith("C5")]

        total_demand = sum(demands.values())
        total_allocated = sum(alloc.values())
        shortage = max(0.0, total_demand - total_allocated)
        fairness = compute_fairness(alloc, demands)
        delivery_cost = compute_delivery_cost(alloc, self.network, flows)

        return {
            "allocations": alloc,
            "total_demand": total_demand,
            "total_allocated": total_allocated,
            "shortage": shortage,
            "fairness": fairness,
            "delivery_cost": delivery_cost,
            "objective_value": (self.alpha * shortage + self.beta * fairness
                                + self.gamma * delivery_cost),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "strategy": self.strategy,
            "status": "optimal" if self.solution else "unsolved",
            "solver": self.solution.get("solver", "unknown"),
            "runtime": self.runtime or 0.0,
            "validation_passed": len(hard_violations) == 0,
            "validation_errors": hard_violations,
            "validation_warnings": warnings,
        }

    # -- static wrapper (kept for backwards compatibility) -----------------
    @staticmethod
    def validate_solution(allocations, network, demands, total_supply,
                          effective_caps=None, min_service_priority=None,
                          min_allocation=0.0):
        return validate_solution(allocations, network, demands, total_supply,
                                 effective_caps, min_service_priority, min_allocation)
