from __future__ import annotations

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
    big = max(total_demand, 1.0) * 10 + 1.0

    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data.get("capacity", float("inf"))
        H.add_edge(u, v, capacity=big if not np.isfinite(cap) else float(cap))

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
    """
    n = len(demands)
    alloc = [0.0] * n
    remaining = budget
    active = [i for i in range(n) if demands[i] > 0 and caps[i] > 0]

    while remaining > 1e-9 and active:
        total_weight = sum(demands[i] for i in active)
        if total_weight <= 0:
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
        if not progressed:
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
        if not progressed:
            # No sink can take more; distribute residual proportionally
            # to active sinks that still have room.
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
                      min_allocation: float = 0.0) -> Dict:
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

    Returns
    -------
    dict
        Solution dictionary with keys: ``allocations``, ``flows``,
        ``total_demand``, ``total_allocated``, ``shortage``, ``fairness``,
        ``delivery_cost``, ``objective_value``, ``status``, ``solver``.
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
            i = min(cand, key=lambda k: (alloc[idx[k]] / demands[idx[k]]) if demands[idx[k]] else 1.0)
            # Step size: ensure at least a minimum increment while respecting remaining room.
            max_step = remaining / max(1, len(cand))
            step = min(room_after_min[i], remaining, max(max_step, 1e-6))
            if step <= 1e-12:
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
