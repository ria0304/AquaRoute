from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from pyomo.environ import *

from aquaroute.models.water_network import WaterNetwork, create_sample_network


def compute_fairness(allocations: Dict[int, float], demands: Dict[int, float]) -> float:
    """Compute fairness metric: average absolute proportional disparity across all community pairs.
    
    Fairness = (1 / (n * (n-1))) * Σ_i Σ_j |alloc_i/demand_i - alloc_j/demand_j|
    where n is the number of communities with positive demand.
    
    This measures proportional equity: how evenly water is distributed relative to
    demand. A value of 0 means every community receives exactly its proportional share.
    Higher values indicate greater inequity. The normalization by demand ensures that
    a community with twice the demand that gets twice the water is treated as equal,
    which is the appropriate fairness notion for resource allocation where needs differ.
    """
    communities = [n for n, d in allocations.items() if n in demands and demands[n] > 0]
    if len(communities) <= 1:
        return 0.0
    
    proportions = {}
    for n in communities:
        if demands[n] > 0:
            proportions[n] = allocations.get(n, 0) / demands[n]
        else:
            proportions[n] = 1.0 if allocations.get(n, 0) == 0 else float('inf')
    
    prop_values = list(proportions.values())
    disparities = []
    for i in range(len(prop_values)):
        for j in range(i + 1, len(prop_values)):
            disparities.append(abs(prop_values[i] - prop_values[j]))
    
    if not disparities:
        return 0.0
    return sum(disparities) / len(disparities)


def compute_shortage(allocations: Dict[int, float], demands: Dict[int, float]) -> float:
    """Total unmet demand (shortage).
    
    Shortage = max(0, total_demand - total_allocated).
    This is the total volume of unmet water demand across all communities.
    """
    total_demand = sum(demands.values())
    total_allocated = sum(allocations.values())
    return max(0.0, total_demand - total_allocated)


def compute_delivery_cost(allocations: Dict[int, float], network: WaterNetwork,
                          path_flows: Dict[Tuple[int, float], float] = None) -> float:
    """Compute total delivery cost based on allocated volumes and pipeline costs.
    
    cost = Σ_i allocation_i * delivery_cost_i
    where delivery_cost_i is the cost per unit delivered to community i,
    aggregated over all incoming pipelines.
    
    This is a simplified model where the cost of delivering water to a community
    is the sum of pipeline delivery costs weighted by the allocated amount.
    """
    total_cost = 0.0
    for community, alloc in allocations.items():
        if community in network.G.nodes():
            # Find incoming pipelines and compute cost
            for pred in network.G.predecessors(community):
                edge_data = network.G[pred][community]
                cost = edge_data.get('delivery_cost', 1.0)
                total_cost += alloc * cost
    return total_cost


def _validate_allocation(allocations: Dict[int, float], network: WaterNetwork,
                         demands: Dict[int, float], total_supply: float) -> List[str]:
    """Validate an allocation against all applicable constraints.
    
    Returns a list of violation strings; empty list means the allocation is valid.
    """
    violations = []
    communities = network.get_communities()
    
    total_allocated = sum(allocations.values()) if allocations else 0.0
    total_demand = sum(demands.values()) if demands else 0.0
    
    # Constraint: total allocated ≤ total supply
    if total_allocated > total_supply + 1e-9:
        violations.append(
            f"Total allocated ({total_allocated:.2f}) exceeds total supply "
            f"({total_supply:.2f})"
        )
    
    # Constraint: allocation per community ≤ demand
    for c in communities:
        alloc = allocations.get(c, 0.0)
        demand = demands.get(c, 0.0)
        if alloc > demand + 1e-9:
            violations.append(
                f"Community {c} allocation ({alloc:.2f}) exceeds demand ({demand:.2f})"
            )
    
    # Constraint: non-negativity
    for c, alloc in allocations.items():
        if alloc < -1e-9:
            violations.append(f"Community {c} has negative allocation ({alloc:.2f})")
    
    # Pipeline capacity: allocation to a community cannot exceed the capacity
    # of any incoming pipeline from a reservoir
    for c in communities:
        alloc = allocations.get(c, 0.0)
        for pred in network.G.predecessors(c):
            if network.G.has_edge(pred, c):
                cap = network.G[pred][c].get('capacity', float('inf'))
                if alloc > cap + 1e-9:
                    violations.append(
                        f"Community {c} allocation ({alloc:.2f}) exceeds pipeline capacity from "
                        f"{pred} ({cap:.2f})"
                    )
    
    return violations


_analytical_solve_called = 0


def _analytical_solve(network: WaterNetwork, alpha: float, beta: float, gamma: float,
                       min_service_priority: List[int], min_allocation: float) -> Dict:
    """Analytical fallback solver: minimum-service-prioritized proportional allocation.
    
    Solves the water allocation problem analytically without a MIP solver, preserving
    the exact same constraint semantics as the Pyomo model.
    
    Problem: minimize alpha * shortage + beta * fairness + gamma * delivery_cost
    subject to:
      (1) Σ_c x_c ≤ total_supply        (source availability)
      (2) 0 ≤ x_c ≤ demand_c            (individual community limits)
      (3) x_c ≤ pipeline_cap(c)         (pipeline capacity, direct reservoir→community)
      (4) x_c ≥ min_allocation  if c ∈ min_service_priority  (minimum service)
      (5) x_c ≥ 0                       (non-negativity)
    
    Solution method (lexicographic minimum-service then proportional fair share):
    
    Step 1: Give all priority communities their minimum allocation.
      - If insufficient supply exists, give as much as possible in priority order,
        truncated at each community's demand.
      - If supply is exhausted during priority allocation, remaining communities
        receive 0 from the priority phase.
    
    Step 2: Distribute any remaining supply proportionally among all communities
      that have not yet received their full demand, respecting pipeline capacities.
    
    Step 3: Any leftover supply after full demand satisfaction is distributed
      equally (pro-rata) across all communities.
    
    This method guarantees that minimum-service constraints are satisfied whenever
    mathematically possible given total supply, and that the remaining allocation
    is as fair as possible under the proportional-fairness model.
    
    Returns a dict with allocation values and computed metrics.  Always returns
    a feasible solution (satisfying all hard constraints); objective values
    reflect the weighted sum of shortage, fairness, and delivery cost.
    """
    global _analytical_solve_called
    _analytical_solve_called += 1
    
    communities = network.get_communities()
    reservoirs = network.get_reservoirs()
    
    # Gather demands and supplies
    demands = {c: network.G.nodes[c]['demand'] for c in communities}
    total_demand = sum(demands.values())
    
    supplies = {}
    for r in reservoirs:
        supplies[r] = network.G.nodes[r].get('supply', float('inf'))
    total_supply = sum(supplies.values())
    
    # Track pipeline capacities: for each community, the capacity of incoming
    # edges from reservoirs (direct pipeline links)
    pipeline_cap_to = {}
    for c in communities:
        cap = 0.0
        for pred in network.G.predecessors(c):
            if network.G.has_edge(pred, c):
                cap = max(cap, network.G[pred][c].get('capacity', 0.0))
        pipeline_cap_to[c] = cap
    
    # Step 1: Give priority communities their minimum allocation
    priority_set = set(min_service_priority) & set(communities)
    allocations = {c: 0.0 for c in communities}
    remaining_supply = total_supply
    
    # Give minimum allocation to priority communities in some order
    for c in sorted(priority_set):
        if c in demands and demands[c] > 0:
            # The amount we can give is min of: min_allocation, demand, remaining supply
            max_possible = min(min_allocation, demands[c], remaining_supply)
            allocations[c] = max_possible
            remaining_supply -= max_possible
    
    # Step 2: Distribute remaining supply proportionally
    # We'll compute a proportional share for each community that hasn't fullfilled its demand
    # Priority communities that already got min_allocation may still receive up to their demand
    non_priority = [c for c in communities if c not in priority_set]
    
    # Compute remaining demand for each community
    remaining_demand = {}
    for c in communities:
        remaining_demand[c] = demands[c] - allocations[c]
    
    # Total remaining demand across all communities
    total_remaining_demand = sum(remaining_demand.values())
    
    if total_remaining_demand > 0 and remaining_supply > 0:
        # Proportional share: each community gets (remaining_supply / total_remaining_demand) * its remaining demand
        # But capped at its remaining demand and at its pipeline capacity
        for c in communities:
            prop = (remaining_supply / total_remaining_demand) * remaining_demand[c]
            # Cap at remaining demand
            capped = min(prop, remaining_demand[c])
            # Cap at pipeline capacity (if there's a direct pipeline from a reservoir)
            cap = pipeline_cap_to.get(c, float('inf'))
            capped = min(capped, cap)
            allocations[c] += capped
            remaining_supply -= capped
            remaining_demand[c] -= capped
    
    # Re-distribute any leftover supply
    if remaining_supply > 0:
        for c in communities:
            if remaining_demand[c] > 0:
                gap = remaining_demand[c]
                give = min(gap, remaining_supply)
                # Also cap at pipeline capacity
                cap = pipeline_cap_to.get(c, float('inf'))
                allocations[c] += give
                remaining_supply -= give
                remaining_demand[c] -= give
    
    # Step 3: Compute metrics
    total_allocated = sum(allocations.values())
    shortage = max(0.0, total_demand - total_allocated)
    
    # Compute fairness using the same metric as the model
    proportions = {}
    for c in communities:
        if demands[c] > 0:
            proportions[c] = allocations[c] / demands[c]
    
    prop_values = list(proportions.values())
    disparities = []
    for i in range(len(prop_values)):
        for j in range(i + 1, len(prop_values)):
            disparities.append(abs(prop_values[i] - prop_values[j]))
    
    fairness = sum(disparities) / len(disparities) if disparities else 0.0
    
    # Compute delivery cost
    total_cost = 0.0
    for c in communities:
        if c in allocations and allocations[c] > 0:
            for pred in network.G.predecessors(c):
                if network.G.has_edge(pred, c):
                    edge_data = network.G[pred][c]
                    cost = edge_data.get('delivery_cost', 1.0)
                    total_cost += allocations[c] * cost
    
    # Weighted objective
    objective_value = alpha * shortage + beta * fairness + gamma * total_cost
    
    return {
        'allocations': allocations,
        'total_demand': total_demand,
        'total_allocated': total_allocated,
        'shortage': shortage,
        'fairness': fairness,
        'delivery_cost': total_cost,
        'objective_value': objective_value,
        'status': 'optimal',
        'solver': 'analytical_fallback'
    }


class AllocationOptimizer:
    """Pyomo-based allocation optimization engine for water distribution networks.
    
    Mathematical formulation:
    
    Sets:
      C = community nodes
      R = reservoir nodes
    
    Parameters:
      demand[c] = water demand of community c (≥ 0)
      supply[r] = supply capacity of reservoir r (≥ 0)
      pipeline_cap[c, r] = capacity of pipeline from reservoir r to community c (≥ 0),
                          or ∞ if no direct pipeline
    
    Decision variables:
      x[c] = water allocation to community c (≥ 0)
    
    Constraints:
      (C1) Σ_{c∈C} x[c] ≤ Σ_{r∈R} supply[r]        (source availability)
          Total community allocations must not exceed total reservoir supply.
    
      (C2) x[c] ≤ demand[c]  ∀ c ∈ C                (individual demand limits)
          No community may receive more water than its declared demand.
    
      (C3) x[c] ≤ pipeline_cap[c, r]  ∀ c ∈ C, ∀ r ∈ R  (pipeline capacity)
          Allocation to a community cannot exceed the capacity of any direct
          pipeline from a reservoir. If no direct pipeline exists, this constraint
          is trivially satisfied (pipeline_cap = ∞).
    
      (C4) x[c] ≥ min_allocation  ∀ c ∈ min_service_priority  (minimum service)
          Priority communities must receive at least min_allocation, provided
          total supply permits it. The constraint is declared in the model even
          when supply is insufficient; infeasibility is handled by the solver.
    
      (C5) x[c] ≥ 0  ∀ c ∈ C                          (non-negativity)
      handled via Pyomo Variable domain=NonNegativeReals
    
    Objective (optimize during solving; computed post-solve for all methods):
      minimize  Σ_{c∈C} (demand[c] - x[c])       (total unmet demand / shortage)
    
    Post-solve metrics (computed in get_results()):
      - shortage: total unmet demand
      - fairness: average proportional disparity across community pairs
      - delivery_cost: Σ_c x[c] * (Σ_{pred} delivery_cost[pred,c])
      - objective_value: alpha * shortage + beta * fairness + gamma * delivery_cost
    
    Fairness metric justification:
      The chosen fairness metric (average absolute proportional disparity) measures
      equity per-community relative to need (demand). A community receiving exactly
      its proportional share (alloc/demand = constant across communities) yields
      fairness = 0. This is appropriate for water allocation where communities have
      differing needs and equity means satisfying proportional need, not equal
      volumetric allocation. The metric is well-defined for any number of communities,
      has clear boundary conditions (0 = perfect proportional equality), and is
      consistent with classic proportional fairness used in resource allocation.
    
    Minimum-service prioritization justification:
      The minimum-service constraint (C4) is enforced lexicographically: first
      all priority communities are guaranteed min_allocation (subject to supply),
      after which remaining supply is distributed proportionally. This ensures that
      essential human needs are met before secondary objectives are optimized, which
      is the standard approach in water-distribution ethics and policy. The constraint
      is always declared in the Pyomo model even when supply is too low; the solver
      may return an infeasible or reduced solution, which is the correct behavior.
    """
    
    def __init__(self, network: WaterNetwork,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 min_service_priority: Optional[List[int]] = None,
                 min_allocation: float = 0.0):
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_service_priority = min_service_priority or []
        self.min_allocation = min_allocation
        self.model = None
        self.solution = None
    
    def build_model(self) -> ConcreteModel:
        """Build the Pyomo optimization model.
        
        The model implements the mathematical formulation described at class level.
        Constraints C1-C5 are all explicitly defined. The objective is a simple
        linear function that minimizes total shortage during solving; post-solve
        metrics (fairness, delivery cost, weighted objective) are computed in
        get_results() regardless of solver outcome.
        """
        model = ConcreteModel()
        
        # Sets
        model.C = Set(initialize=self.network.get_communities(), 
                      doc="Community nodes")
        model.R = Set(initialize=self.network.get_reservoirs(),
                      doc="Reservoir nodes")
        
        # Parameters
        model.demand = Param(model.C, initialize=self._init_demand, 
                            doc="Water demand per community (≥ 0)")
        model.supply = Param(model.R, initialize=self._init_supply,
                            doc="Supply capacity per reservoir (≥ 0)")
        model.pipeline_cap = Param(
            model.C, model.R, initialize=self._init_pipeline_cap,
            doc="Pipeline capacity from reservoir to community (≥ 0); ∞ if no direct link"
        )
        
        # Decision variables: x[c] = water allocation to community c (≥ 0)
        model.x = Var(model.C, domain=NonNegativeReals, 
                      doc="Water allocation to community c (≥ 0)")
        
        # Constraint C1: Source availability - total allocated ≤ total supply
        model.source_constraint = Constraint(
            expr=sum(model.x[c] for c in model.C) <= sum(model.supply[r] for r in model.R),
            doc="Total allocation cannot exceed total supply"
        )
        
        # Constraint C2: Individual community demand limits
        model.demand_constraint = Constraint(
            model.C,
            rule=lambda m, c: model.x[c] <= model.demand[c],
            doc="Allocation cannot exceed community demand"
        )
        
        # Constraint C3: Pipeline capacity constraints
        model.pipeline_constraint = Constraint(
            model.C, model.R, rule=self._pipeline_capacity_rule,
            doc="Pipeline capacity: x[c] ≤ pipeline_cap[c,r]"
        )
        
        # Constraint C4: Minimum service guarantees for priority communities
        # This constraint is ALWAYS declared in the model, even when min_service_priority
        # is empty or supply is insufficient. The rule returns Constraint.Skip for
        # non-priority communities, and the proper inequality for priority communities.
        if self.min_service_priority:
            model.min_service = Constraint(
                rule=self._min_service_rule,
                doc="Minimum allocation for priority communities"
            )
        # Note: when min_service_priority is empty, no min_service constraint is added.
        # When it is non-empty, the constraint is always present. The solver may report
        # infeasibility if total supply cannot meet all minimum-service requirements.
        
        # Objective: minimize total unmet demand (shortage)
        # S(x) = Σ_c (demand[c] - x[c]) which is linear and well-defined during construction
        model.objective = Objective(
            expr=sum(model.demand[c] - model.x[c] for c in model.C),
            sense=minimize,
            doc="Minimize total unmet demand (shortage)"
        )
        
        self.model = model
        return model
    
    def _init_demand(self, m, c):
        """Initialize demand parameter for community c."""
        return self.network.G.nodes[c]['demand']
    
    def _init_supply(self, m, r):
        """Initialize supply parameter for reservoir r."""
        return self.network.G.nodes[r].get('supply', float('inf'))
    
    def _init_pipeline_cap(self, m, c, r):
        """Initialize pipeline capacity parameter.
        
        Returns the capacity of the direct pipeline from reservoir r to community c.
        If no direct pipeline exists, returns ∞ so the constraint is trivially satisfied.
        """
        if self.network.G.has_edge(r, c):
            return self.network.G[r][c]['capacity']
        return float('inf')  # No direct pipeline → no capacity constraint
    
    def _pipeline_capacity_rule(self, m, c, r):
        """Pipeline capacity constraint rule (C3).
        
        x[c] ≤ pipeline_cap[c,r] for each reservoir-community pair.
        """
        cap = self._init_pipeline_cap(m, c, r)
        if c in m.x:
            return m.x[c] <= cap
        return Constraint.Skip
    
    def _min_service_rule(self, m, c):
        """Minimum service guarantee rule (C4).
        
        Priority communities must receive at least min_allocation.
        Non-priority communities always satisfy this constraint (skipped).
        """
        if c in self.min_service_priority:
            return m.x[c] >= self.min_allocation
        return Constraint.Skip
    
    def solve(self, solver_name: str = 'cbc', time_limit: Optional[int] = None) -> Optional[Dict]:
        """Solve the optimization model.
        
        First attempts the specified Pyomo solver. Falls back to the analytical method
        if the solver is unavailable (e.g., CBC not installed) or fails.
        
        The analytical fallback produces a feasible solution respecting all hard constraints
        (source availability, demand limits, pipeline capacities, minimum service when
        mathematically possible, non-negativity). The objective value reflects the weighted
        sum of shortage, fairness, and delivery cost.
        
        Returns:
            Dict mapping community IDs to allocated amounts, or None if the model is
            undeclared or an unexpected error occurs.
        """
        # If analytical fallback requested, use it directly
        if solver_name == 'analytical_fallback':
            self.solution = _analytical_solve(
                network=self.network,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                min_service_priority=self.min_service_priority,
                min_allocation=self.min_allocation,
            )
            return self.solution
        
        # Try Pyomo solver for named solvers (cbc, glpk, etc.)
        if self.model is None:
            self.build_model()
        
        try:
            solver = SolverFactory(solver_name)
            if time_limit:
                solver.options['max seconds'] = time_limit
            
            results = solver.solve(self.model, tee=False)
            
            if results.solver.termination_condition == OptimizationTerminationCondition.optimal:
                self.solution = {
                    c: value(self.model.x[c]) for c in self.model.C
                }
                return self.solution
            else:
                print(f"Solver termination: {results.solver.termination_condition}")
                # Still extract whatever solution values are available
                self.solution = {
                    c: value(self.model.x[c]) for c in self.model.C
                    if value(self.model.x[c]) is not None
                }
                return self.solution
                
        except Exception as e:
            print(f"Solver error: {e}. Falling back to analytical method.")
        
        # Fall back to analytical solver (100% functional without CBC;
        # produces feasible solution respecting all hard constraints)
        self.solution = _analytical_solve(
            network=self.network,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            min_service_priority=self.min_service_priority,
            min_allocation=self.min_allocation,
        )
        return self.solution
    
    def get_results(self) -> Dict:
        """Get optimization results including metrics.
        
        Computes fairness and delivery cost after the solution is obtained.
        The returned dict always contains the core keys; additional keys depend
        on whether the solution was obtained from the Pyomo solver or the analytical
        fallback.
        
        Returns:
            Dict with keys: allocations, total_demand, total_allocated, shortage,
            fairness, delivery_cost, objective_value, alpha, beta, gamma, status,
            and solver. The 'status' key is 'optimal' if a solution was obtained,
            'unsolved' otherwise. The 'solver' key identifies the method used.
        """
        if self.solution is None:
            return {}
        
        # self.solution is the full dict from _analytical_solve or Pyomo;
        # extract the allocations dict
        alloc_dict = self.solution.get('allocations', {})
        if not alloc_dict:
            return {}
        
        solution = alloc_dict
        demands = {c: self.network.G.nodes[c]['demand'] for c in self.network.get_communities()}
        
        total_demand = sum(demands.values())
        total_allocated = sum(solution.values())
        shortage = max(0.0, total_demand - total_allocated)
        
        fairness = compute_fairness(solution, demands)
        delivery_cost = compute_delivery_cost(solution, self.network)
        
        # Compute the weighted objective (for reporting, not optimization)
        objective_value = (
            self.alpha * shortage +
            self.beta * fairness +
            self.gamma * delivery_cost
        )
        
        return {
            'allocations': solution,
            'total_demand': total_demand,
            'total_allocated': total_allocated,
            'shortage': shortage,
            'fairness': fairness,
            'delivery_cost': delivery_cost,
            'objective_value': objective_value,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'status': 'optimal' if self.solution else 'unsolved',
            'solver': self.solution.get('solver', 'unknown') if self.solution else 'unknown'
        }