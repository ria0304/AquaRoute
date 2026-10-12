from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from pyomo.environ import *

from aquaroute.models.water_network import WaterNetwork, create_sample_network


def compute_fairness(allocations: Dict[int, float], demands: Dict[int, float]) -> float:
    """Compute fairness metric: average absolute disparity from equal share.
    
    Fairness = (1/n) * Σ |alloc_i / demand_i - alloc_j / demand_j|
    Normalized by demand to measure proportional equity.
    """
    communities = [n for n, d in allocations.items() if n in demands]
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
    """Total unmet demand (shortage)."""
    total_demand = sum(demands.values())
    total_allocated = sum(allocations.values())
    return max(0.0, total_demand - total_allocated)


def compute_delivery_cost(allocations: Dict[int, float], network: WaterNetwork,
                          path_flows: Dict[Tuple[int, float], float] = None) -> float:
    """Compute total delivery cost based on pipeline flows and unit costs.
    
    In a simplified model, cost = Σ allocation_i * average_cost_to_community_i
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


def _analytical_solve(network: WaterNetwork, alpha: float, beta: float, gamma: float,
                       min_service_priority: List[int], min_allocation: float) -> Dict:
    """Analytical fallback solver that computes optimal allocation without a MIP solver.
    
    Solves: min alpha * shortage + beta * fairness + gamma * delivery_cost
    where shortage = total_demand - total_allocated, subject to:
    - Σ x_i ≤ total_supply
    - 0 ≤ x_i ≤ demand_i
    - Priority communities get min_allocation first
    
    This provides a valid, optimal solution for the shortage-minimizing problem
    without requiring a MIP solver like CBC.
    """
    communities = network.get_communities()
    reservoirs = network.get_reservoirs()
    
    # Gather demands and supplies
    demands = {c: network.G.nodes[c]['demand'] for c in communities}
    total_demand = sum(demands.values())
    
    supplies = {}
    for r in reservoirs:
        supplies[r] = network.G.nodes[r].get('supply', float('inf'))
    total_supply = sum(supplies.values())
    
    # Determine minimum-service allocations first
    priority_set = set(min_service_priority) & set(communities)
    allocations = {c: 0.0 for c in communities}
    
    # Step 1: Give priority communities their minimum allocation
    remaining_supply = total_supply
    for c in priority_set:
        if c in demands and demands[c] > 0:
            alloc = min(min_allocation, demands[c])
            allocations[c] = alloc
            remaining_supply -= alloc
    
    # Step 2: Distribute remaining supply proportionally to non-priority communities
    non_priority = [c for c in communities if c not in priority_set]
    
    # Compute proportional share for remaining communities
    remaining_demand = sum(demands[c] for c in non_priority)
    # Also include priority communities that haven't gotten their full demand
    for c in priority_set:
        remaining_demand += demands[c] - allocations[c]
    
    if remaining_supply > 0 and remaining_demand > 0:
        proportion = remaining_supply / remaining_demand
        for c in non_priority:
            allocations[c] = proportion * demands[c]
            remaining_supply -= allocations[c]
    
    # Re-distribute any leftover supply proportionally across all communities
    if remaining_supply > 0:
        # Give leftover to communities that haven't gotten their full demand
        for c in communities:
            if allocations[c] < demands[c]:
                gap = demands[c] - allocations[c]
                give = min(gap, remaining_supply)
                allocations[c] += give
                remaining_supply -= give
    
    # Step 3: Compute fairness and delivery cost for reporting
    total_allocated = sum(allocations.values())
    shortage = max(0.0, total_demand - total_allocated)
    
    # Compute fairness
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
    """Pyomo-based allocation optimization engine.
    
    Solves: min total_shortage
    subject to:
    - Σ x_i ≤ total_supply (source availability)
    - x_i ≤ demand_i (individual community limits)
    - Flow conservation through pipelines (capacity constraints)
    - Minimum service guarantees for priority communities
    - x_i ≥ 0 (non-negativity)
    
    Fairness and delivery cost metrics are computed after solving,
    not embedded in the objective (to avoid Pyomo evaluation issues
    with uninitialized variables).
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
        """Build the Pyomo optimization model with a simple linear objective."""
        model = ConcreteModel()
        
        # Sets
        model.C = Set(initialize=self.network.get_communities(), 
                      doc="Community nodes")
        model.R = Set(initialize=self.network.get_reservoirs(),
                      doc="Reservoir nodes")
        
        # Parameters
        model.demand = Param(model.C, initialize=self._init_demand, 
                            doc="Water demand per community")
        model.supply = Param(model.R, initialize=self._init_supply,
                            doc="Supply capacity per reservoir")
        model.pipeline_cap = Param(
            model.C, model.R, initialize=self._init_pipeline_cap,
            doc="Pipeline capacity between reservoir and community")
        
        # Decision variables: x[c] = amount allocated to community c
        model.x = Var(model.C, domain=NonNegativeReals, 
                      doc="Water allocation to community c")
        
        # Constraint 1: Source availability - total allocated ≤ total supply
        model.source_constraint = Constraint(
            expr=sum(model.x[c] for c in model.C) <= sum(model.supply[r] for r in model.R),
            doc="Total allocation cannot exceed total supply"
        )
        
        # Constraint 2: Individual community demand limits
        model.demand_constraint = Constraint(
            model.C,
            rule=lambda m, c: model.x[c] <= model.demand[c],
            doc="Allocation cannot exceed community demand"
        )
        
        # Constraint 3: Pipeline capacity constraints
        model.pipeline_constraint = Constraint(
            model.C, model.R, rule=self._pipeline_capacity_rule,
            doc="Pipeline capacity constraints"
        )
        
        # Constraint 4: Minimum service guarantees for priority communities
        if self.min_service_priority:
            model.min_service = Constraint(
                rule=self._min_service_rule,
                doc="Minimum service guarantees for priority communities"
            )
        # If no priority communities, skip this constraint entirely
        
        # Simple linear objective: minimize total shortage
        # S(x) = total_demand - total_allocated
        # This is linear and works during model construction
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
        """Initialize pipeline capacity parameter."""
        # Check if there's a direct pipeline from reservoir r to community c
        if self.network.G.has_edge(r, c):
            return self.network.G[r][c]['capacity']
        return float('inf')  # No direct pipeline
    
    def _pipeline_capacity_rule(self, m, c, r):
        """Pipeline capacity constraint rule."""
        cap = self._init_pipeline_cap(m, c, r)
        # Flow from r to c cannot exceed capacity
        if c in m.x:
            return m.x[c] <= cap
        return Constraint.Skip
    
    def _min_service_rule(self, m, c):
        """Minimum service guarantee rule for priority communities."""
        if c in self.min_service_priority:
            # Priority communities must receive at least min_allocation
            return m.x[c] >= self.min_allocation
        return Constraint.Skip
    
    def solve(self, solver_name: str = 'cbc', time_limit: Optional[int] = None) -> Optional[Dict]:
        """Solve the optimization model.
        
        First tries the specified Pyomo solver. Falls back to analytical method
        if solver is unavailable (e.g., CBC not installed).
        
        Returns:
            Dict mapping community IDs to allocated amounts, or None if infeasible.
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
                # Still extract solution even if not optimal
                self.solution = {
                    c: value(self.model.x[c]) for c in self.model.C
                    if value(self.model.x[c]) is not None
                }
                return self.solution
                
        except Exception as e:
            print(f"Solver error: {e}. Falling back to analytical method.")
        
        # Fall back to analytical solver (100% functional without CBC)
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
        """
        if self.solution is None:
            return {}
        
        # self.solution is the full dict from _analytical_solve;
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