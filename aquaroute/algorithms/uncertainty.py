from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from .optimizer import WaterNetwork, AllocationOptimizer, compute_shortage, compute_fairness
import copy


class ScenarioGenerator:
    """Generates different demand/supply scenarios for robustness testing.
    
    Scenarios:
    1. Normal: baseline demand and supply
    2. Demand spike: elevated community water requirements
    3. Supply reduction: reduced reservoir capacity
    4. Pipeline failure: specific links become unavailable
    5. Multiple disruptions: combination of above
    """
    
    @staticmethod
    def normal_scenario(network: WaterNetwork) -> Dict:
        """Normal demand and supply scenario."""
        return {
            'demands': {c: network.G.nodes[c]['demand'] for c in network.get_communities()},
            'supply': {r: network.G.nodes[r].get('supply', float('inf')) 
                      for r in network.get_reservoirs()},
            'failures': []  # No pipeline failures
        }
    
    @staticmethod
    def demand_spike_scenario(network: WaterNetwork, spike_factor: float = 1.5) -> Dict:
        """Elevated demand scenario (e.g., during heat wave or agricultural season).
        
        Creates an independent copy of the network so the baseline is not mutated.
        """
        # Create an independent copy of the network
        net_copy = copy.copy(network)
        net_copy.G = copy.deepcopy(network.G)
        
        # Apply demand spike to the copy
        demands = {}
        for c in net_copy.get_communities():
            base_demand = net_copy.G.nodes[c]['demand']
            demands[c] = base_demand * spike_factor
        # Update copy node demands
        for c, d in demands.items():
            net_copy.G.nodes[c]['demand'] = d
        
        return {
            'demands': demands,
            'supply': {r: net_copy.G.nodes[r].get('supply', float('inf'))
                      for r in net_copy.get_reservoirs()},
            'failures': []
        }
    
    @staticmethod
    def supply_reduction_scenario(network: WaterNetwork, reduction_factor: float = 0.5) -> Dict:
        """Reduced water supply scenario (e.g., during drought).
        
        Creates an independent copy of the network so the baseline is not mutated.
        """
        # Create an independent copy of the network
        net_copy = copy.copy(network)
        net_copy.G = copy.deepcopy(network.G)
        
        # Apply supply reduction to the copy
        supply = {}
        for r in net_copy.get_reservoirs():
            base_supply = net_copy.G.nodes[r].get('supply', float('inf'))
            supply[r] = base_supply * reduction_factor
        # Update copy reservoir supplies
        for r, s in supply.items():
            net_copy.G.nodes[r]['supply'] = s
        
        return {
            'demands': {c: net_copy.G.nodes[c]['demand'] for c in net_copy.get_communities()},
            'supply': supply,
            'failures': []
        }
    
    @staticmethod
    def pipeline_failure_scenario(network: WaterNetwork, failed_edges: List[Tuple[int, int]]) -> Dict:
        """Specific pipeline links become unavailable.
        
        Creates a copy of the network graph so the baseline is not mutated.
        """
        net_copy = copy.copy(network)
        net_copy.G = network.G.copy()
        
        # Remove failed pipelines from the graph copy
        for u, v in failed_edges:
            if net_copy.G.has_edge(u, v):
                net_copy.G.remove_edge(u, v)
        
        return {
            'demands': {c: net_copy.G.nodes[c]['demand'] for c in net_copy.get_communities()},
            'supply': {r: net_copy.G.nodes[r].get('supply', float('inf'))
                      for r in net_copy.get_reservoirs()},
            'failures': failed_edges
        }
    
    @staticmethod
    def multiple_disruptions_scenario(network: WaterNetwork,
                                       spike_factor: float = 1.3,
                                       reduction_factor: float = 0.6,
                                       failed_edges: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """Combined scenario: demand spike + supply reduction + pipeline failures.
        
        Creates independent copies so the baseline network is never mutated.
        """
        # Start with demand spike on a copy
        spike_scenario = ScenarioGenerator.demand_spike_scenario(network, spike_factor)
        
        # Apply supply reduction on a copy (original network unchanged)
        reduction_scenario = ScenarioGenerator.supply_reduction_scenario(network, reduction_factor)
        spike_scenario['supply'] = reduction_scenario['supply']
        
        # Apply pipeline failures on a copy
        if failed_edges:
            pipeline_scenario = ScenarioGenerator.pipeline_failure_scenario(network, failed_edges)
            spike_scenario['failures'] = pipeline_scenario['failures']
        else:
            spike_scenario['failures'] = []
        
        # The copies already have their own network state; no need to reset
        # since each scenario works on its own copy
        return spike_scenario


class FailureAnalyzer:
    """Analyzes allocation resilience under network disruptions.
    
    Tests how allocation changes when pipelines fail or supply/demand changes,
    and evaluates the fairness and resilience of different optimization approaches.
    """
    
    @staticmethod
    def analyze_resilience(network: WaterNetwork, optimizer_class: type,
                           optimizer_kwargs: Dict,
                           scenarios: Optional[List[Dict]] = None) -> List[Dict]:
        """Run optimization across multiple scenarios and track metric changes.
        
        Tests how allocation changes when pipelines fail or supply/demand changes,
        and evaluates the fairness and resilience of different optimization approaches.
        
        Returns list of result dicts, one per scenario, enabling resilience comparison.
        Even scenarios where optimization fails are included (with None solution).
        """
        if scenarios is None:
            scenarios = [
                ScenarioGenerator.normal_scenario(network),
                ScenarioGenerator.demand_spike_scenario(network, 1.4),
                ScenarioGenerator.supply_reduction_scenario(network, 0.5),
                ScenarioGenerator.pipeline_failure_scenario(network, [(0, 2), (1, 5)]),
            ]
        
        results = []
        for i, scenario in enumerate(scenarios):
            # Create a fresh network copy for each scenario
            net_copy = WaterNetwork()
            net_copy.G = network.G.copy()
            
            # Apply scenario modifications
            demands = scenario['demands']
            for c, d in demands.items():
                net_copy.G.nodes[c]['demand'] = d
            
            supply = scenario['supply']
            for r, s in supply.items():
                net_copy.G.nodes[r]['supply'] = s
            
            failures = scenario.get('failures', [])
            for u, v in failures:
                if net_copy.G.has_edge(u, v):
                    net_copy.G.remove_edge(u, v)
            
            # Solve optimization (may fail if no solver available)
            opt_kwargs = dict(optimizer_kwargs)
            opt_kwargs['min_service_priority'] = opt_kwargs.get('min_service_priority', [])
            
            try:
                optimizer = optimizer_class(network=net_copy, **opt_kwargs)
                solution = optimizer.solve()
            except Exception as e:
                solution = None
            
            if solution is not None and optimizer.get_results():
                results.append({
                    'scenario_index': i,
                    'allocations': solution,
                    'demands': demands,
                    'supply': supply,
                    'failures': failures,
                    'results': optimizer.get_results(),
                    'scenario_type': scenario.get('scenario_type', f'scenario_{i}')
                })
            else:
                # Include scenario even if optimization fails
                results.append({
                    'scenario_index': i,
                    'allocations': solution or {},
                    'demands': demands,
                    'supply': supply,
                    'failures': failures,
                    'results': optimizer.get_results() if optimizer else {},
                    'scenario_type': scenario.get('scenario_type', f'scenario_{i}'),
                    'optimization_failed': solution is None
                })
        
        return results
    
    @staticmethod
    def compute_resilience_metrics(results: List[Dict]) -> Dict:
        """Compute summary resilience metrics from scenario results.
        
        Tracks how much metrics degrade across scenarios,
        especially focusing on fairness and shortage preservation.
        
        Returns dict with metrics, or empty dict if no results.
        """
        if not results:
            return {}
        
        metrics = {}
        
        # Track shortage across scenarios
        # Handle cases where results may not have 'shortage' key
        shortages = []
        for r in results:
            res = r.get('results', {})
            if isinstance(res, dict) and 'shortage' in res:
                shortages.append(res['shortage'])
        
        if shortages:
            metrics['shortage_range'] = max(shortages) - min(shortages)
            metrics['shortage_increase_from_normal'] = shortages[-1] - shortages[0] if len(shortages) > 1 else 0
        else:
            metrics['shortage_range'] = 0
            metrics['shortage_increase_from_normal'] = 0
        
        # Track fairness across scenarios
        fairness_vals = []
        for r in results:
            res = r.get('results', {})
            if isinstance(res, dict) and 'fairness' in res:
                fairness_vals.append(res['fairness'])
        
        if fairness_vals:
            metrics['fairness_range'] = max(fairness_vals) - min(fairness_vals)
            metrics['fairness_degradation'] = fairness_vals[-1] - fairness_vals[0] if len(fairness_vals) > 1 else 0
        else:
            metrics['fairness_range'] = 0
            metrics['fairness_degradation'] = 0
        
        # Track delivery cost across scenarios
        costs = []
        for r in results:
            res = r.get('results', {})
            if isinstance(res, dict) and 'delivery_cost' in res:
                costs.append(res['delivery_cost'])
        
        if costs:
            metrics['cost_range'] = max(costs) - min(costs)
        else:
            metrics['cost_range'] = 0
        
        # Feasibility: % of scenarios where all demands were met
        feasible_counts = 0
        total = 0
        for r in results:
            total += 1
            res = r.get('results', {})
            if isinstance(res, dict) and res.get('shortage') == 0:
                feasible_counts += 1
        
        if total > 0:
            metrics['feasibility_rate'] = feasible_counts / total
        else:
            metrics['feasibility_rate'] = 0
        
        return metrics


def create_stress_test_suite(network: WaterNetwork, 
                             n_communities: int = 5,
                             failure_probability: float = 0.3) -> List[Dict]:
    """Generate a stress test suite with random disruption combinations.
    
    Useful for evaluating algorithm robustness before deploying to real networks.
    """
    import itertools
    edges = list(network.G.edges())
    n_failures = max(1, int(len(edges) * failure_probability))
    
    suite = []
    # Generate several random failure combinations
    for _ in range(n_communities):
        # Randomly select failed edges
        np_random = __import__('numpy').random
        np_random.seed()
        failed = np_random.choice(len(edges), size=min(n_failures, len(edges)), replace=False)
        failed_edges = [edges[i] for i in failed]
        
        scenario = ScenarioGenerator.pipeline_failure_scenario(network, failed_edges)
        suite.append(scenario)
    
    return suite