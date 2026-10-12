from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from .optimizer import WaterNetwork, AllocationOptimizer, compute_shortage, compute_fairness, compute_delivery_cost
from .uncertainty import ScenarioGenerator, FailureAnalyzer


class EvaluationSuite:
    """Compares optimization approaches across standardized scenarios.
    
    Evaluates and reports: demand satisfaction, shortage rate, fairness,
    failure resilience, computational runtime, and feasibility.
    """
    
    def __init__(self, network: WaterNetwork,
                 approaches: Optional[List[Tuple[str, Dict]]] = None,
                 scenarios: Optional[List[Dict]] = None):
        self.network = network
        self.approaches = approaches or self._default_approaches()
        self.scenarios = scenarios or self._default_scenarios()
        self.results: Dict[str, List[Dict]] = {}
    
    def _default_approaches(self) -> List[Tuple[str, Dict]]:
        """Define default comparison baselines.
        
        Baseline A: Standard shortage-minimizing optimization
        Baseline B: Demand-priority allocation  
        Baseline C: Fairness-aware optimization
        AquaRoute candidate: Minimum-service-weighted optimization
        """
        return [
            # Baseline A: Standard shortage-minimizing optimization
            ("baseline_A_shortage_minimizing", {
                'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0,
                'min_service_priority': [],
                'min_allocation': 0.0
            }),
            # Baseline B: Demand-priority allocation
            ("baseline_B_demand_priority", {
                'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0,
                'min_service_priority': [],
                'min_allocation': 0.0
            }),
            # Baseline C: Fairness-aware optimization
            ("baseline_C_fairness_aware", {
                'alpha': 0.5, 'beta': 2.0, 'gamma': 1.0,
                'min_service_priority': [],
                'min_allocation': 0.0
            }),
            # AquaRoute candidate: Minimum-service-weighted optimization
            ("aquaroute_min_service", {
                'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0,
                'min_service_priority': [],
                'min_allocation': 0.0
            }),
        ]
    
    def _default_scenarios(self) -> List[Dict]:
        """Default set of evaluation scenarios."""
        return [
            ScenarioGenerator.normal_scenario(self.network),
            ScenarioGenerator.demand_spike_scenario(self.network, 1.5),
            ScenarioGenerator.supply_reduction_scenario(self.network, 0.5),
            ScenarioGenerator.pipeline_failure_scenario(self.network, [(0, 2), (1, 5)]),
        ]
    
    def _copy_network(self) -> WaterNetwork:
        """Create a copy of the network."""
        net = WaterNetwork()
        net.G = self.network.G.copy()
        # Copy node attributes
        for n in net.G.nodes():
            net.G.nodes[n].update(self.network.G.nodes[n])
        return net
    
    def _apply_scenario(self, net: WaterNetwork, scenario: Dict) -> None:
        """Apply scenario modifications to a network copy."""
        demands = scenario['demands']
        for c, d in demands.items():
            if net.G.has_node(c):
                net.G.nodes[c]['demand'] = d
        
        supply = scenario['supply']
        for r, s in supply.items():
            if net.G.has_node(r):
                net.G.nodes[r]['supply'] = s
        
        failures = scenario.get('failures', [])
        for u, v in failures:
            if net.G.has_edge(u, v):
                net.G.remove_edge(u, v)
    
    def run_comparison(self, n_repeats: int = 1) -> Dict[str, List[Dict]]:
        """Run all approaches across all scenarios and collect results.
        
        n_repeats: number of times to repeat each scenario (for statistical analysis).
        Results are averaged across repeats.
        """
        all_results = {}
        
        for approach_name, approach_kwargs in self.approaches:
            approach_results = []
            
            for scenario in self.scenarios:
                # Repeat the scenario n_repeats times
                repeat_results = []
                for _ in range(n_repeats):
                    # Create fresh network copy
                    net_copy = self._copy_network()
                    
                    # Apply scenario
                    self._apply_scenario(net_copy, scenario)
                    
                    # Solve optimization
                    approach_kw = dict(approach_kwargs)
                    # Set min_service_priority based on scenario if specified
                    if 'min_service_priority' not in approach_kw:
                        approach_kw['min_service_priority'] = []
                    
                    optimizer = AllocationOptimizer(network=net_copy, **approach_kw)
                    solution = optimizer.solve()
                    
                    result = {
                        'approach': approach_name,
                        'scenario_index': scenario.get('scenario_type', 'unknown'),
                        'allocations': solution or {},
                        'scenario': scenario,
                        'results': optimizer.get_results() if solution else {},
                    }
                    repeat_results.append(result)
                
                # Average the results across repeats
                averaged = self._average_results(repeat_results)
                approach_results.append(averaged)
            
            all_results[approach_name] = approach_results
            self.results[approach_name] = approach_results
        
        return all_results
    
    def _average_results(self, results_list: List[Dict]) -> Dict:
        """Average results across multiple repeats of the same scenario."""
        if not results_list:
            return {}
        
        # Get the keys from the first result
        first = results_list[0]
        if not isinstance(first, dict):
            return {}
        
        result = {}
        for key in first.keys():
            values = [r.get(key, np.nan) for r in results_list if isinstance(r.get(key), (int, float))]
            if values:
                result[key] = np.mean(values)
            else:
                result[key] = results_list[0].get(key)
        
        return result
    
    def compute_summary_metrics(self) -> Dict:
        """Compute summary comparison metrics across all approaches and scenarios."""
        summary = {}
        
        for approach_name, approach_results in self.results.items():
            if not approach_results:
                continue
            
            # Extract key metrics per scenario
            shortages = [r['results']['shortage'] for r in approach_results if r['results']]
            fairness_vals = [r['results']['fairness'] for r in approach_results if r['results']]
            costs = [r['results']['delivery_cost'] for r in approach_results if r['results']]
            runtimes = [r['results'].get('runtime', 0) for r in approach_results if r['results']]
            
            summary[approach_name] = {
                'avg_shortage': np.mean(shortages) if shortages else 0,
                'worst_shortage': max(shortages) if shortages else 0,
                'avg_fairness': np.mean(fairness_vals) if fairness_vals else 0,
                'worst_fairness': max(fairness_vals) if fairness_vals else 0,
                'avg_cost': np.mean(costs) if costs else 0,
                'worst_cost': max(costs) if costs else 0,
                'avg_runtime': np.mean(runtimes) if runtimes else 0,
                'std_shortage': np.std(shortages) if len(shortages) > 1 else 0,
                'std_fairness': np.std(fairness_vals) if len(fairness_vals) > 1 else 0,
                'std_cost': np.std(costs) if len(costs) > 1 else 0,
                'std_runtime': np.std(runtimes) if len(runtimes) > 1 else 0,
                'feasibility_rate': sum(1 for r in approach_results if r['results'].get('shortage', 0) == 0) / len(approach_results),
                'scenario_count': len(approach_results),
            }
        
        return summary
    
    def statistical_significance(self, metric: str = 'shortage',
                                   approach_a: str = None, approach_b: str = None) -> Optional[Dict]:
        """Perform paired t-test comparing two approaches on a given metric.
        
        Returns dict with t-statistic, p-value, and interpretation.
        """
        if approach_a not in self.results or approach_b not in self.results:
            return None
        
        results_a = self.results[approach_a]
        results_b = self.results[approach_b]
        
        # Match by scenario index
        n = min(len(results_a), len(results_b))
        if n < 2:
            return None
        
        metric_a = [results_a[i]['results'].get(metric, 0) for i in range(n)]
        metric_b = [results_b[i]['results'].get(metric, 0) for i in range(n)]
        
        # Paired t-test
        diffs = [a - b for a, b in zip(metric_a, metric_b)]
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0
        
        if std_diff == 0:
            p_value = 1.0
            t_stat = 0.0 if mean_diff == 0 else float('inf')
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            # Two-sided p-value approximation
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': mean_diff,
            'significant': p_value < 0.05,
            'interpretation': f"{approach_a} {'better' if mean_diff < 0 else 'worse'} than {approach_b}" if p_value < 0.05 else "No significant difference"
        }
    
    def sensitivity_analysis(self, metric: str = 'shortage',
                             approach: str = None, param: str = 'alpha',
                             values: Optional[List[float]] = None) -> Dict:
        """Analyze how varying a parameter affects the given metric.
        
        Parameters vary the specified optimizer parameter (alpha, beta, gamma)
        and report the resulting metric values.
        """
        if approach not in self.results:
            return {}
        
        if values is None:
            # Default range based on the approach's current value
            matching = [kwargs for name, kwargs in self.approaches if name == approach]
            if not matching:
                return {}
            approach_kw = matching[0]
            base_val = approach_kw.get(param, 1.0)
            values = [base_val * v for v in [0.5, 1.0, 1.5, 2.0]]
        
        results = {}
        for val in values:
            # Temporarily set the parameter
            matching = [kwargs for name, kwargs in self.approaches if name == approach]
            if not matching:
                return {}
            approach_kw = dict(matching[0])
            approach_kw[param] = float(val)
            
            # Run a single scenario to evaluate
            scenario = self.scenarios[0] if self.scenarios else {}
            net_copy = self._copy_network()
            self._apply_scenario(net_copy, scenario)
            
            optimizer = AllocationOptimizer(network=net_copy, **approach_kw)
            solution = optimizer.solve()
            results_val = optimizer.get_results() if solution else {}
            
            results[float(val)] = results_val.get(metric, float('nan'))
        
        return results


def compare_approaches(network: WaterNetwork,
                       approaches: Optional[List[Tuple[str, Dict]]] = None,
                       scenarios: Optional[List[Dict]] = None) -> EvaluationSuite:
    """Convenience function to run a full comparison evaluation."""
    suite = EvaluationSuite(network, approaches=approaches, scenarios=scenarios)
    suite.run_comparison()
    return suite
