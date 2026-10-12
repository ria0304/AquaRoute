from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from aquaroute.models.water_network import WaterNetwork


def apply_scenario(network: WaterNetwork, scenario: Dict) -> None:
    """Mutate ``network`` in place according to a scenario description."""
    for c, d in scenario.get("demands", {}).items():
        if network.G.has_node(c):
            network.G.nodes[c]["demand"] = d
    for r, s in scenario.get("supply", {}).items():
        if network.G.has_node(r):
            network.G.nodes[r]["supply"] = s
    for u, v in scenario.get("failures", []):
        if network.G.has_edge(u, v):
            network.G.remove_edge(u, v)


def _snapshot(network: WaterNetwork, scenario_type: str, failures: List,
              demand_scale: float = 1.0, supply_scale: float = 1.0) -> Dict:
    return {
        "demands": {c: network.G.nodes[c]["demand"] * demand_scale
                    for c in network.get_communities()},
        "supply": {r: network.G.nodes[r].get("supply", float("inf")) * supply_scale
                   for r in network.get_reservoirs()},
        "failures": list(failures),
        "scenario_type": scenario_type,
    }


class ScenarioGenerator:
    """Deterministic generators for demand/supply/failure scenarios."""

    @staticmethod
    def normal_scenario(network: WaterNetwork) -> Dict:
        return _snapshot(network, "normal", [])

    @staticmethod
    def demand_spike_scenario(network: WaterNetwork, spike_factor: float = 1.5) -> Dict:
        return _snapshot(network, "demand_spike", [], demand_scale=spike_factor)

    @staticmethod
    def supply_reduction_scenario(network: WaterNetwork, reduction_factor: float = 0.5) -> Dict:
        return _snapshot(network, "supply_reduction", [], supply_scale=reduction_factor)

    @staticmethod
    def pipeline_failure_scenario(network: WaterNetwork,
                                  failed_edges: List[Tuple[int, int]]) -> Dict:
        existing = [(u, v) for u, v in failed_edges if network.G.has_edge(u, v)]
        return _snapshot(network, "pipeline_failure", existing)

    @staticmethod
    def multiple_disruptions_scenario(network: WaterNetwork,
                                      spike_factor: float = 1.3,
                                      reduction_factor: float = 0.6,
                                      failed_edges: Optional[List[Tuple[int, int]]] = None) -> Dict:
        existing = [(u, v) for u, v in (failed_edges or []) if network.G.has_edge(u, v)]
        return _snapshot(network, "multiple_disruptions", existing,
                         demand_scale=spike_factor, supply_scale=reduction_factor)


class FailureAnalyzer:
    """Analyses allocation resilience under network disruptions."""

    @staticmethod
    def analyze_resilience(network: WaterNetwork, optimizer_class: type,
                           optimizer_kwargs: Dict,
                           scenarios: Optional[List[Dict]] = None) -> List[Dict]:
        if scenarios is None:
            scenarios = [
                ScenarioGenerator.normal_scenario(network),
                ScenarioGenerator.demand_spike_scenario(network, 1.4),
                ScenarioGenerator.supply_reduction_scenario(network, 0.5),
                ScenarioGenerator.pipeline_failure_scenario(network, [(0, 2), (1, 5)]),
            ]

        results = []
        for i, scenario in enumerate(scenarios):
            net_copy = WaterNetwork()
            net_copy.G = copy.deepcopy(network.G)
            apply_scenario(net_copy, scenario)

            optimizer = None
            try:
                optimizer = optimizer_class(network=net_copy, **optimizer_kwargs)
                solution = optimizer.solve()
            except Exception:
                solution = None

            results.append({
                "scenario_index": i,
                "scenario_type": scenario.get("scenario_type", f"scenario_{i}"),
                "allocations": (solution.get("allocations", {}) if solution else {}),
                "demands": dict(scenario.get("demands", {})),
                "supply": dict(scenario.get("supply", {})),
                "failures": scenario.get("failures", []),
                "results": optimizer.get_results() if (optimizer and solution) else {},
                "optimization_failed": solution is None,
            })
        return results

    @staticmethod
    def compute_resilience_metrics(results: List[Dict]) -> Dict:
        if not results:
            return {}

        def _series(key: str) -> List[float]:
            return [r["results"][key] for r in results
                    if isinstance(r.get("results"), dict) and key in r["results"]]

        metrics: Dict[str, float] = {}
        shortages = _series("shortage")
        fairness_vals = _series("fairness")
        costs = _series("delivery_cost")

        metrics["shortage_range"] = (max(shortages) - min(shortages)) if shortages else 0.0
        metrics["shortage_increase_from_normal"] = (shortages[-1] - shortages[0]) if len(shortages) > 1 else 0.0
        metrics["fairness_range"] = (max(fairness_vals) - min(fairness_vals)) if fairness_vals else 0.0
        metrics["fairness_degradation"] = (fairness_vals[-1] - fairness_vals[0]) if len(fairness_vals) > 1 else 0.0
        metrics["cost_range"] = (max(costs) - min(costs)) if costs else 0.0
        feasible = sum(1 for r in results
                       if isinstance(r.get("results"), dict) and r["results"].get("shortage") == 0)
        metrics["feasibility_rate"] = feasible / len(results)
        return metrics


def create_stress_test_suite(network: WaterNetwork,
                             n_communities: int = 5,
                             failure_probability: float = 0.3,
                             seed: Optional[int] = None) -> List[Dict]:
    """Generate reproducible random pipeline-failure scenarios."""
    rng = np.random.default_rng(seed)
    edges = list(network.G.edges())
    if not edges:
        return []
    n_failures = max(1, int(len(edges) * failure_probability))

    suite = []
    for _ in range(n_communities):
        n = min(n_failures, len(edges))
        idx = rng.choice(len(edges), size=n, replace=False)
        failed_edges = [edges[i] for i in idx]
        suite.append(ScenarioGenerator.pipeline_failure_scenario(network, failed_edges))
    return suite
