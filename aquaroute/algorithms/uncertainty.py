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
    def optimize_recovery(network: WaterNetwork, optimizer_class: type,
                          optimizer_kwargs: Dict,
                          failed_edges: List[Tuple[int, int]],
                          n_recovery_scenarios: int = 10) -> Dict:
        """Determine optimal recovery strategy after infrastructure failures.

        Two-stage recovery:
        1. Pre-failure: determine which infrastructure to reinforce
        2. Post-failure: reallocate water through surviving infrastructure

        Parameters
        ----------
        network : WaterNetwork
            Original network state.
        optimizer_class : type
            Optimizer class to use for reallocation.
        optimizer_kwargs : dict
            Keyword arguments for the optimizer.
        failed_edges : list of tuple
            Edges that have failed (u, v).
        n_recovery_scenarios : int, optional
            Number of stochastic scenarios for recovery analysis. Default 10.

        Returns
        -------
        dict
            Recovery analysis with keys:
            - ``pre_failure_allocations``: allocations before failure
            - ``post_failure_allocations``: allocations after failure reoptimization
            - ``recovery_improvement``: improvement in allocation coverage
            - ``affected_communities``: communities impacted by failure
            - ``recovery_scenarios``: per-scenario results
            - ``recommendations``: infrastructure repair recommendations
        """
        import numpy as np

        # Stage 1: Record pre-failure allocations
        opt_original = optimizer_class(network=network, **optimizer_kwargs)
        opt_original.solve()
        pre_failure = opt_original.get_results() if opt_original.solution else {}

        # Stage 2: Apply failures and reoptimize
        net_copy = WaterNetwork()
        net_copy.G = copy.deepcopy(network.G)

        # Remove failed edges
        for u, v in failed_edges:
            if net_copy.G.has_edge(u, v):
                net_copy.G.remove_edge(u, v)

        # Record which communities are affected (lost direct connection)
        affected_communities = []
        for c in net_copy.get_communities():
            # Check if community has any incoming pipes
            incoming = [p for p in net_copy.G.predecessors(c)
                       if net_copy.G.has_edge(p, c)]
            if not incoming:
                affected_communities.append(c)

        # Reoptimize allocation for post-failure state
        opt_post = optimizer_class(network=net_copy, **optimizer_kwargs)
        try:
            solution_post = opt_post.solve(allow_fallback=True)
            post_failure = opt_post.get_results() if opt_post.solution else {}
        except Exception:
            post_failure = {}

        # Run multiple recovery scenarios with demand/supply variation
        recovery_scenarios = []
        base_demands = {c: net_copy.G.nodes[c]["demand"]
                       for c in net_copy.get_communities()}
        base_supply = {r: net_copy.G.nodes[r].get("supply", float("inf"))
                      for r in net_copy.get_reservoirs()}

        for i in range(n_recovery_scenarios):
            # Vary demands and supply for stochastic analysis
            scenario_demands = {c: base_demands.get(c, 0.0) * np.random.uniform(0.7, 1.3)
                               for c in net_copy.get_communities()}
            scenario_supply = {r: base_supply.get(r, float("inf")) * np.random.uniform(0.7, 1.3)
                              for r in net_copy.get_reservoirs()}

            # Update network with scenario parameters
            for c in net_copy.get_communities():
                if net_copy.G.has_node(c):
                    net_copy.G.nodes[c]["demand"] = scenario_demands[c]
            for r in net_copy.get_reservoirs():
                if net_copy.G.has_node(r):
                    net_copy.G.nodes[r]["supply"] = scenario_supply[r]

            # Re-solve
            opt_rec = optimizer_class(network=net_copy,
                                      **optimizer_kwargs)
            try:
                sol_rec = opt_rec.solve(allow_fallback=True)
                recovery_scenarios.append({
                    "scenario_index": i,
                    "allocations": sol_rec.get("allocations", {}) if sol_rec.solution else {},
                    "shortage": sol_rec.get("shortage", float("nan")) if sol_rec.solution else float("nan"),
                    "cvar": sol_rec.get("cvar", 0.0) if sol_rec.solution else 0.0,
                })
            except Exception:
                recovery_scenarios.append({
                    "scenario_index": i,
                    "allocations": {},
                    "shortage": float("nan"),
                    "cvar": 0.0,
                })

        # Compute recovery improvement
        pre_shortage = pre_failure.get("shortage", float("inf"))
        post_shortage = post_failure.get("shortage", float("inf")) if post_failure else float("inf")

        if pre_shortage > 0 and post_shortage < pre_shortage:
            recovery_improvement = (pre_shortage - post_shortage) / pre_shortage * 100.0
        elif post_shortage >= pre_shortage:
            recovery_improvement = 0.0
        else:
            recovery_improvement = None  # Could not improve

        # Compute recommendations: which communities lost the most allocation
        recommendations = []
        if pre_failure and post_failure:
            pre_alloc = pre_failure.get("allocations", {})
            post_alloc = post_failure.get("allocations", {})

            for c in set(list(pre_alloc.keys()) + list(post_alloc.keys())):
                pre_val = pre_alloc.get(c, 0.0)
                post_val = post_alloc.get(c, 0.0)
                if pre_val > post_val:
                    loss_pct = (pre_val - post_val) / pre_val * 100.0 if pre_val > 0 else 0.0
                    recommendations.append({
                        "community": c,
                        "pre_failure_allocation": pre_val,
                        "post_failure_allocation": post_val,
                        "loss_percentage": loss_pct,
                        "priority": net_copy.G.nodes[c].get("priority", 0)
                    })

        # Sort recommendations by loss percentage (most affected first)
        recommendations.sort(key=lambda x: x["loss_percentage"], reverse=True)

        return {
            "pre_failure_allocations": pre_failure.get("allocations", {}),
            "post_failure_allocations": post_failure.get("allocations", {}),
            "recovery_improvement_percentage": recovery_improvement,
            "affected_communities": affected_communities,
            "recovery_scenarios": recovery_scenarios,
            "recommendations": recommendations,
            "failed_edges": failed_edges,
            "n_recovery_scenarios": n_recovery_scenarios,
        }

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
