from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from aquaroute.models.water_network import WaterNetwork

from .optimizer import AllocationOptimizer
from .uncertainty import ScenarioGenerator


def _priority_communities(network: WaterNetwork) -> List[int]:
    """Communities with the highest (numerically lowest) priority level."""
    communities = network.get_communities()
    if not communities:
        return []
    levels = [network.G.nodes[c].get("priority", 0) for c in communities]
    best = min(levels)
    return [c for c, lvl in zip(communities, levels) if lvl == best]


class EvaluationSuite:
    """Compares allocation strategies across standardized scenarios.

    Strategies are genuinely distinct algorithms (see
    :class:`~aquaroute.algorithms.optimizer.AllocationOptimizer`), so the
    comparison is meaningful rather than a relabelling of one LP.
    """

    def __init__(self, network: WaterNetwork,
                 approaches: Optional[List[Tuple[str, Dict]]] = None,
                 scenarios: Optional[List[Dict]] = None,
                 seed: int = 42):
        self.network = network
        self.approaches = approaches or self._default_approaches()
        self.scenarios = scenarios or self._default_scenarios()
        self.seed = seed
        self.results: Dict[str, List[Dict]] = {}

    def _default_approaches(self) -> List[Tuple[str, Dict]]:
        priority = _priority_communities(self.network)
        return [
            # Baseline A: classic shortage minimisation (maximise delivered water).
            ("baseline_A_shortage_min", {"strategy": "shortage_min"}),
            # Baseline B: proportional allocation (share in proportion to demand).
            ("baseline_B_proportional", {"strategy": "proportional"}),
            # Baseline C: egalitarian / max-min fairness.
            ("baseline_C_maxmin", {"strategy": "maxmin"}),
            # Baseline D: uniform volumetric allocation.
            ("baseline_D_uniform", {"strategy": "uniform"}),
            # AquaRoute: weighted fairness/cost objective with minimum-service
            # guarantees for the highest-priority communities.
            ("aquaroute_weighted_min_service", {
                "strategy": "weighted",
                "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
                "min_service_priority": priority,
                "min_allocation": 10.0,
            }),
        ]

    def _default_scenarios(self) -> List[Dict]:
        return [
            ScenarioGenerator.normal_scenario(self.network),
            ScenarioGenerator.demand_spike_scenario(self.network, 1.5),
            ScenarioGenerator.supply_reduction_scenario(self.network, 0.5),
            ScenarioGenerator.pipeline_failure_scenario(self.network, [(0, 2), (1, 5)]),
        ]

    def _copy_network(self) -> WaterNetwork:
        net = WaterNetwork()
        net.G = copy.deepcopy(self.network.G)
        net.n_reservoirs = self.network.n_reservoirs
        net.n_communities = self.network.n_communities
        return net

    def run_comparison(self, n_repeats: int = 1, seed: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Run every approach on every scenario, averaging over ``n_repeats``."""
        all_results: Dict[str, List[Dict]] = {}

        for approach_name, approach_kwargs in self.approaches:
            approach_results = []
            for scenario in self.scenarios:
                repeats = []
                for _ in range(max(1, n_repeats)):
                    net_copy = self._copy_network()
                    from .uncertainty import apply_scenario
                    apply_scenario(net_copy, scenario)

                    kwargs = dict(approach_kwargs)
                    kwargs.setdefault("min_service_priority", [])
                    optimizer = AllocationOptimizer(network=net_copy, **kwargs)
                    solution = optimizer.solve()
                    repeats.append({
                        "scenario_type": scenario.get("scenario_type", "unknown"),
                        "allocations": (solution.get("allocations", {}) if solution else {}),
                        "results": optimizer.get_results() if solution else {},
                    })
                approach_results.append(self._average_repeats(repeats))
            all_results[approach_name] = approach_results
            self.results[approach_name] = approach_results
        return all_results

    @staticmethod
    def _average_repeats(repeats: List[Dict]) -> Dict:
        if not repeats:
            return {}
        if len(repeats) == 1:
            return dict(repeats[0])

        merged: Dict = {"scenario_type": repeats[0].get("scenario_type"),
                        "allocations": repeats[0].get("allocations", {})}
        metric_dicts = [r.get("results", {}) for r in repeats]
        keys = set().union(*[set(d) for d in metric_dicts]) if metric_dicts else set()
        averaged = {}
        for key in keys:
            vals = [d[key] for d in metric_dicts if isinstance(d.get(key), (int, float))]
            averaged[key] = float(np.mean(vals)) if vals else metric_dicts[0].get(key)
        merged["results"] = averaged
        return merged

    def export_results_csv(self, results: Dict[str, List[Dict]], filepath: str) -> None:
        import csv
        if not results:
            return
        all_communities = set()
        for approach_result in results.values():
            for r in approach_result:
                all_communities.update(r.get("allocations", {}).keys())
        fields = (["approach", "scenario_type"]
                  + [f"alloc_{c}" for c in sorted(all_communities)]
                  + ["shortage", "fairness", "delivery_cost", "objective_value",
                     "runtime", "status", "solver"])
        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for approach_name, approach_result in results.items():
                for r in approach_result:
                    metrics = r.get("results", {})
                    row = {"approach": approach_name,
                           "scenario_type": r.get("scenario_type", "unknown")}
                    for c in all_communities:
                        row[f"alloc_{c}"] = r.get("allocations", {}).get(c, 0)
                    for key in ("shortage", "fairness", "delivery_cost",
                                "objective_value", "runtime", "status", "solver"):
                        row[key] = metrics.get(key, "")
                    writer.writerow(row)

    def export_results_json(self, results: Dict[str, List[Dict]], filepath: str) -> None:
        import json
        export = []
        for approach_name, approach_result in results.items():
            for r in approach_result:
                metrics = r.get("results", {})
                export.append({
                    "approach": approach_name,
                    "scenario_type": r.get("scenario_type", "unknown"),
                    "allocations": r.get("allocations", {}),
                    **{k: metrics.get(k) for k in
                       ("shortage", "fairness", "delivery_cost", "objective_value",
                        "total_demand", "total_allocated", "runtime", "status", "solver")},
                })
        with open(filepath, "w") as fh:
            json.dump(export, fh, indent=2)

    def compute_summary_metrics(self) -> Dict:
        summary = {}
        for approach_name, approach_results in self.results.items():
            rows = [r["results"] for r in approach_results
                    if isinstance(r.get("results"), dict) and r["results"]]
            if not rows:
                continue

            def col(key):
                return [row[key] for row in rows if isinstance(row.get(key), (int, float))]

            shortages, fairness_vals, costs, runtimes = (
                col("shortage"), col("fairness"), col("delivery_cost"), col("runtime"))
            summary[approach_name] = {
                "avg_shortage": float(np.mean(shortages)) if shortages else 0.0,
                "worst_shortage": max(shortages) if shortages else 0.0,
                "avg_fairness": float(np.mean(fairness_vals)) if fairness_vals else 0.0,
                "worst_fairness": max(fairness_vals) if fairness_vals else 0.0,
                "avg_cost": float(np.mean(costs)) if costs else 0.0,
                "worst_cost": max(costs) if costs else 0.0,
                "avg_runtime": float(np.mean(runtimes)) if runtimes else 0.0,
                "std_shortage": float(np.std(shortages)) if len(shortages) > 1 else 0.0,
                "std_fairness": float(np.std(fairness_vals)) if len(fairness_vals) > 1 else 0.0,
                "std_cost": float(np.std(costs)) if len(costs) > 1 else 0.0,
                "feasibility_rate": (sum(1 for s in shortages if s == 0) / len(rows)) if rows else 0.0,
                "scenario_count": len(rows),
            }
        return summary

    def statistical_significance(self, metric: str = "shortage",
                                 approach_a: Optional[str] = None,
                                 approach_b: Optional[str] = None) -> Optional[Dict]:
        if approach_a not in self.results or approach_b not in self.results:
            return None
        a, b = self.results[approach_a], self.results[approach_b]
        n = min(len(a), len(b))
        if n < 2:
            return None
        ma = [a[i]["results"].get(metric, 0.0) for i in range(n)]
        mb = [b[i]["results"].get(metric, 0.0) for i in range(n)]
        diffs = [x - y for x, y in zip(ma, mb)]
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        if std_diff == 0:
            t_stat = 0.0 if mean_diff == 0 else float("inf")
            p_value = 1.0
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            try:
                from scipy import stats
                p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
            except Exception:
                p_value = float("nan")
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"{approach_a} {'better' if mean_diff < 0 else 'worse'} than {approach_b}"
                if p_value < 0.05 else "No significant difference"),
        }

    def sensitivity_analysis(self, metric: str = "shortage", approach: Optional[str] = None,
                             param: str = "alpha", values: Optional[List[float]] = None) -> Dict:
        matching = [kw for name, kw in self.approaches if name == approach]
        if not matching:
            return {}
        base = matching[0].get(param, 1.0)
        if values is None:
            values = [base * v for v in (0.5, 1.0, 1.5, 2.0)]

        out = {}
        for val in values:
            kwargs = dict(matching[0])
            kwargs[param] = float(val)
            net_copy = self._copy_network()
            from .uncertainty import apply_scenario
            apply_scenario(net_copy, self.scenarios[0] if self.scenarios else {})
            optimizer = AllocationOptimizer(network=net_copy, **kwargs)
            solution = optimizer.solve()
            out[float(val)] = (optimizer.get_results().get(metric)
                               if solution else float("nan"))
        return out


def compare_approaches(network: WaterNetwork,
                       approaches: Optional[List[Tuple[str, Dict]]] = None,
                       scenarios: Optional[List[Dict]] = None,
                       seed: int = 42) -> EvaluationSuite:
    suite = EvaluationSuite(network, approaches=approaches, scenarios=scenarios, seed=seed)
    suite.run_comparison()
    return suite
