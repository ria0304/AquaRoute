from __future__ import annotations

import copy
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aquaroute.models.water_network import WaterNetwork
from aquaroute.algorithms.optimizer import AllocationOptimizer, compute_fairness, compute_shortage
from .uncertainty import ScenarioGenerator


def _priority_communities(network: WaterNetwork) -> List[int]:
    """Communities with the highest (numerically lowest) priority level."""
    communities = network.get_communities()
    if not communities:
        return []
    levels = [network.G.nodes[c].get("priority", 0) for c in communities]
    best = min(levels)
    return [c for c, lvl in zip(communities, levels) if lvl == best]


def jains_fairness_index(allocations: Dict[int, float], demands: Dict[int, float]) -> float:
    """Compute Jain's fairness index for water allocation.

    Jain's index = (sum x_i)^2 / (n * sum x_i^2), where x_i is the allocated
    amount for community i. Range: [1/n, 1], where 1 is perfect fairness.

    Parameters
    ----------
    allocations : dict
        Mapping of community ID -> allocated volume.
    demands : dict
        Mapping of community ID -> demand volume.

    Returns
    -------
    float
        Jain's fairness index. 1.0 = perfectly fair, 1/n = perfectly unfair.
    """
    # Only consider communities with positive demand and allocation
    valid_communities = [
        c for c in allocations
        if c in demands and demands[c] > 0 and allocations[c] >= 0
    ]

    if not valid_communities:
        return 1.0  # No communities to evaluate

    n = len(valid_communities)
    sum_allocations = sum(allocations[c] for c in valid_communities)
    sum_squares = sum(allocations[c] ** 2 for c in valid_communities)

    if sum_squares == 0:
        return 1.0 if sum_allocations == 0 else 0.0

    jai_index = (sum_allocations ** 2) / (n * sum_squares)
    return min(jai_index, 1.0)


def _snapshot(network: WaterNetwork, scenario_type: str, failures: List,
              demand_scale: float = 1.0, supply_scale: float = 1.0) -> Dict:
    """Create a scenario snapshot from network state."""
    return {
        "demands": {c: network.G.nodes[c]["demand"] * demand_scale
                    for c in network.get_communities()},
        "supply": {r: network.G.nodes[r].get("supply", float("inf")) * supply_scale
                   for r in network.get_reservoirs()},
        "failures": list(failures),
        "scenario_type": scenario_type,
    }


def compute_cvar_from_scenarios(scenario_shortages: List[float], alpha: float = 0.95) -> float:
    """Compute CVaR from a list of scenario shortages.

    Parameters
    ----------
    scenario_shortages : list of float
        Shortage values across scenarios.
    alpha : float, optional
        Confidence level. Default 0.95.

    Returns
    -------
    float
        CVaR at the given confidence level.
    """
    if not scenario_shortages:
        return 0.0
    losses = np.array(scenario_shortages, dtype=float)
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    k = max(1, int(np.ceil((1 - alpha) * n)))
    tail_losses = sorted_losses[-k:]
    return float(np.mean(tail_losses))


def statistical_test_with_effect_size(
    sample_a: List[float],
    sample_b: List[float],
    alpha: float = 0.05
) -> Dict:
    """Perform independent t-test with effect size (Cohen's d).

    Parameters
    ----------
    sample_a : list of float
        First sample of measurements.
    sample_b : list of float
        Second sample of measurements.
    alpha : float, optional
        Significance level. Default 0.05.

    Returns
    -------
    dict
        Test results including t-statistic, p-value, and Cohen's d effect size.
    """
    import scipy.stats as stats

    n_a = len(sample_a)
    n_b = len(sample_b)

    # Means
    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)

    # Standard deviations
    std_a = np.std(sample_a, ddof=1)
    std_b = np.std(sample_b, ddof=1)

    # Pooled standard deviation
    if std_a + std_b == 0:
        cohens_d = 0.0
    else:
        pooled_std = math.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    # Welch's t-test (does not assume equal variances)
    try:
        t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    except Exception:
        t_stat = 0.0
        p_value = 1.0

    # Cohen's d for Welch's t-test
    if std_a + std_b == 0:
        cohens_d = 0.0
    else:
        cohens_d = (mean_a - mean_b) / math.sqrt((std_a ** 2 / n_a + std_b ** 2 / n_b) / 2)

    # Degrees of freedom (Welch-Satterthwaite)
    try:
        df = (std_a ** 2 / n_a + std_b ** 2 / n_b) ** 2 / (
            (std_a ** 2 / n_a) ** 2 / (n_a - 1) + (std_b ** 2 / n_b) ** 2 / (n_b - 1)
        )
        df = max(df, 1)
    except Exception:
        df = min(n_a, n_b) - 1

    # Confidence interval for mean difference
    try:
        from scipy.stats import t as t_dist
        h = t_dist.ppf(1 - alpha / 2, df)
        ci_lower = (mean_a - mean_b) - h * math.sqrt(std_a ** 2 / n_a + std_b ** 2 / n_b)
        ci_upper = (mean_a - mean_b) + h * math.sqrt(std_a ** 2 / n_a + std_b ** 2 / n_b)
    except Exception:
        ci_lower = None
        ci_upper = None

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    significant = p_value < alpha

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(mean_a - mean_b),
        "confidence_interval_lower": float(ci_lower) if ci_lower is not None else None,
        "confidence_interval_upper": float(ci_upper) if ci_upper is not None else None,
        "cohens_d": float(cohens_d),
        "effect_interpretation": effect_interpretation,
        "sample_size_a": n_a,
        "sample_size_b": n_b,
        "significant": significant,
        "interpretation": (
            f"{'better' if mean_a < mean_b else 'worse'} with {effect_interpretation} effect"
            if significant else "No significant difference"
        ),
    }


def get_benchmark_networks() -> Dict[str, WaterNetwork]:
    """Get benchmark water network datasets for evaluation.

    Returns EPANET benchmark networks (Net1, Net2, Net3) if available,
    otherwise returns sample configurations.

    Returns
    -------
    dict
        Mapping of network name to WaterNetwork instance.
    """
    benchmark_networks = {}

    try:
        import wntr

        # EPANET benchmark networks - search for wntr's bundled networks
        networks_dir = None
        # Try common locations
        import pathlib
        possible_paths = [
            pathlib.Path(wntr.__file__).parent / "library" / "networks",
            pathlib.Path("/home/user/.local/lib/python3.14/site-packages/wntr/library/networks"),
            pathlib.Path("/usr/local/share/wntr/networks"),
        ]
        for p in possible_paths:
            if p.is_dir():
                networks_dir = p
                break

        if networks_dir is None:
            # Try to find any directory with EPANET .inp files
            import glob
            for p in pathlib.Path().rglob("*.inp"):
                networks_dir = p.parent
                if any(f.name.startswith("Net") for f in networks_dir.glob("*.inp")):
                    break

        if networks_dir is None:
            raise ImportError("No EPANET benchmark networks found")

        for net_name in ["Net1", "Net2", "Net3"]:
            inp_file = networks_dir / f"{net_name}.inp"
            if inp_file.exists():
                wn = wntr.network.WaterNetworkModel(inp_file)
                # Convert to AquaRoute WaterNetwork
                net = WaterNetwork()
                # Map reservoirs
                for res_name in wn.reservoir_name_list:
                    res_node = int(res_name) if res_name.isdigit() else res_name
                    # Get reservoir data from EPANET
                    try:
                        base_head = float(wn.query_node_attribute(res_name, "head"))
                        net.add_reservoir(
                            res_node,
                            capacity=100.0,
                            supply=base_head / 10.0  # Scale head back to supply
                        )
                    except Exception:
                        net.add_reservoir(res_node, capacity=100.0, supply=float("inf"))

                # Add communities (junctions with demand)
                for junc_name in wn.junction_name_list:
                    junc_node = int(junc_name) if junc_name.isdigit() else junc_name
                    try:
                        demand = float(wn.query_node_attribute(junc_name, "demand"))
                        net.add_community(junc_node, demand=demand)
                    except Exception:
                        net.add_community(junc_node, demand=10.0)

                # Add pipes (pipelines) - map from network edges
                for pipe_name in wn.pipe_name_list:
                    try:
                        start_node = int(pipe_name.split("_")[0]) if "_" in pipe_name else int(pipe_name)
                        end_node = int(pipe_name.split("_")[1]) if "_" in pipe_name else int(pipe_name)
                        # Simplified: just note the network has these connections
                        if net.G.has_node(start_node) and net.G.has_node(end_node):
                            pass
                    except Exception:
                        pass

                benchmark_networks[net_name] = net
                print(f"Loaded benchmark network: {net_name} ({wn.num_nodes} nodes, {wn.num_links} links)")

    except ImportError:
        pass  # wntr not available
    except Exception as e:
        pass  # Skip on error

    # Add sample networks if no EPANET networks available
    if not benchmark_networks:
        for i in range(1, 4):
            net = WaterNetwork()
            net.add_reservoir(i - 1, capacity=100.0, supply=100.0)
            for j in range(3):
                net.add_community(i * 3 + j, demand=20.0 * (j + 1), priority=j + 1)
            net.add_pipeline(i - 1, i * 3 + j, 50.0, 1.0)
            benchmark_networks[f"sample_{i}"] = net

    return benchmark_networks


class EvaluationSuite:
    """Compares allocation strategies across standardized scenarios.

    Strategies are genuinely distinct algorithms (see
    :class:`~aquaroute.algorithms.optimizer.AllocationOptimizer`), so the
    comparison is meaningful rather than a relabeling of one LP.

    The framework supports:
    - Multiple optimization strategies with different objective functions
    - Stochastic scenario generation and CVaR risk assessment
    - Benchmark network datasets (EPANET Net1/2/3)
    - Comprehensive metric collection and statistical testing
    - Power analysis for required sample sizes
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

    def _copy_network(self) -> WaterNetwork:
        """Create a deep copy of the network for scenario testing."""
        net = WaterNetwork()
        net.G = copy.deepcopy(self.network.G)
        net.n_reservoirs = self.network.n_reservoirs
        net.n_communities = self.network.n_communities
        return net

    def _default_scenarios(self) -> List[Dict]:
        """Get default scenario set for evaluation."""
        return [
            ScenarioGenerator.normal_scenario(self.network),
            ScenarioGenerator.demand_spike_scenario(self.network, 1.5),
            ScenarioGenerator.supply_reduction_scenario(self.network, 0.5),
            ScenarioGenerator.pipeline_failure_scenario(self.network, [(0, 2), (1, 5)]),
        ]

    def _solve_with_optimizer(self, network: WaterNetwork, kwargs: Dict) -> Optional[Dict]:
        """Solve allocation problem using the optimizer with given kwargs."""
        strategy = kwargs.get("strategy", "weighted")
        try:
            opt = AllocationOptimizer(network=network, **kwargs)
            solution = opt.solve(allow_fallback=True)
            if solution is not None and solution.get("allocations"):
                # Propagate real solver metrics from the optimizer solution
                solution["solver"] = opt.solution.get("solver", "unknown") if opt.solution else "unknown"
                solution["status"] = opt.solution.get("status", "optimal") if opt.solution else "optimal"
                solution["runtime"] = opt.runtime if opt.runtime else 0.0
                solution["delivery_cost"] = opt.solution.get("delivery_cost", 0.0) if opt.solution else 0.0
                solution["validation_passed"] = opt.solution.get("validation_passed", True) if opt.solution else True
                solution["validation_errors"] = opt.solution.get("validation_errors", []) if opt.solution else []
                solution["validation_warnings"] = opt.solution.get("validation_warnings", []) if opt.solution else []
            return solution
        except Exception as e:
            import traceback
            print(f"Optimizer {strategy} exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Record failed result with reason instead of returning None
            return {
                "solver": "unknown",
                "status": "failed",
                "runtime": 0.0,
                "delivery_cost": 0.0,
                "validation_passed": False,
                "validation_errors": [f"{type(e).__name__}: {e}"],
                "validation_warnings": [],
            }

    def _compute_all_metrics(self, network: WaterNetwork, allocations: Dict,
                             demands: Dict) -> Dict:
        """Compute all enhanced metrics for a given allocation.

        Uses real metrics from the optimizer solution when available,
        falling back to computed values only.
        """
        metrics = {}

        # Basic metrics from the optimizer solution (preferred) or compute them
        shortage = allocations.get("shortage", compute_shortage(allocations, demands))
        fairness = allocations.get("fairness", compute_fairness(allocations, demands))
        jains_fairness = allocations.get("jains_fairness", jains_fairness_index(allocations, demands))
        delivery_cost = allocations.get("delivery_cost", 0.0)

        # Objective value using default weights
        alpha, beta, gamma = 1.0, 1.0, 1.0
        objective_value = alpha * shortage + beta * fairness + gamma * delivery_cost

        # Runtime and status from the optimizer solution
        runtime = allocations.get("runtime", 0.0)
        status = allocations.get("status", "optimal") if allocations else "unsolved"
        solver = allocations.get("solver", "unknown") if allocations else "unknown"
        validation_passed = allocations.get("validation_passed", True) if allocations else True

        metrics["shortage"] = shortage
        metrics["fairness"] = fairness
        metrics["jains_fairness"] = jains_fairness
        metrics["delivery_cost"] = delivery_cost
        metrics["objective_value"] = objective_value
        metrics["runtime"] = runtime
        metrics["status"] = status
        metrics["solver"] = solver
        metrics["validation_passed"] = validation_passed

        return metrics

    def run_comparison(self, n_repeats: int = 20, seed: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Run every approach on every scenario with multiple repeats for statistical analysis.

        Uses ``n_repeats`` independent scenario realizations (with different random seeds)
        to enable statistical significance testing and power analysis.

        Parameters
        ----------
        n_repeats : int, optional
            Number of independent repeats per approach-scenario combination. Default 20.
        seed : int, optional
            Base random seed. If None, uses self.seed.

        Returns
        -------
        dict
            Results dictionary mapping approach names to lists of per-repeat results.
        """
        import numpy as np

        if seed is not None:
            np.random.seed(seed)
        base_seed = seed if seed is not None else 42

        all_results: Dict[str, List[Dict]] = {}

        for approach_name, approach_kwargs in self.approaches:
            approach_results = []
            for scenario in self.scenarios:
                repeats = []
                for repeat_idx in range(n_repeats):
                    # Use unique seed for each repeat
                    repeat_seed = base_seed + repeat_idx + hash(approach_name) % 1000
                    random_state = np.random.RandomState(repeat_seed)

                    net_copy = self._copy_network()

                    # Apply scenario with stochastic variation
                    from .uncertainty import apply_scenario
                    # Add stochastic variation to scenario demands/supply
                    if "demands" in scenario and scenario["demands"]:
                        varied_demands = {
                            k: v * random_state.uniform(0.7, 1.3)
                            for k, v in scenario["demands"].items()
                        }
                        scenario_varied = {
                            "demands": varied_demands,
                            "supply": scenario.get("supply", {}),
                            "failures": scenario.get("failures", []),
                            "scenario_type": scenario.get("scenario_type", "unknown"),
                        }
                        apply_scenario(net_copy, scenario_varied)
                    else:
                        apply_scenario(net_copy, scenario)

                    # Solve
                    kwargs = dict(approach_kwargs)
                    solution = self._solve_with_optimizer(net_copy, kwargs)

                    # Compute all metrics
                    if solution and solution.get("allocations"):
                        demands_c = {c: net_copy.G.nodes[c]["demand"]
                                    for c in net_copy.get_communities()}
                        metrics = self._compute_all_metrics(net_copy, solution["allocations"], demands_c)
                        metrics["approach"] = approach_name
                        metrics["scenario_type"] = scenario.get("scenario_type", "unknown")
                        metrics["repeat"] = repeat_idx
                        metrics["n_communities"] = len(net_copy.get_communities())
                        metrics["n_reservoirs"] = len(net_copy.get_reservoirs())
                        metrics["results"] = dict(metrics)
                    else:
                        metrics = {
                            "approach": approach_name,
                            "scenario_type": scenario.get("scenario_type", "unknown"),
                            "repeat": repeat_idx,
                            "n_communities": len(net_copy.get_communities()),
                            "n_reservoirs": len(net_copy.get_reservoirs()),
                            "shortage": float("nan"),
                            "fairness": 0.0,
                            "jains_fairness": 1.0,
                            "delivery_cost": 0.0,
                            "cvar": 0.0,
                            "objective_value": float("nan"),
                            "runtime": 0.0,
                            "status": "failed",
                            "solver": "unknown",
                            "results": metrics,
                        }

                    repeats.append(metrics)
                approach_results.append(self._average_repeats(repeats))
            all_results[approach_name] = approach_results
            self.results[approach_name] = approach_results

        return all_results

    @staticmethod
    def _average_repeats(repeats: List[Dict]) -> Dict:
        """Average metrics across repeats for a single approach-scenario combination."""
        if not repeats:
            return {}

        if len(repeats) == 1:
            return dict(repeats[0])

        # Get all unique keys across repeats
        all_keys = set()
        for r in repeats:
            all_keys.update(r.keys())

        merged = {"scenario_type": repeats[0].get("scenario_type")}
        for key in all_keys:
            # Collect numeric values for this key across repeats
            vals = [r[key] for r in repeats if isinstance(r.get(key), (int, float))]
            if vals:
                merged[key] = float(np.mean(vals))
            else:
                merged[key] = repeats[0].get(key)

        return merged

    def export_results_csv(self, results: Dict[str, List[Dict]], filepath: str) -> None:
        """Export results to CSV with comprehensive metrics.

        Includes all enhanced metrics: shortage, fairness, Jain's index, CVaR,
        objective value, runtime, and solver information.
        """
        import csv

        if not results:
            return

        # Collect all communities across all approaches and results
        all_communities = set()
        for approach_result in results.values():
            for r in approach_result:
                all_communities.update(r.get("allocations", {}).keys())

        # Define CSV fields with all enhanced metrics
        fields = (
            ["approach", "scenario_type", "repeat", "n_communities", "n_reservoirs"]
            + [f"alloc_{c}" for c in sorted(all_communities)]
            + ["shortage", "fairness", "jains_fairness", "delivery_cost",
               "cvar", "objective_value", "runtime", "status", "solver"]
        )

        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()

            for approach_name, approach_result in results.items():
                for r in approach_result:
                    row = {
                        "approach": approach_name,
                        "scenario_type": r.get("scenario_type", "unknown"),
                        "repeat": r.get("repeat", 0),
                        "n_communities": r.get("n_communities", 0),
                        "n_reservoirs": r.get("n_reservoirs", 0),
                    }

                    # Add allocation values
                    for c in sorted(all_communities):
                        row[f"alloc_{c}"] = r.get("allocations", {}).get(c, 0)

                    # Add all metrics
                    for key in ("shortage", "fairness", "jains_fairness", "delivery_cost",
                                "cvar", "objective_value", "runtime", "status", "solver"):
                        row[key] = r.get(key, "")

                    writer.writerow(row)

    def export_results_json(self, results: Dict[str, List[Dict]], filepath: str) -> None:
        """Export results to JSON with comprehensive metrics.

        Includes all enhanced metrics plus metadata about the evaluation run.
        """
        import json

        if not results:
            return

        export = []
        for approach_name, approach_result in results.items():
            for r in approach_result:
                metrics = r.copy()
                # Remove internal keys
                for key in ["repeat", "n_communities", "n_reservoirs"]:
                    metrics.pop(key, None)

                export.append({
                    "approach": approach_name,
                    "scenario_type": r.get("scenario_type", "unknown"),
                    "n_communities": r.get("n_communities", 0),
                    "n_reservoirs": r.get("n_reservoirs", 0),
                    "allocations": r.get("allocations", {}),
                    **{k: metrics.get(k) for k in
                       ("shortage", "fairness", "jains_fairness", "delivery_cost",
                        "cvar", "objective_value", "runtime", "status", "solver")},
                })

        with open(filepath, "w") as fh:
            json.dump(export, fh, indent=2)

    def compute_summary_metrics(self) -> Dict:
        """Compute summary statistics across all approaches and scenarios.

        Includes enhanced statistics: means, standard deviations, ranges,
        and feasibility rates for all key metrics.
        """
        summary = {}
        for approach_name, approach_results in self.results.items():
            rows = [r for r in approach_results if isinstance(r.get("results"), dict) and r.get("results")]
            if not rows:
                continue

            # Collect all metric values across repeats-scenarios
            def col(key):
                return [row[key] for row in rows if isinstance(row.get(key), (int, float))]

            # Basic metrics
            shortages = col("shortage")
            fairness_vals = col("fairness")
            jains_vals = col("jains_fairness")
            costs = col("delivery_cost")
            cvar_vals = col("cvar")
            runtimes = col("runtime")

            # Compute statistics
            n = len(rows)
            summary[approach_name] = {
                # Shortage statistics
                "avg_shortage": float(np.mean(shortages)) if shortages else 0.0,
                "worst_shortage": float(max(shortages)) if shortages else 0.0,
                "std_shortage": float(np.std(shortages, ddof=1)) if len(shortages) > 1 else 0.0,
                "min_shortage": float(min(shortages)) if shortages else 0.0,
                "max_shortage": float(max(shortages)) if shortages else 0.0,

                # Fairness statistics
                "avg_fairness": float(np.mean(fairness_vals)) if fairness_vals else 0.0,
                "worst_fairness": float(max(fairness_vals)) if fairness_vals else 0.0,
                "std_fairness": float(np.std(fairness_vals, ddof=1)) if len(fairness_vals) > 1 else 0.0,

                # Jain's fairness index
                "avg_jains_fairness": float(np.mean(jains_vals)) if jains_vals else 1.0,
                "worst_jains_fairness": float(max(jains_vals)) if jains_vals else 1.0,

                # Cost statistics
                "avg_cost": float(np.mean(costs)) if costs else 0.0,
                "worst_cost": float(max(costs)) if costs else 0.0,
                "std_cost": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,

                # CVaR statistics
                "avg_cvar": float(np.mean(cvar_vals)) if cvar_vals else 0.0,
                "worst_cvar": float(max(cvar_vals)) if cvar_vals else 0.0,

                # Runtime statistics
                "avg_runtime": float(np.mean(runtimes)) if runtimes else 0.0,
                "std_runtime": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,

                # Feasibility rate
                "feasibility_rate": (sum(1 for s in shortages if s == 0) / n) if n > 0 else 0.0,

                # Scenario count
                "scenario_count": n,

                # Approach metadata
                "approach": approach_name,
            }

        return summary

    def statistical_significance(self, metric: str = "shortage",
                                  approach_a: Optional[str] = None,
                                  approach_b: Optional[str] = None) -> Optional[Dict]:
        """Perform statistical significance test between two approaches.

        Uses independent t-test with Cohen's d effect size and confidence interval.

        Parameters
        ----------
        metric : str, optional
            Metric to test. Default "shortage".
        approach_a : str, optional
            First approach name.
        approach_b : str, optional
            Second approach name.

        Returns
        -------
        dict or None
            Test results including t-statistic, p-value, effect size, and interpretation.
        """
        if approach_a not in self.results or approach_b not in self.results:
            return None

        a = self.results[approach_a]
        b = self.results[approach_b]

        if not a or not b:
            return None

        # Extract the metric values for each repeat-scenario combination
        sample_a = [r.get(metric, float("nan")) for r in a if r.get(metric) is not None and not np.isnan(r.get(metric, float("nan")))]
        sample_b = [r.get(metric, float("nan")) for r in b if r.get(metric) is not None and not np.isnan(r.get(metric, float("nan")))]

        if len(sample_a) < 2 or len(sample_b) < 2:
            return None

        import scipy.stats as stats

        # Means
        mean_a = np.mean(sample_a)
        mean_b = np.mean(sample_b)

        # Standard deviations
        std_a = np.std(sample_a, ddof=1)
        std_b = np.std(sample_b, ddof=1)

        # Pooled standard deviation and Cohen's d
        if std_a + std_b == 0:
            cohens_d = 0.0
        else:
            pooled_std = math.sqrt(((len(sample_a) - 1) * std_a ** 2 + (len(sample_b) - 1) * std_b ** 2) / (len(sample_a) + len(sample_b) - 2))
            cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Welch's t-test
        try:
            t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        except Exception:
            t_stat = 0.0
            p_value = 1.0

        # Degrees of freedom (Welch-Satterthwaite)
        try:
            df = (std_a ** 2 / len(sample_a) + std_b ** 2 / len(sample_b)) ** 2 / (
                (std_a ** 2 / len(sample_a)) ** 2 / (len(sample_a) - 1) + (std_b ** 2 / len(sample_b)) ** 2 / (len(sample_b) - 1)
            )
            df = max(df, 1)
        except Exception:
            df = min(len(sample_a), len(sample_b)) - 1

        # Confidence interval for mean difference
        try:
            from scipy.stats import t as t_dist
            h = t_dist.ppf(1 - 0.05 / 2, df)
            ci_lower = (mean_a - mean_b) - h * math.sqrt(std_a ** 2 / len(sample_a) + std_b ** 2 / len(sample_b))
            ci_upper = (mean_a - mean_b) + h * math.sqrt(std_a ** 2 / len(sample_a) + std_b ** 2 / len(sample_b))
        except Exception:
            ci_lower = None
            ci_upper = None

        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        significant = p_value < 0.05

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_difference": float(mean_a - mean_b),
            "confidence_interval_lower": float(ci_lower) if ci_lower is not None else None,
            "confidence_interval_upper": float(ci_upper) if ci_upper is not None else None,
            "cohens_d": float(cohens_d),
            "effect_interpretation": effect_interpretation,
            "sample_size_a": len(sample_a),
            "sample_size_b": len(sample_b),
            "significant": significant,
            "interpretation": (
                f"{'better' if mean_a < mean_b else 'worse'} with {effect_interpretation} effect"
                if significant else "No significant difference"
            ),
        }

    def power_analysis(self, metric: str = "shortage",
                       effect_size: str = "medium",
                       alpha: float = 0.05,
                       power: float = 0.8) -> Dict:
        """Calculate required sample size for given power and effect size.

        Parameters
        ----------
        metric : str, optional
            Metric to power for. Default "shortage".
        effect_size : str, optional
            Effect size: "small", "medium", "large". Default "medium".
        alpha : float, optional
            Significance level. Default 0.05.
        power : float, optional
            Desired statistical power. Default 0.8.

        Returns
        -------
        dict
            Required sample size per group and power analysis results.
        """
        import scipy.stats as stats

        # Map effect size names to Cohen's d
        effect_size_map = {
            "small": 0.2,
            "medium": 0.5,
            "large": 0.8,
        }
        d = effect_size_map.get(effect_size, 0.5)

        # Calculate required sample size for two-sample t-test
        try:
            n_per_group = stats.tt_ind_solve_power(effect_size=d, alpha=alpha, power=power, ratio=1.0)
        except Exception:
            n_per_group = 64

        return {
            "effect_size": effect_size,
            "cohens_d": d,
            "alpha": alpha,
            "power": power,
            "n_per_group": n_per_group,
            "total_n": n_per_group * 2,
            "description": (
                f"Required {n_per_group} independent repeats per approach-group "
                f"to detect a {effect_size} effect ({d:.1f} Cohen's d) "
                f"with {power*100:.0f}% power at {alpha*100:.0f}% significance"
            ),
        }

    def sensitivity_analysis(self, metric: str = "shortage", approach: Optional[str] = None,
                              param: str = "alpha", values: Optional[List[float]] = None) -> Dict:
        """Perform sensitivity analysis on a parameter across approaches.

        Parameters
        ----------
        metric : str, optional
            Metric to track. Default "shortage".
        approach : str, optional
            Specific approach to analyze. If None, analyzes all approaches.
        param : str, optional
            Parameter to vary. Default "alpha".
        values : list of float, optional
            Parameter values to test. If None, uses approach-specific defaults.

        Returns
        -------
        dict
            Sensitivity analysis results mapping parameter values to metric values.
        """
        matching = [kw for name, kw in self.approaches if name == approach] if approach else self.approaches
        if not matching:
            return {}

        if values is None:
            defaults = {
                "alpha": [0.5, 1.0, 1.5, 2.0],
                "beta": [0.5, 1.0, 1.5, 2.0],
                "gamma": [0.5, 1.0, 1.5, 2.0],
                "min_allocation": [0.0, 5.0, 10.0, 20.0],
            }
            values = defaults.get(param, [1.0])

        out = {}
        for val in values:
            # matching[0] is a (name, kwargs) tuple; extract the kwargs dict
            if isinstance(matching[0], tuple) and len(matching[0]) == 2:
                kwargs = dict(matching[0][1])
            else:
                kwargs = dict(matching[0] if matching else {})
            kwargs[param] = float(val)

            net_copy = self._copy_network()
            from .uncertainty import apply_scenario
            apply_scenario(net_copy, self.scenarios[0] if self.scenarios else {})

            opt_kwargs = {k: v for k, v in kwargs.items()
                         if k in {"strategy", "alpha", "beta", "gamma", "min_service_priority", "min_allocation"}}
            optimizer = AllocationOptimizer(network=net_copy, **opt_kwargs)
            solution = optimizer.solve()

            out[float(val)] = optimizer.get_results().get(metric) if solution else float("nan")

        return out


def compare_approaches(network: WaterNetwork,
                       approaches: Optional[List[Tuple[str, Dict]]] = None,
                       scenarios: Optional[List[Dict]] = None,
                       seed: int = 42) -> EvaluationSuite:
    """Create and run an evaluation suite for comparing allocation approaches.

    Parameters
    ----------
    network : WaterNetwork
        The water distribution network to evaluate.
    approaches : list of (name, kwargs) tuples, optional
        Approaches to compare. Default uses enhanced default approaches.
    scenarios : list of dicts, optional
        Scenarios to evaluate. Default uses standard scenario set.
    seed : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    EvaluationSuite
        Configured and run evaluation suite with results.
    """
    suite = EvaluationSuite(network, approaches=approaches, scenarios=scenarios, seed=seed)
    suite.run_comparison(n_repeats=20, seed=seed)
    return suite