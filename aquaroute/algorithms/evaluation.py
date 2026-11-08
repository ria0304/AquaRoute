from __future__ import annotations

import copy
import json
import math
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from aquaroute.algorithms.optimizer import AllocationOptimizer, compute_fairness, compute_shortage
from aquaroute.models.water_network import WaterNetwork

from .uncertainty import ScenarioGenerator


def _priority_communities(network: WaterNetwork) -> List[int]:
    """Communities with the highest (numerically lowest) priority level."""
    communities = network.get_communities()
    if not communities:
        return []
    levels = [network.G.nodes[c].get("priority", 0) for c in communities]
    best = min(levels)
    return [c for c, lvl in zip(communities, levels) if lvl == best]


def jains_fairness_index(allocations: Dict[int, float], demands: Dict[int, float],
                         normalized: bool = True) -> float:
    """Compute Jain's fairness index for water allocation.

    Jain's index = (sum x_i)^2 / (n * sum x_i^2). Range: [1/n, 1], where 1 is
    perfect fairness.

    Parameters
    ----------
    allocations : dict
        Mapping of community ID -> allocated volume.
    demands : dict
        Mapping of community ID -> demand volume.
    normalized : bool, optional
        If True (default) the index is computed on *service ratios*
        ``x_i = alloc_i / demand_i`` (demand-normalised fairness), which makes
        communities with different demand sizes comparable.  If False the index
        is computed on raw allocated volumes (volume fairness).  Both are
        reported by the evaluation suite.

    Returns
    -------
    float
        Jain's fairness index. 1.0 = perfectly fair, 1/n = perfectly unfair.
    """
    # Only consider communities with positive demand and non-negative allocation
    valid_communities = [
        c for c in allocations
        if c in demands and demands[c] > 0 and allocations[c] >= 0
    ]

    if not valid_communities:
        return 1.0  # No communities to evaluate

    n = len(valid_communities)
    if normalized:
        values = [allocations[c] / demands[c] for c in valid_communities]
    else:
        values = [allocations[c] for c in valid_communities]

    sum_values = sum(values)
    sum_squares = sum(v ** 2 for v in values)

    if sum_squares == 0:
        # Everyone receives nothing: perfectly equal (all zero service).
        return 1.0 if sum_values == 0 else 0.0

    jai_index = (sum_values ** 2) / (n * sum_squares)
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

    Loads the real EPANET benchmark networks (Net1, Net2, Net3) through the
    :func:`aquaroute.data.loaders.load_epanet_inp` loader, which preserves the
    physical topology, pipe diameters/lengths/roughness and demands.  The loader
    metadata (including ``source_path`` and node mappings) is attached to each
    network as ``network.metadata`` so hydraulic validation can be run later.

    Returns
    -------
    dict
        Mapping of network name to WaterNetwork instance.
    """
    benchmark_networks: Dict[str, WaterNetwork] = {}

    try:
        import pathlib

        import wntr

        from aquaroute.data.loaders import load_epanet_inp

        networks_dir = pathlib.Path(wntr.__file__).parent / "library" / "networks"
        if not networks_dir.is_dir():
            networks_dir = None
            for p in pathlib.Path().rglob("*.inp"):
                if p.name in ("Net1.inp", "Net2.inp", "Net3.inp"):
                    networks_dir = p.parent
                    break

        if networks_dir is not None:
            for net_name in ["Net1", "Net2", "Net3"]:
                inp_file = networks_dir / f"{net_name}.inp"
                if inp_file.exists():
                    net, meta = load_epanet_inp(inp_file)
                    net.metadata = meta
                    benchmark_networks[net_name] = net
    except ImportError:
        pass

    # Fall back to synthetic samples only if the EPANET files are unavailable.
    if not benchmark_networks:
        for i in range(1, 4):
            net = WaterNetwork()
            net.add_reservoir(i - 1, capacity=100.0, supply=100.0)
            last = None
            for j in range(3):
                cid = i * 3 + j
                net.add_community(cid, demand=20.0 * (j + 1), priority=j + 1)
                net.add_pipeline(i - 1, cid, 50.0, 1.0)
                last = cid
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
        # Scale the minimum-service guarantee to the network's demand units so
        # it remains feasible on real EPANET networks (demands ~1e-3 m3/s) as
        # well as synthetic networks (demands ~10-100).  Guarantee 25% of the
        # smallest positive priority demand.
        prio_demands = [self.network.G.nodes[c].get("demand", 0.0)
                        for c in priority
                        if self.network.G.nodes[c].get("demand", 0.0) > 0]
        min_allocation = 0.25 * min(prio_demands) if prio_demands else 0.0
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
                "min_allocation": min_allocation,
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
        # Choose real, deterministically-selected *interior* edges for the
        # failure scenario (both endpoints are demand nodes) so the disruption
        # reduces capacity/redundancy without trivially isolating the network.
        communities = set(self.network.get_communities())
        interior = []
        for u, v in sorted(self.network.G.edges()):
            if u in communities and v in communities:
                cap = self.network.G[u][v].get("capacity", 0.0)
                interior.append((cap, u, v))
        # Fail the two highest-capacity interior links (most impactful but not
        # necessarily disconnecting).
        interior.sort(reverse=True)
        failed = [(u, v) for _, u, v in interior[:2]]
        return [
            ScenarioGenerator.normal_scenario(self.network),
            ScenarioGenerator.demand_spike_scenario(self.network, 1.5),
            ScenarioGenerator.supply_reduction_scenario(self.network, 0.5),
            ScenarioGenerator.pipeline_failure_scenario(self.network, failed),
        ]

    def _solve_with_optimizer(self, network: WaterNetwork, kwargs: Dict) -> Optional[Dict]:
        """Solve allocation problem using the optimizer with given kwargs."""
        strategy = kwargs.get("strategy", "weighted")
        try:
            opt = AllocationOptimizer(network=network, **kwargs)
            solution = opt.solve(allow_fallback=True)
            if solution is not None and solution.get("allocations"):
                # Use get_results() which computes full metrics from the solution
                results = opt.get_results()
                # Propagate real solver metrics from the optimizer results
                # into the solution dict so _compute_all_metrics can find them
                solution["solver"] = results.get("solver", opt.solution.get("solver", "unknown") if opt.solution else "unknown")
                solution["status"] = results.get("status", "optimal")
                solution["runtime"] = results.get("runtime", 0.0)
                solution["delivery_cost"] = results.get("delivery_cost", 0.0)
                solution["validation_passed"] = results.get("validation_passed", True)
                solution["validation_errors"] = results.get("validation_errors", [])
                solution["validation_warnings"] = results.get("validation_warnings", [])
                solution["shortage"] = results.get("shortage")
                solution["fairness"] = results.get("fairness")
                solution["jains_fairness"] = results.get("jains_fairness")
                solution["objective_value"] = results.get("objective_value")
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

        # ``allocations`` may be either a raw community->volume map or a full
        # optimizer solution dict (which nests the map under "allocations").
        if "allocations" in allocations and isinstance(allocations["allocations"], dict):
            solution = allocations
            alloc_map = {int(k): float(v) for k, v in solution["allocations"].items()}
        else:
            solution = {}
            alloc_map = {int(k): float(v) for k, v in allocations.items()}

        # Basic metrics from the optimizer solution (preferred) or compute them
        shortage = solution.get("shortage")
        if shortage is None:
            shortage = compute_shortage(alloc_map, demands)
        fairness = solution.get("fairness")
        if fairness is None:
            fairness = compute_fairness(alloc_map, demands)
        jains_fairness = solution.get("jains_fairness")
        if jains_fairness is None:
            jains_fairness = jains_fairness_index(alloc_map, demands, normalized=True)
        jains_fairness_volume = solution.get("jains_fairness_volume")
        if jains_fairness_volume is None:
            jains_fairness_volume = jains_fairness_index(alloc_map, demands, normalized=False)
        delivery_cost = solution.get("delivery_cost", 0.0)

        # Objective value using default weights
        alpha, beta, gamma = 1.0, 1.0, 1.0
        objective_value = alpha * shortage + beta * fairness + gamma * delivery_cost

        # Runtime and status from the optimizer solution
        runtime = solution.get("runtime", 0.0)
        status = solution.get("status", "optimal") if solution else "unsolved"
        solver = solution.get("solver", "unknown") if solution else "unknown"
        validation_passed = solution.get("validation_passed", True) if solution else True

        metrics["shortage"] = shortage
        metrics["fairness"] = fairness
        metrics["jains_fairness"] = jains_fairness
        metrics["jains_fairness_volume"] = jains_fairness_volume
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
                    # Deterministic per-approach seed.  Python's built-in
                    # ``hash`` is salted per process (PYTHONHASHSEED) and would
                    # make results irreproducible across runs; CRC32 is stable.
                    approach_offset = zlib.crc32(approach_name.encode("utf-8")) % 100000
                    repeat_seed = base_seed + repeat_idx * 1009 + approach_offset
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
                        metrics = self._compute_all_metrics(net_copy, solution, demands_c)
                        metrics["allocations"] = {
                            int(k): float(v) for k, v in solution["allocations"].items()}
                        metrics["cvar"] = solution.get("cvar", 0.0)
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
                            "allocations": {},
                            "shortage": float("nan"),
                            "fairness": 0.0,
                            "jains_fairness": 1.0,
                            "jains_fairness_volume": 1.0,
                            "delivery_cost": 0.0,
                            "cvar": 0.0,
                            "objective_value": float("nan"),
                            "runtime": 0.0,
                            "status": "failed",
                            "solver": "unknown",
                        }
                        metrics["results"] = dict(metrics)

                    repeats.append(metrics)
                approach_results.append(self._average_repeats(repeats))
            all_results[approach_name] = approach_results
            self.results[approach_name] = approach_results

        return all_results

    @staticmethod
    def _average_repeats(repeats: List[Dict]) -> Dict:
        """Average metrics across repeats for a single approach-scenario combination.

        The raw per-repeat observations are retained under ``"repeats_raw"`` so
        that uncertainty intervals and significance tests can be computed from
        the individual realisations rather than from already-averaged values.
        """
        if not repeats:
            return {}

        if len(repeats) == 1:
            merged = dict(repeats[0])
            merged["repeats_raw"] = [dict(repeats[0])]
            return merged

        # Get all unique keys across repeats
        all_keys = set()
        for r in repeats:
            all_keys.update(r.keys())

        merged = {"scenario_type": repeats[0].get("scenario_type")}
        for key in all_keys:
            if key == "repeats_raw":
                continue
            # Collect numeric values for this key across repeats
            vals = [r[key] for r in repeats if isinstance(r.get(key), (int, float))]
            if vals:
                merged[key] = float(np.mean(vals))
                if len(vals) > 1:
                    merged[f"{key}_std"] = float(np.std(vals, ddof=1))
            else:
                merged[key] = repeats[0].get(key)

        merged["n_repeats"] = len(repeats)
        merged["repeats_raw"] = [dict(r) for r in repeats]
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
            ["approach", "scenario_type", "repeat", "n_repeats",
             "n_communities", "n_reservoirs"]
            + [f"alloc_{c}" for c in sorted(all_communities)]
            + ["shortage", "shortage_std", "fairness", "fairness_std",
               "jains_fairness", "jains_fairness_std",
               "jains_fairness_volume", "jains_fairness_volume_std",
               "delivery_cost", "delivery_cost_std",
               "cvar", "cvar_std", "objective_value", "runtime", "status", "solver"]
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
                        "n_repeats": r.get("n_repeats", 1),
                        "n_communities": r.get("n_communities", 0),
                        "n_reservoirs": r.get("n_reservoirs", 0),
                    }

                    # Add allocation values
                    for c in sorted(all_communities):
                        row[f"alloc_{c}"] = r.get("allocations", {}).get(c, 0)

                    # Add all metrics
                    for key in ("shortage", "shortage_std", "fairness", "fairness_std",
                                "jains_fairness", "jains_fairness_std",
                                "jains_fairness_volume", "jains_fairness_volume_std",
                                "delivery_cost", "delivery_cost_std", "cvar", "cvar_std",
                                "objective_value", "runtime", "status", "solver"):
                        row[key] = r.get(key, "")

                    writer.writerow(row)

    def export_results_json(self, results: Dict[str, List[Dict]], filepath: str) -> None:
        """Export results to JSON with comprehensive metrics.

        Includes all enhanced metrics plus metadata about the evaluation run.
        """

        if not results:
            return

        export = []
        for approach_name, approach_result in results.items():
            for r in approach_result:
                entry = {
                    "approach": approach_name,
                    "scenario_type": r.get("scenario_type", "unknown"),
                    "n_communities": r.get("n_communities", 0),
                    "n_reservoirs": r.get("n_reservoirs", 0),
                    "n_repeats": r.get("n_repeats", 1),
                    "allocations": r.get("allocations", {}),
                }
                for k in ("shortage", "fairness", "jains_fairness",
                          "jains_fairness_volume", "delivery_cost", "cvar",
                          "objective_value", "runtime", "status", "solver"):
                    entry[k] = r.get(k)
                for k in ("shortage", "fairness", "jains_fairness",
                          "jains_fairness_volume", "delivery_cost", "cvar"):
                    std_key = f"{k}_std"
                    if std_key in r:
                        entry[std_key] = r.get(std_key)
                # Preserve per-repeat observations for uncertainty analysis.
                raw = r.get("repeats_raw")
                if raw:
                    entry["repeats"] = [
                        {
                            "repeat": rr.get("repeat", i),
                            **{k: rr.get(k) for k in
                               ("shortage", "fairness", "jains_fairness",
                                "jains_fairness_volume", "delivery_cost", "cvar",
                                "objective_value", "runtime", "status")},
                        }
                        for i, rr in enumerate(raw)
                    ]
                export.append(entry)

        with open(filepath, "w") as fh:
            json.dump(export, fh, indent=2)

    def hydraulic_validation_report(self, scenario_type: str = "supply_reduction") -> Dict[str, Dict]:
        """Validate each approach's allocation on the *original* EPANET model.

        Each approach is solved under the chosen disruption scenario (default
        ``supply_reduction``, which yields partial-service allocations) and the
        resulting allocation is fed back into the real network and simulated
        with the EPANET engine.  The report captures pressure/velocity
        feasibility and fidelity relative to the full-demand baseline.  Returns
        an empty dict when the original network file is unavailable.
        """
        metadata = getattr(self.network, "metadata", None)
        if not metadata or not metadata.get("source_path"):
            return {}
        try:
            from aquaroute.algorithms.hydraulics import HydraulicValidator
            from .uncertainty import apply_scenario
            validator = HydraulicValidator(self.network, metadata)
        except Exception:
            return {}

        scenario = next((s for s in self.scenarios
                         if s.get("scenario_type") == scenario_type), None)

        report: Dict[str, Dict] = {}
        for approach_name, approach_kwargs in self.approaches:
            net_copy = self._copy_network()
            if scenario is not None:
                apply_scenario(net_copy, scenario)
            try:
                solution = self._solve_with_optimizer(net_copy, dict(approach_kwargs))
            except Exception as exc:  # pragma: no cover - defensive
                report[approach_name] = {"error": str(exc)}
                continue
            if not solution or not solution.get("allocations"):
                report[approach_name] = {"error": "no allocation"}
                continue
            allocs = {int(k): float(v) for k, v in solution["allocations"].items()}
            try:
                rep = validator.validate(allocs)
                rep["scenario_type"] = scenario_type
                report[approach_name] = rep
            except Exception as exc:  # pragma: no cover - defensive
                report[approach_name] = {"error": str(exc)}
        return report

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

            # Prefer raw per-repeat observations for statistics when present.
            def raw_col(key):
                vals = []
                for row in rows:
                    raw = row.get("repeats_raw")
                    if raw:
                        vals.extend(rr[key] for rr in raw
                                    if isinstance(rr.get(key), (int, float)))
                return vals

            # Basic metrics
            shortages = raw_col("shortage") or col("shortage")
            fairness_vals = raw_col("fairness") or col("fairness")
            jains_vals = raw_col("jains_fairness") or col("jains_fairness")
            jains_vol_vals = (raw_col("jains_fairness_volume")
                              or col("jains_fairness_volume"))
            costs = raw_col("delivery_cost") or col("delivery_cost")
            cvar_vals = raw_col("cvar") or col("cvar")
            runtimes = raw_col("runtime") or col("runtime")

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

                # Jain's fairness index (service-normalised)
                "avg_jains_fairness": float(np.mean(jains_vals)) if jains_vals else 1.0,
                "worst_jains_fairness": float(max(jains_vals)) if jains_vals else 1.0,
                "std_jains_fairness": (float(np.std(jains_vals, ddof=1))
                                       if len(jains_vals) > 1 else 0.0),

                # Jain's fairness index (raw volume)
                "avg_jains_fairness_volume": (float(np.mean(jains_vol_vals))
                                              if jains_vol_vals else 1.0),
                "std_jains_fairness_volume": (float(np.std(jains_vol_vals, ddof=1))
                                              if len(jains_vol_vals) > 1 else 0.0),

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
                "feasibility_rate": (sum(1 for s in shortages if s == 0)
                                     / len(shortages)) if shortages else 0.0,

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

        def _samples(rows):
            """Flatten per-repeat observations when available, else scenario means."""
            vals = []
            for r in rows:
                raw = r.get("repeats_raw")
                if raw:
                    vals.extend(rr[metric] for rr in raw
                                if isinstance(rr.get(metric), (int, float)))
                elif isinstance(r.get(metric), (int, float)):
                    vals.append(r[metric])
            return [v for v in vals if not np.isnan(v)]

        # Extract the metric values for each repeat-scenario combination
        sample_a = _samples(a)
        sample_b = _samples(b)

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

    def bootstrap_ci(self, data: List[float], alpha: float = 0.05, n_bootstraps: int = 1000) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval for the mean.

        Parameters
        ----------
        data : list of float
            Sample data.
        alpha : float, optional
            Significance level. Default 0.05.
        n_bootstraps : int, optional
            Number of bootstrap samples. Default 1000.

        Returns
        -------
        tuple of (mean, lower_ci, upper_ci)
        """
        import numpy as np

        data = np.array(data, dtype=float)
        n = len(data)
        boot_means = np.empty(n_bootstraps)

        rng = np.random.default_rng(42)
        for i in range(n_bootstraps):
            sample = rng.choice(data, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        mean = np.mean(data)
        ci_lower = np.percentile(boot_means, lower_percentile)
        ci_upper = np.percentile(boot_means, upper_percentile)

        return float(mean), float(ci_lower), float(ci_upper)

    def cliffs_delta(self, data_a: List[float], data_b: List[float]) -> Tuple[float, str]:
        """Compute Cliff's delta effect size between two independent samples.

        Parameters
        ----------
        data_a : list of float
            First sample.
        data_b : list of float
            Second sample.

        Returns
        -------
        tuple of (delta, interpretation)
            Cliff's delta ranges: [-1, 1], where:
            - abs(delta) < 0.147: negligible
            - 0.147 <= abs(delta) < 0.33: small
            - 0.33 <= abs(delta) < 0.474: medium
            - abs(delta) >= 0.474: large
            sign(delta) indicates which sample tends to have larger values.
        """
        import numpy as np

        data_a = np.array(data_a, dtype=float)
        data_b = np.array(data_b, dtype=float)

        # Count how many times a > b and a < b
        n_a, n_b = len(data_a), len(data_b)
        greater = 0
        less = 0

        for val_a in data_a:
            for val_b in data_b:
                if val_a > val_b:
                    greater += 1
                elif val_a < val_b:
                    less += 1

        delta = (greater - less) / (n_a * n_b) if n_a * n_b > 0 else 0.0

        # Interpretation
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            interpretation = "negligible"
        elif abs_delta < 0.33:
            interpretation = "small"
        elif abs_delta < 0.474:
            interpretation = "medium"
        else:
            interpretation = "large"

        return float(delta), interpretation

    def wilcoxon_paired_test(self, data_a: List[float], data_b: List[float]) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test for paired samples.

        Parameters
        ----------
        data_a : list of float
            First sample (paired with data_b).
        data_b : list of float
            Second sample (paired with data_a).

        Returns
        -------
        dict with test statistics:
            - statistic: Wilcoxon test statistic (T)
            - p_value: two-sided p-value
            - mean_diff: mean of differences (data_a - data_b)
        """
        import numpy as np
        from scipy.stats import wilcoxon

        data_a = np.array(data_a, dtype=float)
        data_b = np.array(data_b, dtype=float)

        # Ensure equal length by trimming to minimum
        n = min(len(data_a), len(data_b))
        data_a = data_a[:n]
        data_b = data_b[:n]

        # Compute differences
        diffs = data_a - data_b
        mean_diff = float(np.mean(diffs))

        # Wilcoxon test (only non-zero differences)
        non_zero_diffs = diffs[np.abs(diffs) > 1e-10]

        if len(non_zero_diffs) == 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "mean_diff": mean_diff,
            }

        try:
            stat, p_value = wilcoxon(non_zero_diffs)
            return {
                "statistic": float(stat),
                "p_value": float(p_value),
                "mean_diff": mean_diff,
            }
        except Exception:
            return {
                "statistic": float(np.sum(np.sign(diffs))),
                "p_value": 1.0,
                "mean_diff": mean_diff,
            }

    def plot_results(self, metric: str = "shortage",
                     approaches: Optional[List[str]] = None,
                     scenario: str = "normal",
                     xlabel: str = "Approach",
                     ylabel: str = "Shortage (lps)",
                     title: str = "Comparison of Optimization Approaches",
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 6),
                     dpi: int = 300) -> None:
        """Plot comparison results with colorblind-safe palette.

        Creates a publication-ready bar plot with mean +/- bootstrap CI
        for the specified metric across approaches.

        Parameters
        ----------
        metric : str, optional
            Metric to plot. Default "shortage".
        approaches : list of str, optional
            List of approach names. If None, uses all approaches.
        scenario : str, optional
            Scenario type for labeling. Default "normal".
        xlabel : str, optional
            X-axis label. Default "Approach".
        ylabel : str, optional
            Y-axis label. Default "Shortage (lps)".
        title : str, optional
            Plot title. Default "Comparison of Optimization Approaches".
        save_path : str, optional
            Path to save figure. If None, shows the plot.
        figsize : tuple of int, optional
            Figure size. Default (10, 6).
        dpi : int, optional
            Dots per inch. Default 300.
        """
        import matplotlib
        import numpy as np
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Colorblind-safe palette
        cb_palette = [
            '#0173B2', '#DE8F05', '#029E73', '#56B4E9', '#E07A5F',
            '#EDC949', '#7876B1', '#43AA8B', '#DDAA9H', '#7B4173'
        ]


        if approaches is None:
            approaches = list(self.results.keys())

# Collect data
        means = []
        cis = []
        stds = []

        for approach in approaches:
            if approach not in self.results:
                means.append(np.nan)
                cis.append((np.nan, np.nan))
                stds.append(np.nan)
                continue

            # Extract metric values across repeats
            # results[approach] is a list of dicts (one per repeat, from _average_repeats)
            values = []
            for r in self.results[approach]:
                if isinstance(r, dict) and 'shortage' in r:
                    values.append(float(r['shortage']))

            if values:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                # Bootstrap CI
                mean_boot, ci_low, ci_up = self.bootstrap_ci(values, alpha=0.05)
                means.append(mean_val)
                cis.append((ci_low, ci_up))
                stds.append(std_val)
            else:
                means.append(np.nan)
                cis.append((np.nan, np.nan))
                stds.append(np.nan)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        x_pos = np.arange(len(approaches))
        colors = cb_palette[:len(approaches)]

        # Plot bars
        bars = ax.bar(x_pos, means, yerr=[[m - ci[0] for ci, m in zip(cis, means)],
                                              [ci[1] - m for ci, m in zip(cis, means)]],
                      capsize=5, color=colors, error_kw={'linewidth': 1.5, 'capthick': 1.5},
                      alpha=0.8)

        # Add value labels
        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)

        # Formatting
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(approaches, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add sample size annotations
        for i, (mean_val, std_val) in enumerate(zip(means, stds)):
            if not np.isnan(mean_val) and not np.isnan(std_val):
                ax.text(i, mean_val + 0.05, f'n={len(self.results[approaches[i]]) if approaches[i] in self.results else 0}',
                       ha='center', va='bottom', fontsize=7, style='italic')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

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