from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from aquaroute.algorithms.evaluation import EvaluationSuite, get_benchmark_networks
from aquaroute.models.water_network import WaterNetwork, create_sample_network


def build_network(name_or_path: str) -> tuple[WaterNetwork, str]:
    """Resolve a benchmark name, JSON path, or fall back to the sample network."""
    if not name_or_path:
        return create_sample_network(), "sample"
    if name_or_path in ("Net1", "Net2", "Net3"):
        nets = get_benchmark_networks()
        if name_or_path not in nets:
            raise SystemExit(f"Benchmark {name_or_path} could not be loaded (is WNTR installed?)")
        return nets[name_or_path], name_or_path
    if os.path.isfile(name_or_path):
        return WaterNetwork.from_json(name_or_path), os.path.basename(name_or_path)
    raise SystemExit(f"Unknown network {name_or_path!r} (use Net1/Net2/Net3 or a .json path)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the AquaRoute comparative evaluation and export results.")
    parser.add_argument("--network", default="",
                        help="Benchmark name (Net1/Net2/Net3), JSON network path, "
                             "or empty for the sample network.")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Independent repeats per scenario (default 20).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducible scenarios.")
    parser.add_argument("--outdir", default="results",
                        help="Directory for exported results (created if needed).")
    parser.add_argument("--label", default="",
                        help="Optional label appended to output filenames.")
    parser.add_argument("--hydraulic", action="store_true",
                        help="Run EPANET hydraulic validation on the original network.")
    args = parser.parse_args(argv)

    network, net_label = build_network(args.network)
    suite = EvaluationSuite(network=network, seed=args.seed)

    print(f"Network: {net_label} -- {len(network.get_communities())} communities, "
          f"{len(network.get_reservoirs())} sources, "
          f"{network.G.number_of_edges()} directed arcs")
    print(f"Running {len(suite.approaches)} approaches x {len(suite.scenarios)} scenarios "
          f"x {args.repeats} repeats (seed={args.seed})...")

    start = time.time()
    results = suite.run_comparison(n_repeats=args.repeats, seed=args.seed)
    elapsed = time.time() - start

    summary = suite.compute_summary_metrics()
    print(f"\nCompleted in {elapsed:.2f}s.\n")
    header = (f"{'approach':<34}{'shortage':>11}{'fairness':>10}"
              f"{'jain_S':>9}{'prio_svc':>10}{'cost':>10}{'feas%':>8}")
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        print(f"{name:<34}{m['avg_shortage']:>11.5f}{m['avg_fairness']:>10.4f}"
              f"{m.get('avg_jains_fairness', 0.0):>9.4f}"
              f"{m.get('avg_priority_service', 0.0):>10.4f}"
              f"{m['avg_cost']:>10.4f}{100 * m['feasibility_rate']:>8.1f}")

    # Statistical comparison of AquaRoute against each baseline.
    significance = {}
    aqua = "aquaroute_weighted_min_service"
    if aqua in self_approach_names(results):
        for metric in ("shortage", "fairness", "jains_fairness",
                       "priority_service", "delivery_cost"):
            significance[metric] = {}
            for other in results:
                if other == aqua:
                    continue
                try:
                    significance[metric][other] = suite.statistical_significance(
                        metric, aqua, other)
                except Exception as exc:  # pragma: no cover - defensive
                    significance[metric][other] = {"error": str(exc)}

    hydraulic = {}
    if args.hydraulic:
        print("\nRunning EPANET hydraulic validation...")
        hydraulic = suite.hydraulic_validation_report()
        for name, rep in hydraulic.items():
            if "error" in rep:
                print(f"  {name:<34} error: {rep['error']}")
            else:
                print(f"  {name:<34} feasible={rep['feasible']} "
                      f"abs={rep.get('absolute_feasible')} "
                      f"min_p={rep['hydraulics']['min_pressure']:.2f}m "
                      f"max_v={rep['hydraulics']['max_velocity']:.2f}m/s "
                      f"new_p_viol={len(rep.get('new_pressure_violations', []))} "
                      f"dP={rep['fidelity']['mean_abs_pressure_delta_m']:.2f}m")

    label = f"_{args.label}" if args.label else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"results_{net_label}{label}_{stamp}.csv")
    json_path = os.path.join(args.outdir, f"results_{net_label}{label}_{stamp}.json")
    suite.export_results_csv(results, csv_path)
    suite.export_results_json(results, json_path)

    meta_path = os.path.join(args.outdir, f"meta_{net_label}{label}_{stamp}.json")
    with open(meta_path, "w") as fh:
        json.dump({
            "network": net_label,
            "seed": args.seed,
            "repeats": args.repeats,
            "elapsed_seconds": elapsed,
            "n_communities": len(network.get_communities()),
            "n_sources": len(network.get_reservoirs()),
            "n_directed_arcs": network.G.number_of_edges(),
            "solver": next(iter(results.values()), [{}])[0]
                        .get("results", {}).get("solver", "unknown"),
            "summary": summary,
            "significance_vs_aquaroute": significance,
            "hydraulic_validation": hydraulic,
            "scenario_types": [r.get("scenario_type")
                               for r in next(iter(results.values()), [])],
        }, fh, indent=2, default=str)

    print(f"\nExported: {csv_path}\n          {json_path}\n          {meta_path}")
    return 0


def self_approach_names(results):
    return list(results.keys())


if __name__ == "__main__":
    raise SystemExit(main())
