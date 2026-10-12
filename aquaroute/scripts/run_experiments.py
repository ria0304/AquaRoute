from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from aquaroute.algorithms.evaluation import EvaluationSuite
from aquaroute.models.water_network import WaterNetwork, create_sample_network


def build_network(path: str) -> WaterNetwork:
    if path:
        return WaterNetwork.from_json(path)
    return create_sample_network()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the AquaRoute comparative evaluation and export results.")
    parser.add_argument("--network", default="",
                        help="Path to a JSON network file (defaults to the sample network).")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Repeats per scenario for averaging (default 5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible scenario ordering.")
    parser.add_argument("--outdir", default="results",
                        help="Directory for exported results (created if needed).")
    parser.add_argument("--label", default="",
                        help="Optional label appended to the output filenames.")
    args = parser.parse_args(argv)

    network = build_network(args.network)
    suite = EvaluationSuite(network=network, seed=args.seed)

    print(f"Network: {len(network.get_communities())} communities, "
          f"{len(network.get_reservoirs())} reservoirs, "
          f"{len(list(network.G.edges()))} pipelines")
    print(f"Running {len(suite.approaches)} approaches x {len(suite.scenarios)} scenarios "
          f"x {args.repeats} repeats (seed={args.seed})...")

    start = time.time()
    results = suite.run_comparison(n_repeats=args.repeats, seed=args.seed)
    elapsed = time.time() - start

    summary = suite.compute_summary_metrics()
    print(f"\nCompleted in {elapsed:.2f}s.\n")
    header = f"{'approach':<34}{'shortage':>10}{'fairness':>10}{'cost':>10}{'feas%':>8}"
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        print(f"{name:<34}{m['avg_shortage']:>10.2f}{m['avg_fairness']:>10.3f}"
              f"{m['avg_cost']:>10.2f}{100 * m['feasibility_rate']:>8.1f}")

    label = f"_{args.label}" if args.label else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"results{label}_{stamp}.csv")
    json_path = os.path.join(args.outdir, f"results{label}_{stamp}.json")
    suite.export_results_csv(results, csv_path)
    suite.export_results_json(results, json_path)

    meta_path = os.path.join(args.outdir, f"meta{label}_{stamp}.json")
    with open(meta_path, "w") as fh:
        json.dump({
            "network": args.network or "sample",
            "seed": args.seed,
            "repeats": args.repeats,
            "elapsed_seconds": elapsed,
            "solver": next(iter(results.values()), [{}])[0]
                        .get("results", {}).get("solver", "unknown"),
            "summary": summary,
            "scenario_types": [r.get("scenario_type")
                               for r in next(iter(results.values()), [])],
        }, fh, indent=2)

    print(f"\nExported: {csv_path}\n          {json_path}\n          {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())