"""Generate publication-quality comparison figures from AquaRoute results.

Reads the most recent ``results_<network>_*.json`` files (which now preserve
per-repeat observations) and produces:

* ``optimization_comparison`` -- 3 networks x 3 metrics headline comparison.
* ``scenario_breakdown``      -- per-scenario comparison for the largest network.
* ``fairness_cost_tradeoff``  -- Pareto view of service deficit vs cost.

All figures use a colour-blind-safe palette, constrained layout, wrapped labels
and external legends so text does not overlap.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Okabe-Ito colour-blind-safe palette.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
           "#56B4E9", "#F0E442", "#000000"]

APPROACH_LABELS = {
    "baseline_A_shortage_min": "Shortage-min (A)",
    "baseline_B_proportional": "Proportional (B)",
    "baseline_C_maxmin": "Max-min (C)",
    "baseline_D_uniform": "Uniform (D)",
    "aquaroute_weighted_min_service": "AquaRoute",
}
APPROACH_ORDER = [
    "baseline_A_shortage_min",
    "baseline_B_proportional",
    "baseline_C_maxmin",
    "baseline_D_uniform",
    "aquaroute_weighted_min_service",
]

# Compact two-line tick labels avoid collisions in dense panels.
APPROACH_TICKS = {
    "baseline_A_shortage_min": "Shortage\n(A)",
    "baseline_B_proportional": "Proportional\n(B)",
    "baseline_C_maxmin": "Max-min\n(C)",
    "baseline_D_uniform": "Uniform\n(D)",
    "aquaroute_weighted_min_service": "AquaRoute",
}

METRICS = {
    "deficit_pct": ("Service deficit (%)", "lower"),
    "jains_fairness": ("Service fairness\n(Jain's index)", "higher"),
    "priority_service": ("Priority service\n(weighted)", "higher"),
    "delivery_cost": ("Delivery cost (relative)", "lower"),
}

NETWORK_TITLES = {"Net1": "Net1 (small)", "Net2": "Net2 (medium)", "Net3": "Net3 (large)"}


def _latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def load_network_results(results_dir: str, network: str) -> Optional[dict]:
    path = _latest(os.path.join(results_dir, f"results_{network}_*.json"))
    if not path:
        return None
    with open(path) as fh:
        rows = json.load(fh)
    return {"rows": rows, "path": path}


def _network_total_demand(network: str) -> float:
    try:
        from aquaroute.algorithms.evaluation import get_benchmark_networks
        net = get_benchmark_networks().get(network)
        if net is not None:
            return float(net.total_demand())
    except Exception:
        pass
    return 1.0


def _raw_values(rows: List[dict], approach: str, metric: str,
                scenario: Optional[str] = None) -> List[float]:
    vals: List[float] = []
    for r in rows:
        if r.get("approach") != approach:
            continue
        if scenario is not None and r.get("scenario_type") != scenario:
            continue
        reps = r.get("repeats")
        if reps:
            vals.extend(float(rr[metric]) for rr in reps
                        if isinstance(rr.get(metric), (int, float))
                        and np.isfinite(rr.get(metric)))
        elif isinstance(r.get(metric), (int, float)) and np.isfinite(r.get(metric)):
            vals.append(float(r[metric]))
    return vals


def _bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(12345)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def _deficit_values(rows, approach, total_demand, scenario=None):
    raw = _raw_values(rows, approach, "shortage", scenario)
    if total_demand <= 0:
        return raw
    return [100.0 * v / total_demand for v in raw]


def figure_comparison(results: Dict[str, dict], total_demands: Dict[str, float],
                      outdir: str, dpi: int = 300) -> str:
    networks = [n for n in ("Net1", "Net2", "Net3") if n in results]
    metrics = list(METRICS.keys())
    nrows, ncols = len(networks), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows),
                             constrained_layout=True, squeeze=False)
    # Reserve vertical space for the suptitle so it never collides with ticks.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.955), h_pad=0.04, w_pad=0.04)

    x = np.arange(len(APPROACH_ORDER))
    width = 0.72

    for i, net in enumerate(networks):
        rows = results[net]["rows"]
        td = total_demands.get(net, 1.0)
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            means, los, his = [], [], []
            for approach in APPROACH_ORDER:
                if metric == "deficit_pct":
                    vals = _deficit_values(rows, approach, td)
                else:
                    vals = _raw_values(rows, approach, metric)
                m, lo, hi = _bootstrap_ci(vals)
                means.append(m)
                los.append(lo)
                his.append(hi)
            means = np.array(means)
            err = np.array([means - np.array(los), np.array(his) - means])
            err = np.nan_to_num(err, nan=0.0)
            colors = PALETTE[:len(APPROACH_ORDER)]
            bars = ax.bar(x, means, width=width, color=colors,
                          edgecolor="black", linewidth=0.6,
                          yerr=err, capsize=3,
                          error_kw={"elinewidth": 0.8, "capthick": 0.8})
            # Emphasise AquaRoute.
            bars[-1].set_edgecolor("#000000")
            bars[-1].set_linewidth(1.6)

            label = METRICS[metric][0]
            ax.set_title(f"{NETWORK_TITLES.get(net, net)}  |  {label.replace(chr(10), ' ')}",
                         fontsize=10.5, fontweight="bold")
            ax.set_ylabel(label, fontsize=9.5)
            ax.set_xticks(x)
            ax.set_xticklabels([APPROACH_TICKS[a] for a in APPROACH_ORDER],
                               rotation=0, ha="center", fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.set_axisbelow(True)
            if metric in ("jains_fairness", "priority_service"):
                ax.set_ylim(0, 1.05)

    handles = [Patch(facecolor=PALETTE[k], edgecolor="black",
                     label=APPROACH_LABELS[a])
               for k, a in enumerate(APPROACH_ORDER)]
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False, fontsize=9.5)
    fig.suptitle("AquaRoute vs. established allocation strategies across EPANET benchmarks",
                 fontsize=13, fontweight="bold")

    _report_overlaps(fig, "optimization_comparison")
    png = os.path.join(outdir, "optimization_comparison.png")
    pdf = os.path.join(outdir, "optimization_comparison.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def figure_scenarios(results: Dict[str, dict], total_demands: Dict[str, float],
                     network: str, outdir: str, dpi: int = 300) -> Optional[str]:
    if network not in results:
        return None
    rows = results[network]["rows"]
    scenarios = ["normal", "demand_spike", "supply_reduction", "pipeline_failure"]
    scenarios = [s for s in scenarios if any(r.get("scenario_type") == s for r in rows)]
    if not scenarios:
        return None
    metrics = ["deficit_pct", "jains_fairness", "priority_service", "delivery_cost"]
    td = total_demands.get(network, 1.0)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.0),
                             constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    n_sc = len(scenarios)
    n_ap = len(APPROACH_ORDER)
    width = 0.8 / n_ap
    x = np.arange(n_sc)

    for j, metric in enumerate(metrics):
        ax = axes[j]
        for k, approach in enumerate(APPROACH_ORDER):
            means, los, his = [], [], []
            for sc in scenarios:
                if metric == "deficit_pct":
                    vals = _deficit_values(rows, approach, td, scenario=sc)
                else:
                    vals = _raw_values(rows, approach, metric, scenario=sc)
                m, lo, hi = _bootstrap_ci(vals)
                means.append(m)
                los.append(lo)
                his.append(hi)
            means = np.array(means)
            err = np.nan_to_num(np.array([means - np.array(los),
                                          np.array(his) - means]), nan=0.0)
            ax.bar(x + (k - (n_ap - 1) / 2) * width, means, width=width,
                   color=PALETTE[k], edgecolor="black", linewidth=0.5,
                   yerr=err, capsize=2,
                   error_kw={"elinewidth": 0.6, "capthick": 0.6})
        label, _ = METRICS[metric]
        ax.set_title(f"{NETWORK_TITLES.get(network, network)} | {label.replace(chr(10), ' ')}",
                     fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)
        if metric in ("jains_fairness", "priority_service"):
            ax.set_ylim(0, 1.05)
        ax.set_ylabel(label, fontsize=9.5)

    handles = [Patch(facecolor=PALETTE[k], edgecolor="black",
                     label=APPROACH_LABELS[a])
               for k, a in enumerate(APPROACH_ORDER)]
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False, fontsize=9)
    fig.suptitle(f"Scenario breakdown on {NETWORK_TITLES.get(network, network)}",
                 fontsize=12.5, fontweight="bold")
    _report_overlaps(fig, f"scenario_breakdown_{network}")
    png = os.path.join(outdir, f"scenario_breakdown_{network}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def figure_tradeoff(results: Dict[str, dict], total_demands: Dict[str, float],
                    outdir: str, dpi: int = 300) -> Optional[str]:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    markers = {"Net1": "o", "Net2": "s", "Net3": "^"}
    for k, approach in enumerate(APPROACH_ORDER):
        for net in ("Net1", "Net2", "Net3"):
            if net not in results:
                continue
            rows = results[net]["rows"]
            td = total_demands.get(net, 1.0)
            deficit = _deficit_values(rows, approach, td)
            cost = _raw_values(rows, approach, "delivery_cost")
            fair = _raw_values(rows, approach, "jains_fairness")
            if not deficit or not cost:
                continue
            # Average per network (pooled across scenarios).
            d_m, d_lo, d_hi = _bootstrap_ci(deficit)
            c_m, c_lo, c_hi = _bootstrap_ci(cost)
            f_m = float(np.mean(fair)) if fair else np.nan
            size = 40 + 160 * (f_m if np.isfinite(f_m) else 0.5)
            ax.errorbar(d_m, c_m, xerr=[[d_m - d_lo], [d_hi - d_m]],
                        yerr=[[c_m - c_lo], [c_hi - c_m]],
                        fmt=markers[net], color=PALETTE[k], markersize=9,
                        markeredgecolor="black", markeredgewidth=0.6,
                        ecolor=PALETTE[k], elinewidth=0.8, capsize=2,
                        alpha=0.9, zorder=3)
            # Larger marker = better service fairness.
            ax.scatter([d_m], [c_m], s=size, facecolors="none",
                       edgecolors=PALETTE[k], linewidths=0.8, zorder=2)

    approach_handles = [Patch(facecolor=PALETTE[k], edgecolor="black",
                              label=APPROACH_LABELS[a])
                        for k, a in enumerate(APPROACH_ORDER)]
    net_handles = [plt.Line2D([], [], marker=m, color="gray", linestyle="None",
                              markersize=8, label=NETWORK_TITLES.get(n, n))
                   for n, m in markers.items() if n in results]
    leg1 = ax.legend(handles=approach_handles, loc="upper right",
                     fontsize=8.5, frameon=True, title="Approach",
                     title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=net_handles, loc="lower left", fontsize=8.5,
              frameon=True, title="Network", title_fontsize=9)
    ax.set_xlabel("Service deficit (%)  [lower is better]", fontsize=10.5)
    ax.set_ylabel("Delivery cost (relative)  [lower is better]", fontsize=10.5)
    ax.set_title("Fairness-cost-service trade-off\n(marker size = service fairness)",
                 fontsize=12, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    _report_overlaps(fig, "fairness_cost_tradeoff")
    png = os.path.join(outdir, "fairness_cost_tradeoff.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def _report_overlaps(fig, name: str) -> int:
    """Detect overlapping text bounding boxes; print a warning and return count."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []

    def add(text_artist):
        try:
            label = text_artist.get_text()
        except Exception:
            return
        if not label or not label.strip():
            return
        try:
            bb = text_artist.get_window_extent(renderer)
        except Exception:
            return
        items.append((label, bb))

    for ax in fig.axes:
        for t in (ax.get_xticklabels() + ax.get_yticklabels()):
            add(t)
        add(ax.title)
        add(ax.xaxis.label)
        add(ax.yaxis.label)
        for t in ax.texts:
            add(t)
    for leg in fig.legends:
        for t in leg.get_texts():
            add(t)
    for t in fig.texts:
        add(t)

    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1].overlaps(items[j][1]):
                overlaps.append((items[i][0], items[j][0]))
    if overlaps:
        print(f"[{name}] WARNING: {len(overlaps)} overlapping text pairs:")
        for a, b in overlaps[:12]:
            print(f"    {a!r} <-> {b!r}")
    else:
        print(f"[{name}] no text overlaps detected")
    return len(overlaps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate AquaRoute figures.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    os.makedirs(args.figures_dir, exist_ok=True)
    results = {}
    for net in ("Net1", "Net2", "Net3"):
        loaded = load_network_results(args.results_dir, net)
        if loaded:
            results[net] = loaded
            print(f"Loaded {net}: {loaded['path']}")
    if not results:
        raise SystemExit("No results found. Run run_experiments first.")

    total_demands = {net: _network_total_demand(net) for net in results}

    out = figure_comparison(results, total_demands, args.figures_dir, args.dpi)
    print("Wrote", out)
    largest = "Net3" if "Net3" in results else max(results)
    out = figure_scenarios(results, total_demands, largest, args.figures_dir, args.dpi)
    if out:
        print("Wrote", out)
    out = figure_tradeoff(results, total_demands, args.figures_dir, args.dpi)
    if out:
        print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
