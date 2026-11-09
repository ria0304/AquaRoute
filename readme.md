<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-AquaRoute-black?style=flat-square&logo=github)](https://github.com/ria0304/AquaRoute)
[![CI](https://img.shields.io/github/actions/workflow/status/ria0304/AquaRoute/ci.yml?branch=main&style=flat-square&label=CI&logo=githubactions&logoColor=white)](https://github.com/ria0304/AquaRoute/actions/workflows/ci.yml)
<img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Pyomo-Optimization-green?style=flat-square" />
<img src="https://img.shields.io/badge/Solver-HiGHS%20%2F%20CBC%20%2F%20GLPK-orange?style=flat-square" />
<img src="https://img.shields.io/badge/Graph-NetworkX-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Hydraulics-EPANET%20via%20WNTR-teal?style=flat-square" />
<img src="https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square&logo=streamlit" />
<img src="https://img.shields.io/badge/Tests-47-brightgreen?style=flat-square&logo=pytest&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />


# AquaRoute — Algorithmic Optimization for Reliable Community Water Distribution

**Build the network. Set the priorities. Solve the allocation. Compare the strategies. Check it against EPANET.**

AquaRoute turns a water distribution network — reservoirs, communities, and the pipelines between them — into a **single-commodity network-flow program**. It decides how much water each route should carry, enforces minimum-service guarantees for priority communities, and compares fairness-aware allocation strategies against four baselines under demand spikes, supply cuts, and pipeline failures. On EPANET benchmark networks, the resulting allocations can be fed back into the **original EPANET model** for a post-hoc pressure and velocity check.

**Research domain:** Algorithms & Theory · Operations Research · Optimization · Water-Resource Management · Resilient Infrastructure

</div>

---

## The Problem

Allocating a limited water supply across many communities is a balancing act with no single right answer.

**Planners** must deliver as much water as possible, but also avoid repeatedly leaving the same communities with severe shortages, adapt when a pipeline fails or supply drops, and still protect communities designated as high priority.

**Existing network-flow and optimization methods** give a foundation, but real-world planning involves several competing requirements at once:

- **Efficiency** — deliver as much of the available water as possible
- **Fairness** — avoid concentrating shortages on a few communities
- **Reliability** — adapt when a pipeline becomes unavailable or supply falls unexpectedly
- **Priority protection** — respect minimum service levels for designated communities
- **Practicality** — find good allocations within a reasonable computation time

---

## The Solution

AquaRoute balances those requirements inside **one optimization framework**.

A user supplies a network in one of four ways:

- **Code** — build it with `add_reservoir`, `add_community`, `add_pipeline`
- **JSON** — load a network file such as `data/sample_network.json`
- **EPANET** — convert a `.inp` file with `load_epanet_inp`, or use the bundled Net1 / Net2 / Net3 benchmarks (via `wntr`, installed with AquaRoute)
- **Dashboard** — start from the built-in sample and tune scenarios interactively

The system outputs:

- A **per-community allocation** and the **flow carried by every pipeline**
- **Five strategy variants** solved on the same network, so they can be compared like for like
- **Validated metrics** — shortage, fairness, Jain's index, priority service, delivery cost, runtime, and constraint checks C1–C5
- A **resilience profile** across demand spikes, supply cuts, and pipeline failures
- **Paired statistics** — tests, effect sizes, bootstrap intervals, and Holm-corrected p-values against each baseline
- An optional **EPANET hydraulic check** of every allocation (pressure, velocity, fidelity to the full-demand baseline)
- **Reproducible experiment exports** — CSV, JSON, and a metadata file recording seed, solver, and runtime
- **Publication figures** generated from the exported results
- An **interactive Streamlit dashboard** for scenario exploration

### The model

The network is a directed graph:

- **Nodes** are reservoirs (supply) or communities (demand)
- **Edges** are pipelines with a **capacity** and a per-unit **delivery cost**
- Water may be **transshipped** through communities, so a community can be served by routing water through its neighbours

Decision variables and constraints:

- `f[e] >= 0` — flow on each pipeline, `f[e] <= capacity[e]`
- `x[c]` — net water delivered to community `c`, `0 <= x[c] <= demand[c]`
- Reservoir balance: `sum(out) - sum(in) <= supply[r]`
- Community balance: `sum(in) - sum(out) = x[c]`
- Junction balance: `sum(in) - sum(out) = 0`

Every strategy first **maximises service** (minimises shortage) subject to any minimum-service guarantee, then breaks ties with a strategy-specific objective. The proportional-fairness metric is **linearised**, so the entire model stays a linear program, solved with **HiGHS** (CBC and GLPK are also tried) — with a **solver-free analytical fallback** when no solver is available or the LP is infeasible.

---

## Core User Flow

```
Water network (code / JSON / EPANET .inp / sample)
        ↓
WaterNetwork — NetworkX DiGraph: reservoirs, communities, pipelines
        ↓
AllocationOptimizer — builds the Pyomo LP
        ↓
Two-stage lexicographic solve
  · Stage 1 — minimise total shortage (maximise service)
  · Stage 2 — optimise the strategy-specific objective
        ↓
HiGHS / CBC / GLPK   ──or──   analytical fallback (no solver / infeasible)
        ↓
Metrics + constraint validation (shortage, fairness, cost, C1–C5)
        ↓
EvaluationSuite — approaches × scenarios × repeats (paired, seeded)
        ↓
(optional) HydraulicValidator — allocations replayed in the original EPANET model
        ↓
CSV + JSON + metadata export  →  figures  /  Streamlit dashboard
```

---

## Features

| Feature | Status |
|---|---|
| Single-commodity network-flow model (reservoir / community / junction balance) | ✅ |
| Transshipment — water routed through intermediate communities | ✅ |
| Five allocation strategies (`shortage_min`, `proportional`, `maxmin`, `uniform`, `weighted`) | ✅ |
| Lexicographic two-stage solve: service first, then strategy objective | ✅ |
| Linearised proportional-fairness objective (stays an LP) | ✅ |
| Minimum-service guarantees for priority communities | ✅ |
| Solver-free analytical fallback when no solver is installed | ✅ |
| Constraint validation (C1–C5) with hard / soft violation handling | ✅ |
| Scenario generation: demand spikes, supply cuts, pipeline failures, combined disruptions | ✅ |
| Resilience analysis, recovery planning, and seeded stress-test suites | ✅ |
| Comparative evaluation against four baselines, with a paired design (identical perturbed networks for every approach) | ✅ |
| Process-independent seeding (SHA-256-derived seeds, not Python's salted `hash`) | ✅ |
| Fairness and service metrics: mean pairwise disparity, Jain's index, priority service | ✅ |
| Statistics: paired t-test + Cohen's d, Wilcoxon, Cliff's delta, bootstrap CI, Holm correction, power analysis | ✅ |
| Sensitivity analysis over `alpha` / `beta` / `gamma` | ✅ |
| Reproducible experiments (seeded) → CSV / JSON / metadata export | ✅ |
| Interactive Streamlit + Plotly dashboard | ✅ |
| Editable install, console script for experiments, GitHub Actions CI (3.10–3.12) | ✅ |
| EPANET `.inp` import with real topology and Hazen–Williams capacities (`load_epanet_inp`) | 🧪 experimental |
| EPANET Net1 / Net2 / Net3 benchmark runs (`--network Net1`) | 🧪 experimental |
| Post-hoc hydraulic validation on the original EPANET model (`HydraulicValidator`, `--hydraulic`) | 🧪 experimental |
| Figure generation (`make_figures.py`, needs `matplotlib`) | 🧪 experimental |
| Scenario-sampled risk analysis (`solve_stochastic`) and multi-period storage (`solve_dynamic`) | 🧪 experimental |

🧪 marks code that exists and runs but has **limited or no automated tests** (the EPANET loader and validator are tested; figures and `solve_stochastic` are not) and the caveats listed in [Experimental Modules](#experimental-modules). Read that section before relying on its output.

---

## Allocation Strategies

Every strategy is a distinct objective on the same network, so differences in results come from the method, not from relabelling one model.

| Strategy | Objective (after maximising service) | Role in the comparison |
|---|---|---|
| `shortage_min` | none — pure maximum delivery | Baseline A |
| `proportional` | minimise proportional disparity | Baseline B |
| `maxmin` | maximise the minimum service ratio (egalitarian) | Baseline C |
| `uniform` | minimise volumetric disparity | Baseline D |
| `weighted` | `alpha * shortage + beta * fairness + gamma * cost` | **AquaRoute** |

AquaRoute additionally supports **minimum-service guarantees** for priority communities via `min_service_priority` and `min_allocation`. In the default comparison the guarantee is **60 % of the smallest demand among the highest-priority communities**, so it scales with the network's units (synthetic demands of ~10–100 and EPANET demands of ~1e-3 m³/s alike). Each community's guarantee is capped at its own demand.

### How the two-stage solve works

1. **Stage 1** minimises total shortage — every strategy starts from the same maximum-service solution.
2. **Stage 2** applies the strategy-specific objective, with shortage pinned near its optimum (absolute tolerance `1e-7`), so no strategy trades away service to look better on its own metric.

Strategies therefore differ only in how they distribute unavoidable shortages and what the routing costs. In the default comparison **all five approaches deliver the same total shortage** (see the example output below), so the comparison measures distribution, cost, and priority service, not total delivery.

### Minimum service as a hard constraint with a safety net

A minimum-service guarantee is a **hard** constraint. When it is physically impossible to satisfy, the LP is infeasible and the **analytical fallback** serves priority communities as far as supply allows, reporting a `C5` warning. In `get_results`, `C1`–`C4` are hard (they trigger the fallback) while `C5` is a soft warning.

---

## Fairness, Metrics & Resilience

**Fairness** is the *mean pairwise proportional disparity*: the mean, over all community pairs, of `|alloc_i/demand_i - alloc_j/demand_j|`. Zero means every community receives the same fraction of its demand; larger values mean greater inequity. Demand normalisation makes communities with different needs comparable. It is linearised with auxiliary variables to keep the model an LP.

Metrics reported per solution:

| Metric | What it tells you |
|---|---|
| `total_allocated` / `shortage` | How much requested water was delivered / left unmet |
| `fairness` | Whether shortages are distributed unevenly (lower is fairer) |
| `jains_fairness` | Jain's index over service ratios (1 = perfectly even) |
| `priority_service` | Service level received by the priority communities |
| `delivery_cost` | True routed cost `sum_e flow_e * cost_e` (or a cheapest-incoming lower bound) |
| `runtime`, `solver`, `status` | What ran, how long it took, and how it ended |
| `validation_passed` | Whether all hard operational constraints are respected |
| `objective_value` | `alpha * shortage + beta * fairness + gamma * cost` |

`EvaluationSuite.compute_summary_metrics()` aggregates these per approach — average, standard deviation, and worst case — plus CVaR across scenarios and a **`feasibility_rate`**.

> **Reading `feasibility_rate` / `feas%`.** In the current code this is the **share of runs with zero shortage** — i.e. the full-service rate. It is *not* the constraint-validation result (`validation_passed`) and not a hydraulic result. Under the default stress scenarios (demand spike, supply cut) a low value is expected for every approach.

### Uncertainty & failure handling

`ScenarioGenerator` builds perturbed copies of a network without mutating the original:

| Scenario | Perturbation |
|---|---|
| `normal_scenario` | unchanged |
| `demand_spike_scenario` | demands × `spike_factor` (default 1.5) |
| `supply_reduction_scenario` | finite reservoir supply × `reduction_factor` (default 0.5); for EPANET sources, which are unbounded, total supply is capped at `reduction_factor` × total demand and split equally across sources |
| `pipeline_failure_scenario` | one or more pipelines removed (both directions) |
| `multiple_disruptions_scenario` | combined disruptions |

`FailureAnalyzer.analyze_resilience` runs an optimizer across a scenario list; `compute_resilience_metrics` summarises recovery; `optimize_recovery` reallocates through surviving infrastructure; `create_stress_test_suite(network, seed=...)` builds a reproducible set of random pipeline failures. Scenario generation, the stress-test suite, and the experiment runner are seeded; `solve_stochastic` (see below) draws from NumPy's global random state and is not.

---

## Research Methodology

The project follows five stages:

1. **Problem formulation** — define the water network, supply limits, demand, and minimum-service requirements.
2. **Mathematical modelling** — express allocation using graph-based flow variables, constraints, and an objective function.
3. **Algorithm design** — balance efficiency, fairness, and resilience in one framework.
4. **Experimental evaluation** — test under normal supply, demand spikes, supply reductions, and pipeline outages, on a synthetic sample network and on EPANET benchmarks.
5. **Baseline comparison** — compare against standard network-flow and optimization methods.

The weighted objective minimises a combination of total shortage `S(x)`, unfairness `F(x)`, and delivery cost `C(x)`:

```
min_x   alpha * S(x) + beta * F(x) + gamma * C(x)
```

subject to source availability, pipeline capacities, and required minimum service levels. `alpha`, `beta`, and `gamma` set the relative importance of each term and are exposed as optimizer parameters.

---

## Evaluation Suite

`EvaluationSuite` runs **approaches × scenarios × repeats** and collects the same metrics for each.

**Default approaches:** the four baselines (A–D) and `aquaroute_weighted_min_service`.
**Default scenarios:** normal, demand spike ×1.5, supply reduction ×0.5, and a two-pipeline failure `(0→2, 1→5)`.

**Paired design.** Each repeat's random perturbation is derived from the scenario and repeat index only (via SHA-256, never Python's per-process `hash`), so every approach is evaluated on the *same* perturbed networks and the same `--seed` reproduces the same numbers across processes.

**Priorities on EPANET networks.** EPANET files carry no priority data. When a network has a single priority level, `EvaluationSuite` assigns demand-based tiers: the top quartile of communities by demand is priority 0, the second quartile priority 1, the rest priority 2. This is a modelling assumption, not data. Priorities already present in a network are preserved.

| Capability | Method |
|---|---|
| Run the comparison | `run_comparison(n_repeats, seed)` |
| Per-approach summary | `compute_summary_metrics()` |
| Paired significance testing | `statistical_significance(metric, approach_a, approach_b)` — t-test with Cohen's d and a confidence interval |
| Multiple-comparison correction | `holm_correction(p_values)` |
| Non-parametric checks | `wilcoxon_paired_test`, `cliffs_delta` |
| Uncertainty on estimates | `bootstrap_ci` |
| Required sample size | `power_analysis(metric, effect_size, alpha, power)` |
| Weight sensitivity | `sensitivity_analysis(metric, approach, param, values)` |
| EPANET hydraulic check | `hydraulic_validation_report()` (🧪, needs an EPANET-sourced network) |
| Export | `export_results_csv`, `export_results_json` |
| Plot | `plot_results(...)` (needs `matplotlib`: `pip install -e ".[figures]"`) |

---

## Example Output

Sample network, `--repeats 3 --seed 42`, HiGHS 1.15.1 (`solver: appsi_highs`). Exact values can differ slightly across solver versions; treat this as an illustration of the table's shape.

```
approach                             shortage  fairness   jain_S  prio_svc      cost   feas%
--------------------------------------------------------------------------------------------
baseline_A_shortage_min               6.80956    0.0793   0.9880    0.9414  105.9276    41.7
baseline_B_proportional               6.80956    0.0540   0.9948    0.9376  107.0308    25.0
baseline_C_maxmin                     6.80956    0.0542   0.9948    0.9378  106.7560    25.0
baseline_D_uniform                    6.80956    0.0739   0.9901    0.9415  107.9050     0.0
aquaroute_weighted_min_service        6.80956    0.0854   0.9846    0.9421  105.5466     0.0
```

Two things to read from this honestly:

- Total shortage is identical across approaches, as the lexicographic design requires. Differences appear in fairness, cost, and priority service.
- AquaRoute is **not** best on every metric. Here it has the lowest cost and the highest priority service, but higher disparity than the proportional and max-min baselines. That trade-off is what the `alpha` / `beta` / `gamma` weights and the minimum-service guarantee control.

---

## Dashboard

An interactive Streamlit app (`aquaroute/scripts/streamlit_dashboard.py`) for exploring a single scenario at a time.

- **Scenario selector** — Normal, Demand Spike, Supply Reduction, Pipeline Failure, Multiple Disruptions
- **Parameter sliders** — `alpha`, `beta`, `gamma`, minimum-service allocation, spike factor, reduction factor, and number of failed pipelines
- **Headline metrics** — total demand, total supply, shortage, and fairness
- **Allocation results** — per-community allocation, plus runtime, objective value, and solver status
- **Multi-scenario comparison** — one click evaluates two weighting configurations (`min_service_first` vs `fairness_weighted`) across scenarios and reports a significance test

The dashboard is a demonstration layer. The five-strategy, four-baseline comparison lives in `EvaluationSuite` and `run_experiments`, and the dashboard does not use the EPANET loader or hydraulic validation.

---

## Experimental Modules

These pieces exist and ran successfully in manual checks (EPANET Net1 and Net3 comparisons, a Net1 hydraulic run, and figure generation), but have only partial automated tests. The loader and `HydraulicValidator` are covered by `test_reproducibility_hydraulics.py`, which runs in CI (`wntr` is now a declared dependency); `make_figures.py` and `solve_stochastic` are untested, and CI does not install `matplotlib`. Treat their output as exploratory, not as a research result.

| Module | What it does today | Caveats |
|---|---|---|
| `load_epanet_inp` (`aquaroute/data/loaders.py`) | Converts an EPANET `.inp` into a `WaterNetwork` using the real topology: reservoirs and tanks become sources, junctions become communities, pipes become links. Pipe capacity comes from pipe geometry through the Hazen–Williams relation `Q = 0.278·C·D^2.63·S^0.54` at a fixed reference gradient `S = 0.001` m/m. Pumps and valves get a large finite capacity. Returns a metadata record of what was preserved and what was invented. | Sources are modelled as **unbounded** (head does not cap volume). Capacity is a **proxy** tied to the reference gradient, not a hydraulic result. Delivery cost is a flat `1.0` per link. Every community is loaded with priority `1`. Links are added in both directions. The `capacity_formula`, `cost_formula`, and `demand_unit` arguments are currently only **recorded in the metadata**; they do not change the conversion. The module imports `wntr` at the top; `wntr` is a declared dependency, so a normal install provides it. |
| `get_benchmark_networks` (`evaluation.py`) | Loads Net1, Net2, and Net3 through `load_epanet_inp` from WNTR's bundled example files and attaches the loader metadata (including `source_path`) to each network. Falls back to three tiny synthetic `sample_N` networks if `wntr` or the files are unavailable. | Inherits every caveat of the loader. Net1–Net3 are standard EPANET *example* networks, not utility data. |
| `HydraulicValidator`, `validate_allocation` (`aquaroute/algorithms/hydraulics.py`) | Replays each allocation as junction demands in the **original EPANET model** and runs the EPANET engine through WNTR. Scenarios are applied physically: demand spikes scale demands, pipeline failures close pipes, supply reductions cap source outflow (flow-control valves on pipe outlets, reduced speed on pumps). Reports pressure against a 20–100 m band, velocity against 2 m/s, new violations versus the full-demand baseline, and how far pressures and flows depart from that baseline. | Needs `metadata["source_path"]` from an EPANET-sourced network. Steady-state, first period only. It is a **post-hoc check**, not part of the optimisation. In a 2-repeat Net1 run all 20 approach × scenario checks reported feasible with minimum pressure around 78–80 m; treat that as a smoke result, not validation. |
| `check_hydraulic_feasibility` (`optimizer.py`) | An older WNTR check that rebuilds a synthetic model (1000 m pipes, 0.3 m diameter, roughness 100) and simulates demands. | Superseded by `HydraulicValidator`. It ignores the `allocations` argument, so its result does not depend on the strategy. |
| `make_figures.py` | Reads the latest `results_<network>_*.json` / `meta_*.json` files for Net1, Net2, and Net3 and writes comparison, scenario, trade-off, fairness, hydraulic, statistical, and resilience figures (PNG and PDF, colour-blind-safe palette). | Needs `matplotlib`. The hydraulic figures need results produced with `--hydraulic`. The overlap checker may print a few label-overlap warnings. The committed `figures/` were generated from earlier runs; regenerate them to reproduce. |
| `AllocationOptimizer.solve_stochastic` | Draws demand and supply multipliers in 0.7–1.3, solves each scenario independently, averages the allocations, and reports the CVaR of the scenario shortages. | CVaR is **measured after the fact, not optimised** (no Rockafellar–Uryasev formulation). Randomness is not seeded by the method. |
| `AllocationOptimizer.solve_dynamic` | Loops over periods and tracks a clamped storage level `S(t+1) = S(t) + supply − allocated`. | Storage is bookkeeping, not an LP constraint, and the per-period demands are not applied to the network that is solved. Its shortage and storage numbers are not evidence about the method. |

---

## Architecture

```mermaid
flowchart LR
    A["🗺️ Water network<br/>code / JSON / EPANET / sample"]:::gray
    B["🔗 WaterNetwork<br/>NetworkX graph"]:::teal
    C["🧮 AllocationOptimizer<br/>Pyomo LP"]:::blue
    D["⚡ HiGHS / CBC / GLPK<br/>solver"]:::amber
    E["🧯 Analytical fallback<br/>no solver / infeasible"]:::green
    F["📏 Metrics &amp; validation<br/>C1–C5"]:::purple
    G["📊 EvaluationSuite<br/>approaches × scenarios"]:::blue
    H["🌪️ Scenarios &amp; resilience<br/>spikes · cuts · failures"]:::purple
    K["💧 HydraulicValidator<br/>EPANET via WNTR"]:::teal
    I["📄 CSV / JSON export<br/>+ metadata"]:::gray
    L["📈 Figures"]:::gray
    J["🖥️ Streamlit dashboard"]:::green

    A --> B
    B --> C
    C --> D
    C --> E
    D --> F
    E --> F
    H --> G
    C --> G
    F --> G
    G --> K
    G --> I
    K --> I
    I --> L
    G --> J

    classDef gray   fill:#e8e6e1,stroke:#9c9a92,color:#2C2C2A
    classDef teal   fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef blue   fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    classDef amber  fill:#FAEEDA,stroke:#854F0B,color:#633806
    classDef purple fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef green  fill:#EAF3DE,stroke:#3B6D11,color:#173404
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Main research software |
| Graph modelling | [NetworkX](https://networkx.org/) | Reservoirs, communities and pipelines as a directed graph |
| Optimization | [Pyomo](https://www.pyomo.org/) | Express allocation objectives and constraints |
| Solver | HiGHS (via `highspy`), CBC, or GLPK | Solve the linear program |
| Data processing | NumPy, pandas, SciPy | Demands, capacities, simulation data, statistical tests |
| Hydraulics | [WNTR](https://usepa.github.io/WNTR/) (EPANET engine) | EPANET import and hydraulic validation — core dependency (plus `six`, which wntr imports but does not declare) |
| Figures | Matplotlib | `make_figures.py` and `plot_results` — optional `figures` extra |
| Dashboard | Streamlit, Plotly | Interactive demonstration |
| Testing | pytest, pytest-cov | Algorithm behaviour and constraints |
| Linting | Ruff (configured in `pyproject.toml`) | Correctness and import hygiene |
| CI | GitHub Actions | pytest on Python 3.10 / 3.11 / 3.12 plus an experiment smoke test on the sample network |

---

## Project Structure

```
AquaRoute/
│
├── aquaroute/
│   ├── __init__.py
│   ├── models/
│   │   └── water_network.py        # NetworkX graph model, JSON / dict loading, sample network
│   ├── algorithms/
│   │   ├── optimizer.py            # Network-flow LP, 5 strategies, metrics, validation, fallback
│   │   ├── uncertainty.py          # Scenario generation, resilience analysis, recovery planning
│   │   ├── evaluation.py           # Paired comparison, statistics, sensitivity, export, benchmarks
│   │   └── hydraulics.py           # EPANET hydraulic validation of allocations (experimental)
│   ├── data/
│   │   └── loaders.py              # EPANET .inp → WaterNetwork loader (experimental, needs wntr)
│   ├── scripts/
│   │   ├── run_experiments.py      # Reproducible experiment runner (CLI)
│   │   ├── make_figures.py         # Figure generation from exported results (experimental)
│   │   └── streamlit_dashboard.py  # Interactive demo
│   └── tests/
│       ├── test_optimizer.py       # supply / pipeline binding, transshipment, min service, cost
│       ├── test_uncertainty.py     # scenarios, resilience, seeded reproducibility
│       ├── test_water_network.py   # graph construction + JSON loading
│       ├── test_evaluation.py      # baseline distinctness, export, summary metrics
│       └── test_reproducibility_hydraulics.py  # seed derivation, EPANET loader, hydraulic validation
│
├── data/
│   └── sample_network.json         # Synthetic benchmark: 2 reservoirs, 5 communities, 9 pipelines
│
├── figures/                        # PNG + PDF output of make_figures.py (regenerate to reproduce)
│
├── .github/workflows/ci.yml        # CI: pytest matrix (3.10–3.12) + experiment smoke test
├── requirements.txt                # Pinned runtime dependencies (incl. wntr, six, HiGHS)
├── requirements-dev.txt            # Dashboard + test tooling
├── pyproject.toml                  # Packaging, extras, console scripts, pytest + ruff config
├── research_project.odt            # Research proposal and project background
├── LICENSE                         # MIT
└── readme.md
```

---

## Run Locally

**Step 1 — Clone and install**

```bash
git clone https://github.com/ria0304/AquaRoute.git
cd AquaRoute

# Editable install with the HiGHS solver (recommended)
pip install -e ".[solver]"

# …or a pinned, known-good set
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e ".[dashboard]"   # Streamlit + Plotly
pip install -e ".[dev]"         # pytest + coverage
pip install -e ".[figures]"     # Matplotlib, for make_figures.py and plot_results
```

`wntr` (EPANET import, benchmarks, hydraulic validation) and its undeclared `six` dependency are installed automatically by both commands above.

Without a solver backend installed, AquaRoute still runs — it automatically uses the built-in analytical fallback. **That fallback is a heuristic, so check the `solver` field in the output before drawing conclusions** (see [Verify it works](#verify-it-works)).

**Step 2 — Run an experiment**

```bash
# Sample network
python -m aquaroute.scripts.run_experiments --repeats 5 --seed 42 --outdir results

# EPANET benchmark, with hydraulic validation (needs wntr)
python -m aquaroute.scripts.run_experiments --network Net1 --repeats 5 --seed 42 --outdir results --hydraulic

# or, after installing: aquaroute-experiments --repeats 5 --seed 42 --outdir results
```

**Step 3 — Generate figures (optional)**

```bash
python -m aquaroute.scripts.make_figures --results-dir results --figures-dir figures
```

**Step 4 — Launch the dashboard**

```bash
streamlit run aquaroute/scripts/streamlit_dashboard.py
```

---

## Verify it works

```bash
pytest
```

Then a one-minute check on the bundled network:

```python
from aquaroute.models.water_network import WaterNetwork
from aquaroute.algorithms.optimizer import AllocationOptimizer

network = WaterNetwork.from_json("data/sample_network.json")
print(network)                              # WaterNetwork: 2 reservoirs, 5 communities, 9 pipelines

opt = AllocationOptimizer(network, strategy="shortage_min")
opt.solve()
r = opt.get_results()
print(r["solver"], r["shortage"], r["validation_passed"])
```

`solver` reports the backend that actually ran. It should read `appsi_highs` (or `highs` / `cbc` / `glpk`); `analytical_fallback` means no LP solver was found or the LP failed. Check it on your own machine: `python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('appsi_highs').available(exception_flag=False))"` should print `True`. Tests that need an LP solver are skipped automatically when none is installed.

---

## Reproducible Experiments

| Flag | Description |
|---|---|
| `--network` | `Net1`, `Net2`, `Net3` (EPANET benchmarks), or a path to a JSON network file. Empty uses the built-in sample network. |
| `--repeats` | Independent repeats per scenario, averaged (default **20**) |
| `--seed` | Base random seed (default 42) |
| `--outdir` | Output directory, created if needed (default `results`) |
| `--label` | Optional label appended to output filenames |
| `--hydraulic` | Replay every allocation in the original EPANET model (needs an EPANET benchmark) |

The runner evaluates every approach on every scenario, prints a summary table (shortage, fairness, Jain's index, priority service, cost, full-service rate), runs paired significance tests of AquaRoute against each baseline with Holm correction, and writes three files:

- `results_<network>_<timestamp>.csv` — per-approach summary metrics
- `results_<network>_<timestamp>.json` — full results, including per-repeat observations
- `meta_<network>_<timestamp>.json` — seed, repeats, runtime, solver, summary, significance results, hydraulic report, scenario types

The same `--seed` reproduces the same results, including across separate processes. Net3 is much slower than the sample network (about a minute for 3 repeats on a laptop), so use a small `--repeats` for a first try.

### Network file format

```json
{
  "reservoirs":  [{"node_id": 0, "capacity": 100.0, "supply": 100.0}],
  "communities": [{"node_id": 2, "demand": 20.0, "priority": 1}],
  "pipelines":   [{"from_node": 0, "to_node": 2, "capacity": 30.0, "delivery_cost": 1.0}]
}
```

`data/sample_network.json` is a complete example: 2 reservoirs, 5 communities, and 9 pipelines, including community-to-community links for transshipment.

---

## Programmatic Use

```python
from aquaroute.models.water_network import WaterNetwork
from aquaroute.algorithms.optimizer import AllocationOptimizer
from aquaroute.algorithms.evaluation import EvaluationSuite

network = WaterNetwork.from_json("data/sample_network.json")

optimizer = AllocationOptimizer(
    network,
    strategy="weighted",          # shortage_min | proportional | maxmin | uniform | weighted
    alpha=1.0, beta=1.0, gamma=1.0,
    min_service_priority=[2, 4],  # priority community IDs
    min_allocation=10.0,
)
optimizer.solve()
results = optimizer.get_results()

print(results["shortage"], results["fairness"], results["delivery_cost"])
print(results["validation_passed"], results["validation_warnings"])

# Comparative evaluation across approaches and scenarios
suite = EvaluationSuite(network=network, seed=42)
comparison = suite.run_comparison(n_repeats=5, seed=42)
summary = suite.compute_summary_metrics()
```

EPANET benchmark with hydraulic validation:

```python
from aquaroute.algorithms.evaluation import EvaluationSuite, get_benchmark_networks

net = get_benchmark_networks()["Net1"]      # metadata (incl. source_path) is attached
suite = EvaluationSuite(network=net, seed=42)
suite.run_comparison(n_repeats=3, seed=42)
report = suite.hydraulic_validation_report()
```

---

## Configuration Reference

**`AllocationOptimizer(network, ...)`**

| Parameter | Default | Description |
|---|---|---|
| `strategy` | `"weighted"` | One of `weighted`, `shortage_min`, `uniform`, `maxmin`, `proportional` |
| `alpha` | `1.0` | Weight on total shortage (used by `weighted`) |
| `beta` | `1.0` | Weight on unfairness (used by `weighted`) |
| `gamma` | `1.0` | Weight on delivery cost (used by `weighted`) |
| `min_service_priority` | `None` | Community IDs that must receive a guaranteed minimum (zero-demand communities are skipped) |
| `min_allocation` | `0.0` | The guaranteed minimum for those communities, capped at each community's demand |
| `solver_name` | `None` | Force a solver; otherwise tries `appsi_highs`, `highs`, `cbc`, `glpk` |

**`.solve(solver_name=None, time_limit=None, allow_fallback=True)`** — set `allow_fallback=False` to get `None` instead of a heuristic answer when no solver is available or the LP is infeasible.

**`EvaluationSuite(network, approaches=None, scenarios=None, seed=42)`** — pass your own `approaches` (a list of `(name, optimizer_kwargs)`) or `scenarios` to override the defaults.

**Packaging extras (`pyproject.toml`)**

| Extra | Installs | Needed for |
|---|---|---|
| `solver` | `highspy` | Exact LP solves (otherwise the analytical fallback is used) |
| `dashboard` | `streamlit`, `plotly` | The interactive dashboard |
| `figures` | `matplotlib` | `make_figures.py` and `plot_results` |
| `dev` | `pytest`, `pytest-cov` | Running the test suite |

`wntr` and `six` are core dependencies (not extras). `matplotlib` is only needed for figures, via the `figures` extra.

---

## API Reference

**Model — `aquaroute.models.water_network`**

| Item | Description |
|---|---|
| `WaterNetwork()` | Directed graph of reservoirs, communities, and pipelines |
| `.add_reservoir(node_id, capacity, supply)` | Add a supply node |
| `.add_community(node_id, demand, priority)` | Add a demand node |
| `.add_pipeline(from_node, to_node, capacity, delivery_cost)` | Add a capacitated, costed edge |
| `.get_reservoirs()` / `.get_communities()` | Node ID lists |
| `.total_demand()` / `.total_supply()` | Network totals |
| `WaterNetwork.from_json(path)` / `.from_dict(data)` | Load a network |
| `create_sample_network()` | The built-in 2-reservoir, 5-community benchmark |

**Optimizer — `aquaroute.algorithms.optimizer`**

| Item | Description |
|---|---|
| `AllocationOptimizer` | Builds and solves the network-flow LP |
| `.build_model()` | Build the Pyomo model |
| `.solve(...)` | Two-stage lexicographic solve with analytical fallback |
| `.get_results()` | Allocations, flows, metrics, validation, runtime |
| `compute_shortage`, `compute_fairness`, `compute_delivery_cost` | Standalone metric functions |
| `validate_solution` | Constraint checks C1–C5 |
| `effective_capacities`, `max_total_deliverable` | Upper bounds on what the network can deliver |
| `compute_cvar(losses, alpha)` | CVaR of a list of losses (a reported metric, not an optimisation objective) |
| 🧪 `.solve_stochastic(...)`, `.solve_dynamic(...)` | Experimental — see [Experimental Modules](#experimental-modules) |
| 🧪 `check_hydraulic_feasibility(network, allocations, demands)` | Older synthetic WNTR check; prefer `HydraulicValidator` |

**Uncertainty — `aquaroute.algorithms.uncertainty`**

| Item | Description |
|---|---|
| `ScenarioGenerator` | Normal, demand-spike, supply-reduction, pipeline-failure, and combined scenarios |
| `apply_scenario(network, scenario)` | Mutate a network in place to match a scenario |
| `FailureAnalyzer.analyze_resilience` | Run an optimizer across scenarios |
| `FailureAnalyzer.compute_resilience_metrics` | Summarise recovery across results |
| `FailureAnalyzer.optimize_recovery` | Reallocate through surviving infrastructure |
| `create_stress_test_suite(network, seed=...)` | Reproducible random-failure suite |

**Evaluation — `aquaroute.algorithms.evaluation`**

| Item | Description |
|---|---|
| `EvaluationSuite` | Paired approaches × scenarios × repeats, with statistics and export |
| `compare_approaches(network, ...)` | One-call convenience wrapper (20 repeats) |
| `jains_fairness_index`, `compute_cvar_from_scenarios` | Extra fairness and tail-risk metrics |
| 🧪 `get_benchmark_networks()` | EPANET Net1–Net3 via the loader, or three tiny sample networks as a fallback |
| 🧪 `EvaluationSuite.hydraulic_validation_report()` | Run the EPANET check for every approach and scenario |

**Data & hydraulics (experimental, need `wntr`)**

| Item | Description |
|---|---|
| 🧪 `aquaroute.data.loaders.load_epanet_inp(path)` | EPANET import returning `(network, metadata)` |
| 🧪 `aquaroute.data.loaders.hazen_williams_capacity(...)` | Pipe capacity proxy from geometry and roughness |
| 🧪 `aquaroute.algorithms.hydraulics.HydraulicValidator` | Replay allocations in the original EPANET model under a scenario |
| 🧪 `aquaroute.algorithms.hydraulics.validate_allocation(...)` | One-call wrapper around the validator |

---

## Tests

```bash
pytest
```

The suite (60 tests) verifies:

- **Optimiser correctness** — supply / pipeline binding, transshipment routing, minimum service
- **Metric boundary conditions** — fairness and shortage at the extremes; delivery cost is not double-counted
- **Baseline distinctness** — strategies do not collapse to the same allocation
- **Weight sensitivity** — `alpha` / `beta` / `gamma` change the solution as expected
- **Reproducibility** — identical seeds give identical results, with process-independent seed derivation and paired seeds across approaches
- **EPANET loader fidelity and hydraulic validation** — Net2 demand clamping, WNTR name mapping, demand conservation, and pipeline-failure / demand-spike handling in `HydraulicValidator`
- **Result export** — CSV / JSON round-trips

**Not covered by tests:** `make_figures.py`, `solve_stochastic`, `solve_dynamic`, and the equal-total-shortage property across strategies. CI runs `pytest` on Python 3.10, 3.11, and 3.12, then a one-repeat experiment on the sample network as a smoke test. It installs `.[solver]` and `pytest`; `wntr` comes in as a core dependency, so the EPANET loader and validator tests run there too, but `matplotlib` is not installed. To lint locally, install Ruff yourself (`pip install ruff`) and run `ruff check aquaroute`.

---

## Common Issues

| Problem | Fix |
|---|---|
| `solver` shows `analytical_fallback` | No LP solver was found or the LP failed. Install HiGHS with `pip install -e ".[solver]"` and check with the `SolverFactory('appsi_highs')` one-liner in [Verify it works](#verify-it-works). Make sure you are using the same Python environment you installed into. |
| `ModuleNotFoundError` on startup | Re-run `pip install -r requirements.txt` |
| Solver-dependent tests are skipped | Expected without a solver; install the `solver` extra to run them |
| Infeasible LP | A minimum-service guarantee may be impossible; check `validation_warnings` for `C5` |
| Dashboard won't launch | `pip install -e ".[dashboard]"`, then use `streamlit run aquaroute/scripts/streamlit_dashboard.py` |
| `ModuleNotFoundError: wntr` (or `six`) | Core dependencies; re-run `pip install -e ".[solver]"` or `pip install -r requirements.txt` |
| `--network Net1` says the benchmark could not be loaded | Same cause — install `wntr` |
| `ModuleNotFoundError: matplotlib` | Needed for `make_figures.py` and `plot_results`; `pip install -e ".[figures]"` |
| `--hydraulic` reports errors | It needs an EPANET benchmark network (`Net1`/`Net2`/`Net3`), not the sample or a JSON file |
| `make_figures` says no results found | Run `run_experiments` first, with `--outdir` matching `--results-dir`, for Net1, Net2, and/or Net3 |
| Net3 run is slow | Lower `--repeats`; the default is 20 |
| Results not reproducible | Pass the same `--seed` to `run_experiments` |
| Mermaid diagram looks cut off on GitHub | Use the zoom / pan controls, or open the README full-screen |

---

## Deployment

Local-only. The project runs from the command line and a local Streamlit server — see **Run Locally** above. There is no Dockerfile in this repository, and the `results/` output folder is gitignored.

---

## Known Limitations

- **Volumetric flow plus a post-hoc hydraulic check.** The LP models flow, capacity, and cost. Pressure, head loss, water quality, and pipe sizing are not part of the optimisation. `HydraulicValidator` checks the finished allocations against EPANET afterwards (steady-state, first period); it does not feed anything back into the optimiser.
- **EPANET conversion rests on assumptions.** Sources are unbounded, pipe capacity is a Hazen–Williams proxy at a fixed reference gradient, delivery cost is a flat 1.0, and priorities come from demand quartiles rather than data. Shortage on these networks is therefore driven by the capacity proxy and by the supply-cap scenario, not by real supply limits.
- **Total shortage is identical across the default approaches by design.** The comparison shows how strategies distribute shortage and what routing costs, not who delivers more water. It does not show that AquaRoute outperforms the baselines overall; on the sample network it has higher disparity than the proportional and max-min baselines.
- **`feasibility_rate` means "no shortage".** It is the full-service rate, not constraint or hydraulic feasibility.
- **Costs are estimates.** Delivery cost uses a per-unit edge cost, or a cheapest-incoming lower bound when only volumetric allocations are known.
- **The fallback is heuristic.** The analytical fallback (used without a solver or on infeasibility) prioritises service as far as supply allows but is not guaranteed optimal. Always check the reported `solver`.
- **Fairness is one metric in the objective.** Proportional disparity drives the model; Jain's index and CVaR are reported but not optimised, and other equity notions (e.g. Gini) are not modelled.
- **Scenario-based uncertainty.** Disruptions are sampled or specified scenarios, re-solved one at a time. CVaR is computed from the resulting shortages; it is not optimised, and the earlier `cvar_robust` strategy is not available, so this is not a stochastic or robust optimisation model.
- **No multi-period model yet.** `solve_dynamic` tracks storage but does not enforce it in the LP, so the project does not currently model shortages carried across time.
- **Parts of the experimental code are untested.** See [Experimental Modules](#experimental-modules). `make_figures.py` and `solve_stochastic` have no automated tests.
- **Dashboard comparison is narrower than the suite.** The dashboard compares two weighting configurations; the full baseline comparison runs through `EvaluationSuite` and `run_experiments`.



---

## Future Scope

| Feature | Why |
|---|---|
| Pressure & hydraulic constraints in the model | Move from a post-hoc hydraulic check to hydraulically aware routing |
| Richer equity metrics | Optimise against Gini and priority-weighted fairness, not just report them |
| Stochastic / robust optimization | Model supply and demand uncertainty directly (e.g. a proper CVaR or budgeted-robust formulation) rather than by scenario sampling |
| History-aware fairness | Let past shortages shift who bears the next one, so the same communities are not repeatedly short — the stated motivation, not yet modelled |

---

## License

Released under the [MIT License](LICENSE).
