## AquaRoute — Algorithmic Optimization for Reliable Community Water Distribution

AquaRoute is a software research project on allocating limited water across
communities under scarce supply, changing demand, and network disruptions. It
formulates the problem as a **single-commodity network-flow program** and
compares fairness-aware, minimum-service allocation strategies against
established baselines.

**Research domain:** Algorithms & Theory, Operations Research, Optimization,
Water-Resource Management, Resilient Infrastructure.

### What the model does

Nodes are reservoirs (supply) or communities (demand). Edges are pipelines with
a capacity and a per-unit delivery cost. Water may be **transshipped** through
communities, so a community can be served by routing water through its
neighbours.

Decision variables and constraints:

- `f[e] >= 0` — flow on each pipeline, `f[e] <= capacity[e]`.
- `x[c]` — net water delivered to community `c`, `0 <= x[c] <= demand[c]`.
- Reservoir balance: `sum(out) - sum(in) <= supply[r]`.
- Community balance: `sum(in) - sum(out) = x[c]`.
- Junction balance: `sum(in) - sum(out) = 0`.

Every strategy first **maximises service** (minimises shortage) subject to a
(possible) minimum-service guarantee, then breaks ties with a
strategy-specific objective. The proportional-fairness metric is linearised so
the whole model remains a linear program, solved with HiGHS (with a
solver-free analytical fallback).

### Allocation strategies

| Strategy | Objective (after maximising service) |
|---|---|
| `shortage_min` | none — pure maximum delivery (baseline A) |
| `proportional` | minimise proportional disparity (baseline B) |
| `maxmin` | maximise the minimum service ratio — egalitarian (baseline C) |
| `uniform` | minimise volumetric disparity (baseline D) |
| `weighted` | `beta * fairness + gamma * cost` (AquaRoute) |

AquaRoute additionally supports **minimum-service guarantees** for priority
communities (`min_service_priority`, `min_allocation`). When the guarantee is
physically impossible the LP is infeasible and the analytical fallback serves
priority communities as far as supply allows, reporting a `C5` warning.

### Installation

```bash
pip install -e ".[solver]"        # research install with HiGHS solver
pip install -r requirements.txt   # or a pinned set of known-good versions
```

The dashboard is optional: `pip install -e ".[dashboard]"`.

### Reproducible experiments

```bash
python -m aquaroute.scripts.run_experiments --repeats 5 --seed 42 --outdir results
```

This runs every strategy on every scenario and writes a CSV, a JSON results
file, and a metadata file (seed, repeats, runtime, solver, summary). All
randomness is seeded, so runs with the same `--seed` reproduce exactly.

A synthetic benchmark network is provided in `data/sample_network.json`; load
it with `WaterNetwork.from_json("data/sample_network.json")`.

### Dashboard

```bash
aquaroute-dashboard
# or: streamlit run aquaroute/scripts/streamlit_dashboard.py
```

### Tests

```bash
pytest
```

The suite verifies optimiser correctness (supply/pipeline binding,
transshipment, minimum service), metric boundary conditions, baseline
distinctness, weight sensitivity, reproducibility, and result export.

### Project layout

```
aquaroute/
  models/water_network.py        graph model + JSON loading
  algorithms/optimizer.py        network-flow LP, strategies, metrics, validation
  algorithms/uncertainty.py      scenario generation + resilience analysis
  algorithms/evaluation.py       comparative evaluation + statistical tests
  scripts/run_experiments.py     reproducible experiment runner
  scripts/streamlit_dashboard.py interactive demo
  tests/                         pytest suite
data/sample_network.json         synthetic benchmark network
```

**Project status:** research prototype with reproducible evaluation. Novelty
relative to existing robust water-distribution optimization is still to be
established through a dedicated literature review.
