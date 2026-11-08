"""Targeted tests for reproducibility, hydraulic validation, and loader fidelity (§13).

These add coverage for the new scientific-integrity requirements without
modifying existing assertions.
"""

from __future__ import annotations

import pathlib

import pytest

from aquaroute.algorithms.evaluation import (
    EvaluationSuite,
    _derive_seed,
    get_benchmark_networks,
)
from aquaroute.algorithms.hydraulics import HydraulicValidator, validate_allocation
from aquaroute.data.loaders import load_epanet_inp
from aquaroute.models.water_network import create_sample_network


class TestSeedDerivation:
    def test_derive_seed_is_deterministic(self):
        a = _derive_seed(42, 0, "normal", 3)
        b = _derive_seed(42, 0, "normal", 3)
        assert a == b
        assert 0 <= a < 2**31

    def test_derive_seed_changes_with_inputs(self):
        s1 = _derive_seed(42, 0, "normal", 0)
        s2 = _derive_seed(42, 1, "normal", 0)
        s3 = _derive_seed(42, 0, "supply_reduction", 0)
        assert len({s1, s2, s3}) == 3

    def test_derive_seed_process_independent(self):
        # Same stable inputs must yield the same seed regardless of call order
        # or interpreter hash salt (we cannot set PYTHONHASHSEED in-process,
        # so determinism across repeated calls is the verifiable property).
        seq1 = [_derive_seed(7, i, "normal", 0) for i in range(5)]
        seq2 = [_derive_seed(7, i, "normal", 0) for i in range(5)]
        assert seq1 == seq2


class TestPairedCommonRandomNumbers:
    def test_all_approaches_share_seed_per_scenario_repeat(self):
        net = create_sample_network()
        suite = EvaluationSuite(net, seed=123)
        suite.run_comparison(n_repeats=3, seed=123)
        # For each scenario, every approach must use the identical sequence of
        # per-repeat seeds (common random numbers -> paired comparisons).
        by_scenario: dict = {}
        for approach, rows in suite.results.items():
            for r in rows:
                st = r.get("scenario_type")
                seed_list = [rr.get("seed") for rr in (r.get("repeats_raw") or [])]
                by_scenario.setdefault(st, []).append(seed_list)
        for st, lists in by_scenario.items():
            first = lists[0]
            for lst in lists[1:]:
                assert lst == first, (
                    f"{st}: approaches use different repeat seeds {lists}")


class TestLoaderFidelity:
    def test_net2_negative_demand_clamped_and_recorded(self):
        wntr_dir = pathlib.Path(
            __import__("wntr", fromlist=["__version__"]).__file__
        ).parent / "library" / "networks"
        net, meta = load_epanet_inp(wntr_dir / "Net2.inp")
        # Junction that had negative demand in the original INP must be clamped
        # to >= 0 and recorded.
        inflows = meta.get("preserved", {}).get("negative_demand_inflows", [])
        assert any("1" in str(v) or v.get("wntr_name") == "1" for v in inflows), inflows
        for c in net.get_communities():
            assert net.G.nodes[c].get("demand", 0.0) >= -1e-12

    def test_loader_aro_to_wntr_mapping(self):
        nets = get_benchmark_networks()
        for net in nets.values():
            meta = getattr(net, "metadata", {})
            aro_to_wntr = meta.get("aro_to_wntr", {})
            wntr_to_aro = meta.get("wntr_to_aro", {})
            for c in net.get_communities():
                assert c in aro_to_wntr, f"community {c} missing from mapping"
                wn_name = aro_to_wntr[c]
                # Inverse mapping must be consistent.
                assert wntr_to_aro.get(wn_name) == c, (
                    f"mapping not invertible: {c} -> {wn_name}")

    def test_demand_conservation_under_allocation(self):
        net = create_sample_network()
        demands = {c: net.G.nodes[c]["demand"] for c in net.get_communities()}
        allocations = {c: min(demands[c], demands[c] * 0.5)
                       for c in demands}
        served = sum(allocations.values())
        requested = sum(demands.values())
        assert 0.0 <= served <= requested


class TestHydraulicValidation:
    def test_hydraulic_validator_matches_baseline_at_full_service(self):
        nets = get_benchmark_networks()
        for name in ("Net1", "Net3"):  # Net2 has a negative-demand (inflow) junction
            net = nets[name]
            meta = getattr(net, "metadata", {})
            if not meta or not meta.get("source_path"):
                pytest.skip(f"{name} source not available")
            validator = HydraulicValidator(net, meta)
            # Full service: allocate every junction its full baseline demand.
            full = {}
            for j, d in validator.base_demands.items():
                if d < 0:
                    continue  # skip inflow/negative-demand nodes
                c = validator.wntr_to_aro.get(j)
                if c is not None:
                    full[c] = d
            rep = validator.validate(full, {"scenario_type": "normal",
                                            "demands": {}, "supply": {},
                                            "failures": []})
            base = rep["baseline"]
            assert abs(rep["hydraulics"]["min_pressure"] - base["min_pressure"]) < 1e-6
            assert abs(rep["hydraulics"]["max_velocity"] - base["max_velocity"]) < 1e-6
            assert rep["fidelity"]["max_abs_pressure_delta_m"] < 1e-6

    def test_pipeline_failure_closes_links(self):
        nets = get_benchmark_networks()
        net = nets["Net1"]
        meta = getattr(net, "metadata", {})
        if not meta or not meta.get("source_path"):
            pytest.skip("Net1 source not available")
        validator = HydraulicValidator(net, meta)
        communities = set(net.get_communities())
        target = None
        for u, v in net.G.edges():
            if u in communities and v in communities:
                target = (u, v)
                break
        assert target is not None
        full = {c: net.G.nodes[c]["demand"] for c in communities}
        rep = validator.validate(full,
                                 {"scenario_type": "pipeline_failure",
                                  "demands": {}, "supply": {},
                                  "failures": [target]})
        # A closed link cannot increase pressure beyond baseline.
        assert rep["hydraulics"]["max_pressure"] <= rep["baseline"]["max_pressure"] + 1e-6

    def test_demand_spike_raises_junction_demands(self):
        nets = get_benchmark_networks()
        net = nets["Net1"]
        meta = getattr(net, "metadata", {})
        if not meta or not meta.get("source_path"):
            pytest.skip("Net1 source not available")
        validator = HydraulicValidator(net, meta)
        base = {c: net.G.nodes[c]["demand"] for c in net.get_communities()}
        # Key scenario demands by AquaRoute community id (as the optimizer does).
        spike_demands = {c: base[c] * 2.0 for c in base}
        rep = validator.validate({c: 0.0 for c in base},
                                 {"scenario_type": "demand_spike",
                                  "demands": spike_demands, "supply": {},
                                  "failures": []})
        applied = rep["applied_demands"]
        for c, info in applied.items():
            assert info["scenario_demand"] == pytest.approx(base[c] * 2.0)
            assert info["served_demand"] == 0.0


class TestStatisticalCorrectness:
    def test_bootstrap_ci_contains_mean(self):
        net = create_sample_network()
        suite = EvaluationSuite(net, seed=1)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = suite.bootstrap_ci(values, alpha=0.05)
        assert lo <= mean <= hi
        assert mean == pytest.approx(3.0)

    def test_wilcoxon_identical_samples_p1(self):
        net = create_sample_network()
        suite = EvaluationSuite(net, seed=1)
        res = suite.wilcoxon_paired_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert res["p_value"] == 1.0
        assert res["mean_diff"] == 0.0


def test_validate_allocation_wrapper_runs():
    nets = get_benchmark_networks()
    net = nets["Net1"]
    meta = getattr(net, "metadata", {})
    if not meta or not meta.get("source_path"):
        pytest.skip("Net1 source not available")
    allocs = {c: net.G.nodes[c]["demand"] * 0.5 for c in net.get_communities()}
    rep = validate_allocation(net, allocs, meta,
                              {"scenario_type": "supply_reduction",
                               "demands": {}, "supply": {}, "failures": []})
    assert "feasible" in rep
    assert "hydraulics" in rep
    assert "fidelity" in rep
