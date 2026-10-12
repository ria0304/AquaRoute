from __future__ import annotations

import copy

import pytest

from aquaroute.algorithms.optimizer import AllocationOptimizer
from aquaroute.algorithms.uncertainty import (
    FailureAnalyzer,
    ScenarioGenerator,
    apply_scenario,
    create_stress_test_suite,
)
from aquaroute.models.water_network import WaterNetwork, create_sample_network


class TestScenarioGenerator:
    def test_normal_scenario(self):
        net = create_sample_network()
        sc = ScenarioGenerator.normal_scenario(net)
        assert sc["scenario_type"] == "normal"
        assert sc["failures"] == []
        assert len(sc["demands"]) == len(net.get_communities())

    def test_demand_spike(self):
        net = create_sample_network()
        base = {c: net.G.nodes[c]["demand"] for c in net.get_communities()}
        sc = ScenarioGenerator.demand_spike_scenario(net, 1.5)
        assert sc["scenario_type"] == "demand_spike"
        for c, d in sc["demands"].items():
            assert d == pytest.approx(base[c] * 1.5)
        # Original network must not be mutated.
        for c in net.get_communities():
            assert net.G.nodes[c]["demand"] == pytest.approx(base[c])

    def test_supply_reduction(self):
        net = create_sample_network()
        base = {r: net.G.nodes[r]["supply"] for r in net.get_reservoirs()}
        sc = ScenarioGenerator.supply_reduction_scenario(net, 0.5)
        assert sc["scenario_type"] == "supply_reduction"
        for r, s in sc["supply"].items():
            assert s == pytest.approx(base[r] * 0.5)

    def test_pipeline_failure_only_existing_edges(self):
        net = create_sample_network()
        sc = ScenarioGenerator.pipeline_failure_scenario(net, [(0, 2), (99, 100)])
        assert sc["failures"] == [(0, 2)]

    def test_apply_scenario_mutates_copy(self):
        net = create_sample_network()
        sc = ScenarioGenerator.supply_reduction_scenario(net, 0.5)
        net_copy = WaterNetwork()
        net_copy.G = copy.deepcopy(net.G)
        apply_scenario(net_copy, sc)
        for r, s in sc["supply"].items():
            assert net_copy.G.nodes[r]["supply"] == pytest.approx(s)
        # Original untouched.
        assert net_copy.G.nodes[0]["supply"] != net.G.nodes[0]["supply"]

    def test_multiple_disruptions(self):
        net = create_sample_network()
        sc = ScenarioGenerator.multiple_disruptions_scenario(
            net, spike_factor=1.2, reduction_factor=0.7, failed_edges=[(0, 2)])
        assert sc["scenario_type"] == "multiple_disruptions"
        assert sc["failures"] == [(0, 2)]


class TestFailureAnalyzer:
    def test_resilience_analysis_records_all_scenarios(self):
        net = create_sample_network()
        results = FailureAnalyzer.analyze_resilience(
            network=net,
            optimizer_class=AllocationOptimizer,
            optimizer_kwargs={"strategy": "weighted"},
            scenarios=[ScenarioGenerator.normal_scenario(net),
                       ScenarioGenerator.supply_reduction_scenario(net, 0.4)])
        assert len(results) == 2
        assert all("results" in r for r in results)
        assert [r["scenario_type"] for r in results] == ["normal", "supply_reduction"]

    def test_resilience_metrics(self):
        net = create_sample_network()
        results = FailureAnalyzer.analyze_resilience(
            network=net,
            optimizer_class=AllocationOptimizer,
            optimizer_kwargs={"strategy": "weighted"})
        metrics = FailureAnalyzer.compute_resilience_metrics(results)
        for key in ("shortage_range", "fairness_range", "cost_range", "feasibility_rate"):
            assert key in metrics

    def test_empty_metrics(self):
        assert FailureAnalyzer.compute_resilience_metrics([]) == {}


class TestStressSuite:
    def test_reproducible_with_seed(self):
        net = create_sample_network()
        a = create_stress_test_suite(net, n_communities=4, seed=123)
        b = create_stress_test_suite(net, n_communities=4, seed=123)
        assert [s["failures"] for s in a] == [s["failures"] for s in b]

    def test_different_seeds_differ(self):
        net = create_sample_network()
        a = create_stress_test_suite(net, n_communities=6, seed=1)
        b = create_stress_test_suite(net, n_communities=6, seed=2)
        assert [s["failures"] for s in a] != [s["failures"] for s in b]

    def test_suite_nonempty(self):
        net = create_sample_network()
        suite = create_stress_test_suite(net, n_communities=3, seed=0)
        assert len(suite) == 3
        assert all(s["scenario_type"] == "pipeline_failure" for s in suite)