from __future__ import annotations

import json
import os

from aquaroute.algorithms.evaluation import EvaluationSuite
from aquaroute.algorithms.uncertainty import ScenarioGenerator
from aquaroute.models.water_network import create_sample_network


def _tight_network():
    """Sample network with reduced supply so strategies actually differ."""
    net = create_sample_network()
    for r in net.get_reservoirs():
        net.G.nodes[r]["supply"] *= 0.3
    return net


class TestEvaluationSuite:
    def test_default_approaches_are_distinct_strategies(self):
        suite = EvaluationSuite(create_sample_network())
        strategies = [kw.get("strategy") for _, kw in suite.approaches]
        assert len(set(strategies)) >= 4
        assert "weighted" in strategies

    def test_comparison_runs(self):
        net = _tight_network()
        suite = EvaluationSuite(net)
        results = suite.run_comparison(n_repeats=2, seed=3)
        assert set(results) == {name for name, _ in suite.approaches}
        for rows in results.values():
            assert len(rows) == len(suite.scenarios)
            assert all("results" in r for r in rows)

    def test_strategies_yield_different_fairness(self):
        net = _tight_network()
        suite = EvaluationSuite(
            net,
            scenarios=[ScenarioGenerator.supply_reduction_scenario(net, 0.3)],
        )
        results = suite.run_comparison()
        fairness = {
            name: rows[0]["results"].get("fairness")
            for name, rows in results.items()
        }
        # At least three different fairness values across strategies.
        assert len({round(v, 6) for v in fairness.values()}) >= 3

    def test_weighted_weights_change_outcome(self):
        net = _tight_network()
        scenario = [ScenarioGenerator.supply_reduction_scenario(net, 0.3)]
        fair_suite = EvaluationSuite(
            net,
            approaches=[("fair", {"strategy": "weighted", "beta": 1.0, "gamma": 0.0})],
            scenarios=scenario)
        cost_suite = EvaluationSuite(
            net,
            approaches=[("cost", {"strategy": "weighted", "beta": 0.0, "gamma": 1.0})],
            scenarios=scenario)
        rf = fair_suite.run_comparison()["fair"][0]["results"]
        rc = cost_suite.run_comparison()["cost"][0]["results"]
        # The fairness-weighted approach must be at least as fair.
        assert rf["fairness"] <= rc["fairness"] + 1e-6

    def test_summary_keys(self):
        net = _tight_network()
        suite = EvaluationSuite(net)
        suite.run_comparison()
        summary = suite.compute_summary_metrics()
        assert summary
        for metrics in summary.values():
            for key in ("avg_shortage", "avg_fairness", "avg_cost",
                        "feasibility_rate", "scenario_count"):
                assert key in metrics

    def test_reproducible_with_seed(self):
        net = _tight_network()
        a = EvaluationSuite(net, seed=11).run_comparison(n_repeats=2, seed=11)
        b = EvaluationSuite(net, seed=11).run_comparison(n_repeats=2, seed=11)
        for name in a:
            av = [r["results"].get("fairness") for r in a[name]]
            bv = [r["results"].get("fairness") for r in b[name]]
            assert av == bv

    def test_statistical_significance(self):
        net = _tight_network()
        suite = EvaluationSuite(net)
        suite.run_comparison()
        names = list(suite.results)
        sig = suite.statistical_significance("fairness", names[0], names[-1])
        assert sig is None or "p_value" in sig

    def test_export_csv_and_json(self, tmp_path):
        net = _tight_network()
        suite = EvaluationSuite(net)
        results = suite.run_comparison()
        csv_path = os.path.join(tmp_path, "results.csv")
        json_path = os.path.join(tmp_path, "results.json")
        suite.export_results_csv(results, csv_path)
        suite.export_results_json(results, json_path)
        assert os.path.getsize(csv_path) > 0
        with open(json_path) as fh:
            data = json.load(fh)
        assert isinstance(data, list) and data
        assert "approach" in data[0]