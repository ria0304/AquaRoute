from __future__ import annotations

import pytest

from aquaroute.algorithms.optimizer import (
    AllocationOptimizer,
    compute_delivery_cost,
    compute_fairness,
    compute_shortage,
    effective_capacities,
    max_total_deliverable,
    validate_solution,
)
from aquaroute.models.water_network import WaterNetwork


def _solver_available() -> bool:
    net = WaterNetwork()
    net.add_reservoir(0, 10.0, 10.0)
    net.add_community(1, 5.0, 1)
    net.add_pipeline(0, 1, 10.0, 1.0)
    opt = AllocationOptimizer(net)
    return opt.solve(allow_fallback=False) is not None


HAS_SOLVER = _solver_available()
requires_solver = pytest.mark.skipif(not HAS_SOLVER,
                                     reason="No LP solver backend available")


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_shortage_boundaries(self):
        assert compute_shortage({1: 50.0, 2: 30.0}, {1: 50.0, 2: 30.0}) == pytest.approx(0.0)
        assert compute_shortage({1: 10.0}, {1: 50.0}) == pytest.approx(40.0)
        # Over-allocation still yields zero shortage.
        assert compute_shortage({1: 80.0}, {1: 50.0}) == pytest.approx(0.0)

    def test_fairness_boundaries(self):
        assert compute_fairness({1: 50.0, 2: 30.0}, {1: 50.0, 2: 30.0}) == pytest.approx(0.0)
        assert compute_fairness({1: 50.0}, {1: 50.0}) == 0.0
        assert compute_fairness({}, {}) == 0.0
        # Unequal proportional service is penalised.
        f = compute_fairness({1: 50.0, 2: 0.0}, {1: 50.0, 2: 30.0})
        assert f > 0.0

    def test_delivery_cost_no_double_count(self):
        net = WaterNetwork()
        net.add_reservoir(0, capacity=100.0, supply=100.0)
        net.add_reservoir(1, capacity=100.0, supply=100.0)
        net.add_community(2, demand=50.0, priority=1)
        net.add_pipeline(0, 2, 60.0, delivery_cost=1.0)
        net.add_pipeline(1, 2, 60.0, delivery_cost=5.0)
        # Without explicit flows, the cheapest incoming route is a lower bound;
        # the same volume must not be counted once per incoming pipe.
        assert compute_delivery_cost({2: 10.0}, net) == pytest.approx(10.0)

    def test_delivery_cost_with_flows(self):
        net = WaterNetwork()
        net.add_reservoir(0, capacity=100.0, supply=100.0)
        net.add_community(1, demand=50.0, priority=1)
        net.add_pipeline(0, 1, 50.0, delivery_cost=2.0)
        flows = {(0, 1): 40.0}
        assert compute_delivery_cost({1: 40.0}, net, path_flows=flows) == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Network-flow formulation
# ---------------------------------------------------------------------------
class TestNetworkFlow:
    def test_effective_capacity_direct(self):
        net = WaterNetwork()
        net.add_reservoir(0, capacity=100.0, supply=100.0)
        net.add_community(1, demand=50.0, priority=1)
        net.add_pipeline(0, 1, 30.0, delivery_cost=1.0)
        assert effective_capacities(net)[1] == pytest.approx(30.0)

    def test_effective_capacity_via_transshipment(self):
        # Community 2 can only be served through community 1.
        net = WaterNetwork()
        net.add_reservoir(0, capacity=100.0, supply=100.0)
        net.add_community(1, demand=10.0, priority=1)
        net.add_community(2, demand=20.0, priority=1)
        net.add_pipeline(0, 1, 40.0, delivery_cost=1.0)
        net.add_pipeline(1, 2, 25.0, delivery_cost=2.0)
        assert effective_capacities(net)[2] == pytest.approx(25.0)

    def test_global_budget_shares_pipelines(self):
        """Max total deliverable must not double-count a shared pipe."""
        net = WaterNetwork()
        net.add_reservoir(0, capacity=100.0, supply=100.0)
        net.add_community(1, demand=50.0, priority=1)
        net.add_community(2, demand=50.0, priority=1)
        net.add_pipeline(0, 1, 60.0, delivery_cost=1.0)
        net.add_pipeline(0, 2, 60.0, delivery_cost=1.0)
        # Global budget is bounded by the shared reservoir supply.
        assert max_total_deliverable(net) == pytest.approx(100.0)
        # ...whereas the sum of individual capacities would over-report.
        caps = effective_capacities(net)
        assert caps[1] + caps[2] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Optimizer correctness
# ---------------------------------------------------------------------------
class TestOptimizer:
    def test_build_model_structure(self):
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        net.add_community(1, 50.0, 1)
        net.add_pipeline(0, 1, 60.0, 1.0)
        model = AllocationOptimizer(net).build_model()
        for attr in ("C", "R", "E", "x", "f", "node_balance",
                     "demand_limit", "pipeline_capacity", "objective"):
            assert hasattr(model, attr), attr

    @requires_solver
    def test_full_demand_satisfied(self):
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        net.add_community(1, 50.0, 1)
        net.add_pipeline(0, 1, 60.0, 1.0)
        opt = AllocationOptimizer(net)
        opt.solve()
        res = opt.get_results()
        assert res["shortage"] == pytest.approx(0.0, abs=1e-6)
        assert res["allocations"][1] == pytest.approx(50.0)

    @requires_solver
    def test_supply_binding(self):
        net = WaterNetwork()
        net.add_reservoir(0, 40.0, 40.0)
        net.add_community(1, 100.0, 1)
        net.add_pipeline(0, 1, 100.0, 1.0)
        opt = AllocationOptimizer(net)
        opt.solve()
        res = opt.get_results()
        assert res["shortage"] == pytest.approx(60.0)
        assert res["allocations"][1] == pytest.approx(40.0)

    @requires_solver
    def test_pipeline_capacity_binding(self):
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        net.add_community(1, 50.0, 1)
        net.add_pipeline(0, 1, 30.0, 1.0)
        opt = AllocationOptimizer(net)
        opt.solve()
        res = opt.get_results()
        assert res["allocations"][1] == pytest.approx(30.0)
        assert res["shortage"] == pytest.approx(20.0)

    @requires_solver
    def test_transshipment_serves_both(self):
        """Water must route through community 1 to reach community 2."""
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        net.add_community(1, 10.0, 1)
        net.add_community(2, 20.0, 1)
        net.add_pipeline(0, 1, 40.0, delivery_cost=1.0)
        net.add_pipeline(1, 2, 25.0, delivery_cost=1.0)
        opt = AllocationOptimizer(net, strategy="shortage_min")
        opt.solve()
        res = opt.get_results()
        assert res["shortage"] == pytest.approx(0.0, abs=1e-6)
        assert res["allocations"][1] == pytest.approx(10.0)
        assert res["allocations"][2] == pytest.approx(20.0)

    @requires_solver
    def test_min_service_honoured_when_feasible(self):
        net = WaterNetwork()
        net.add_reservoir(0, 30.0, 30.0)
        net.add_community(1, 50.0, 1)
        net.add_community(2, 50.0, 2)
        net.add_pipeline(0, 1, 30.0, 1.0)
        net.add_pipeline(0, 2, 30.0, 1.0)
        opt = AllocationOptimizer(net, strategy="weighted",
                                  min_service_priority=[1], min_allocation=20.0)
        opt.solve()
        res = opt.get_results()
        assert res["allocations"][1] >= 20.0 - 1e-6
        assert res["validation_passed"] is True

    @requires_solver
    def test_min_service_impossible_reports_warning(self):
        net = WaterNetwork()
        net.add_reservoir(0, 5.0, 5.0)
        net.add_community(1, 50.0, 1)
        net.add_pipeline(0, 1, 60.0, 1.0)
        opt = AllocationOptimizer(net, strategy="weighted",
                                  min_service_priority=[1], min_allocation=20.0)
        opt.solve()
        res = opt.get_results()
        assert res["validation_passed"] is True
        assert any("C5" in w for w in res["validation_warnings"])

    @requires_solver
    def test_strategies_produce_different_allocations(self):
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        net.add_reservoir(1, 80.0, 40.0)  # tight supply
        for c, d in ((2, 20.0), (3, 15.0), (4, 25.0), (5, 15.0), (6, 10.0)):
            net.add_community(c, d, 1)
        net.add_pipeline(0, 2, 30.0, 1.0)
        net.add_pipeline(0, 3, 20.0, 1.5)
        net.add_pipeline(0, 4, 25.0, 1.2)
        net.add_pipeline(1, 5, 20.0, 1.3)
        net.add_pipeline(1, 6, 15.0, 1.1)
        net.add_pipeline(2, 5, 15.0, 2.0)
        net.add_pipeline(3, 6, 15.0, 2.0)
        net.add_pipeline(4, 5, 15.0, 1.8)
        net.add_pipeline(4, 6, 15.0, 1.9)

        sigs = {}
        for strategy in ("shortage_min", "uniform", "maxmin", "proportional", "weighted"):
            opt = AllocationOptimizer(net, strategy=strategy)
            opt.solve()
            sigs[strategy] = sorted(opt.get_results()["allocations"].items())
        distinct = {tuple(v) for v in sigs.values()}
        assert len(distinct) >= 3

    @requires_solver
    def test_weights_affect_fairness_cost_tradeoff(self):
        from aquaroute.models.water_network import create_sample_network
        net = create_sample_network()
        for r in net.get_reservoirs():
            net.G.nodes[r]["supply"] *= 0.3  # induce shortage so a tradeoff exists

        fair_first = AllocationOptimizer(net, strategy="weighted", beta=1.0, gamma=0.0)
        cost_first = AllocationOptimizer(net, strategy="weighted", beta=0.0, gamma=1.0)
        fair_first.solve()
        cost_first.solve()
        r_fair, r_cost = fair_first.get_results(), cost_first.get_results()

        # Maximising fairness yields lower disparity ...
        assert r_fair["fairness"] < r_cost["fairness"] - 1e-6
        # ... while prioritising cost yields a cheaper routing.
        assert r_cost["delivery_cost"] < r_fair["delivery_cost"] - 1e-6
        # Both achieve the same (minimum feasible) shortage.
        assert r_fair["shortage"] == pytest.approx(r_cost["shortage"], abs=1e-4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_rejects_supply_excess(self):
        net = WaterNetwork()
        net.add_reservoir(0, 10.0, 10.0)
        net.add_community(1, 50.0, 1)
        net.add_pipeline(0, 1, 60.0, 1.0)
        ok, violations = validate_solution(
            {1: 40.0}, net, {1: 50.0}, total_supply=10.0,
            effective_caps={1: 60.0})
        assert ok is False
        assert any("C1" in v for v in violations)

    def test_accepts_valid(self):
        net = WaterNetwork()
        net.add_reservoir(0, 10.0, 10.0)
        net.add_community(1, 50.0, 1)
        net.add_pipeline(0, 1, 60.0, 1.0)
        ok, violations = validate_solution(
            {1: 10.0}, net, {1: 50.0}, total_supply=10.0,
            effective_caps={1: 60.0})
        assert ok is True
        assert violations == []