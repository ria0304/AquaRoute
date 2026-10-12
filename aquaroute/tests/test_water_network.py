from __future__ import annotations

import os

import pytest

from aquaroute.models.water_network import WaterNetwork, create_sample_network

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")


class TestWaterNetwork:
    def test_basic_initialization(self):
        net = WaterNetwork()
        assert net.G is not None
        assert net.total_demand() == 0.0
        assert net.total_supply() == 0.0

    def test_add_reservoir(self):
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        assert 0 in net.get_reservoirs()
        assert net.total_supply() == 100.0

    def test_add_community(self):
        net = WaterNetwork()
        net.add_community(0, 20.0, 1)
        assert 0 in net.get_communities()
        assert net.total_demand() == 20.0

    def test_add_pipeline_metadata(self):
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        net.add_community(1, 20.0, 1)
        net.add_pipeline(0, 1, 30.0, delivery_cost=1.5)
        assert net.G.has_edge(0, 1)
        assert net.G[0][1]["capacity"] == 30.0
        assert net.G[0][1]["delivery_cost"] == 1.5

    def test_check_feasibility(self):
        net = WaterNetwork()
        net.add_reservoir(0, 100.0, 100.0)
        net.add_community(1, 50.0, 1)
        net.add_pipeline(0, 1, 60.0, 1.0)
        assert net.check_feasibility({1: 50.0}) is True
        assert net.check_feasibility({1: 30.0}) is False


class TestSampleNetwork:
    def test_counts(self):
        net = create_sample_network()
        assert net.n_communities == 5
        assert net.n_reservoirs == 2
        assert len(net.get_communities()) == 5
        assert len(net.get_reservoirs()) == 2
        assert net.total_demand() > 0

    def test_positive_demands_and_supplies(self):
        net = create_sample_network()
        assert all(net.G.nodes[c]["demand"] > 0 for c in net.get_communities())
        assert all(net.G.nodes[r]["supply"] > 0 for r in net.get_reservoirs())


class TestJsonLoading:
    def test_from_dict_roundtrip(self):
        original = create_sample_network()
        data = {
            "reservoirs": [{"node_id": r, "capacity": original.G.nodes[r]["capacity"],
                            "supply": original.G.nodes[r]["supply"]}
                           for r in original.get_reservoirs()],
            "communities": [{"node_id": c, "demand": original.G.nodes[c]["demand"],
                             "priority": original.G.nodes[c]["priority"]}
                            for c in original.get_communities()],
            "pipelines": [{"from_node": u, "to_node": v,
                           "capacity": original.G[u][v]["capacity"],
                           "delivery_cost": original.G[u][v]["delivery_cost"]}
                          for u, v in original.G.edges()],
        }
        restored = WaterNetwork.from_dict(data)
        assert restored.total_demand() == pytest.approx(original.total_demand())
        assert restored.total_supply() == pytest.approx(original.total_supply())
        assert set(restored.get_communities()) == set(original.get_communities())

    def test_from_json_file(self):
        path = os.path.join(DATA_DIR, "sample_network.json")
        if not os.path.exists(path):
            pytest.skip("sample_network.json not present")
        net = WaterNetwork.from_json(path)
        assert len(net.get_reservoirs()) == 2
        assert len(net.get_communities()) == 5
        assert net.total_demand() == pytest.approx(85.0)