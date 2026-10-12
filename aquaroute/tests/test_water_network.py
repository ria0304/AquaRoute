from __future__ import annotations

import pytest
from aquaroute.models.water_network import WaterNetwork, create_sample_network


class TestWaterNetwork:
    """Tests for the water network model."""

    def test_basic_initialization(self):
        """Test basic network creation."""
        net = WaterNetwork()
        assert net.G is not None
        assert net.total_demand() == 0.0
        assert net.total_supply() == 0.0

    def test_add_reservoir(self):
        """Test adding reservoirs."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        assert 0 in net.get_reservoirs()
        assert net.total_supply() == 100.0

    def test_add_community(self):
        """Test adding communities."""
        net = WaterNetwork()
        net.add_community(node_id=0, demand=20.0, priority=1)
        assert 0 in net.get_communities()
        assert net.total_demand() == 20.0

    def test_add_pipeline(self):
        """Test adding pipelines."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        net.add_community(node_id=1, demand=20.0, priority=1)
        net.add_pipeline(from_node=0, to_node=1, capacity=30.0, delivery_cost=1.5)
        
        assert net.G.has_edge(0, 1)
        assert net.G[0][1]['capacity'] == 30.0
        assert net.G[0][1]['delivery_cost'] == 1.5

    def test_check_feasibility_full(self):
        """Test feasibility check with full allocation."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        net.add_community(node_id=1, demand=50.0, priority=1)
        net.add_pipeline(from_node=0, to_node=1, capacity=60.0, delivery_cost=1.0)
        
        # Full allocation should be feasible
        assert net.check_feasibility({1: 50.0}) == True

    def test_check_feasibility_partial(self):
        """Test feasibility check with partial allocation."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        net.add_community(node_id=1, demand=50.0, priority=1)
        net.add_pipeline(from_node=0, to_node=1, capacity=60.0, delivery_cost=1.0)
        
        # Partial allocation should be infeasible (demand not met)
        assert net.check_feasibility({1: 30.0}) == False

    def test_sample_network(self):
        """Test the sample network creation."""
        net = create_sample_network()
        assert net.n_communities == 5
        assert net.n_reservoirs == 2
        assert net.total_demand() > 0
        assert len(net.get_communities()) == 5
        assert len(net.get_reservoirs()) == 2


class TestSampleNetwork:
    """Tests for sample network fixture."""

    def test_community_demands(self):
        """Test that communities have proper demand values."""
        net = create_sample_network()
        communities = net.get_communities()
        for c in communities:
            demand = net.G.nodes[c]['demand']
            assert demand > 0

    def test_reservoir_supplies(self):
        """Test that reservoirs have proper supply values."""
        net = create_sample_network()
        reservoirs = net.get_reservoirs()
        for r in reservoirs:
            supply = net.G.nodes[r].get('supply', float('inf'))
            assert supply > 0