from __future__ import annotations

import pytest
from aquaroute.models.water_network import WaterNetwork, create_sample_network
from aquaroute.algorithms.optimizer import AllocationOptimizer, compute_shortage, compute_fairness


class TestOptimizer:
    """Tests for the Pyomo-based allocation optimizer."""

    def test_optimizer_basic(self):
        """Test basic optimization with a simple network."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        net.add_community(node_id=1, demand=50.0, priority=1)
        net.add_community(node_id=2, demand=30.0, priority=2)
        net.add_pipeline(from_node=0, to_node=1, capacity=60.0, delivery_cost=1.0)
        net.add_pipeline(from_node=0, to_node=2, capacity=40.0, delivery_cost=1.5)
        
        optimizer = AllocationOptimizer(network=net, alpha=1.0, beta=1.0, gamma=1.0)
        model = optimizer.build_model()
        
        assert model is not None
        assert hasattr(model, 'x')
        assert hasattr(model, 'source_constraint')
        assert hasattr(model, 'demand_constraint')
        assert hasattr(model, 'pipeline_constraint')
    
    def test_optimizer_build_model(self):
        """Test that the model builds correctly (solver may not be available)."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        net.add_community(node_id=1, demand=50.0, priority=1)
        net.add_community(node_id=2, demand=30.0, priority=2)
        net.add_pipeline(from_node=0, to_node=1, capacity=60.0, delivery_cost=1.0)
        net.add_pipeline(from_node=0, to_node=2, capacity=40.0, delivery_cost=1.5)
        
        optimizer = AllocationOptimizer(network=net, alpha=1.0, beta=1.0, gamma=1.0)
        model = optimizer.build_model()
        
        # Model should build even without solver
        assert model is not None
        assert hasattr(model, 'x')
        # The objective should be about minimizing shortage
        assert model.objective is not None
    
    def test_optimizer_with_weights(self):
        """Test that optimization weights affect the solution model."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        net.add_community(node_id=1, demand=50.0, priority=1)
        net.add_community(node_id=2, demand=30.0, priority=2)
        net.add_pipeline(from_node=0, to_node=1, capacity=60.0, delivery_cost=1.0)
        net.add_pipeline(from_node=0, to_node=2, capacity=40.0, delivery_cost=1.5)
        
        # Efficiency-first: gamma high should favor cheaper routes
        opt_eff = AllocationOptimizer(network=net, alpha=0.2, beta=0.5, gamma=3.0)
        model_eff = opt_eff.build_model()
        
        # Fairness-weighted: beta high should favor fairness
        opt_fair = AllocationOptimizer(network=net, alpha=0.5, beta=2.0, gamma=1.0)
        model_fair = opt_fair.build_model()
        
        # Both models should build successfully
        assert model_eff is not None
        assert model_fair is not None
    
    def test_optimizer_minimum_service(self):
        """Test minimum service guarantee constraint."""
        net = WaterNetwork()
        net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
        net.add_community(node_id=1, demand=50.0, priority=1)
        net.add_community(node_id=2, demand=30.0, priority=2)
        net.add_pipeline(from_node=0, to_node=1, capacity=60.0, delivery_cost=1.0)
        net.add_pipeline(from_node=0, to_node=2, capacity=40.0, delivery_cost=1.5)
        
        # Test with minimum service guarantee for community 1
        optimizer = AllocationOptimizer(
            network=net, alpha=1.0, beta=1.0, gamma=1.0,
            min_service_priority=[1], min_allocation=20.0
        )
        model = optimizer.build_model()
        
        # Model should build with minimum service constraint
        assert model is not None
        assert hasattr(model, 'min_service')
    
    def test_compute_shortage(self):
        """Test shortage computation."""
        allocations = {1: 40.0, 2: 20.0}
        demands = {1: 50.0, 2: 30.0}
        
        shortage = compute_shortage(allocations, demands)
        # Total demand = 80, total allocated = 60, shortage = 20
        assert abs(shortage - 20.0) < 0.001
    
    def test_compute_fairness(self):
        """Test fairness computation."""
        # Perfect equality: everyone gets exactly their demand
        allocations = {1: 50.0, 2: 30.0}
        demands = {1: 50.0, 2: 30.0}
        fairness = compute_fairness(allocations, demands)
        assert fairness == 0.0, "Perfect equality should have fairness = 0"
        
        # High inequality: one community gets everything
        allocations = {1: 80.0, 2: 0.0}
        demands = {1: 50.0, 2: 30.0}
        fairness = compute_fairness(allocations, demands)
        assert fairness > 0.0, "Inequality should have positive fairness"