from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import networkx as nx


class WaterNetwork:
    """Graph-based model of a water distribution network.
    
    Nodes: reservoirs (supply), communities (demand), junctions
    Edges: pipelines with capacity and delivery cost
    """
    
    def __init__(self, n_reservoirs: int = 0, n_communities: int = 0,
                 pipelines: Optional[List[Tuple[int, int, Dict]]] = None):
        self.G = nx.DiGraph()
        self.n_reservoirs = n_reservoirs
        self.n_communities = n_communities
        
        if pipelines:
            for u, v, data in pipelines:
                self.G.add_edge(u, v, **data)
    
    def add_reservoir(self, node_id: int, capacity: float, 
                      supply: float = float('inf')) -> None:
        """Add a reservoir node with supply capacity."""
        self.G.add_node(node_id, type='reservoir', capacity=capacity, supply=supply)
        self.n_reservoirs += 1
    
    def add_community(self, node_id: int, demand: float, 
                      priority: int = 0) -> None:
        """Add a community node with water demand and priority level."""
        self.G.add_node(node_id, type='community', demand=demand, priority=priority)
        self.n_communities += 1
    
    def add_pipeline(self, from_node: int, to_node: int, 
                     capacity: float, delivery_cost: float = 1.0) -> None:
        """Add a pipeline with capacity and per-unit delivery cost."""
        self.G.add_edge(from_node, to_node,
                       capacity=capacity, 
                       delivery_cost=delivery_cost,
                       flow=0.0)
    
    def get_communities(self) -> List[int]:
        """Return list of community node IDs."""
        return [n for n, d in self.G.nodes(data=True) if d.get('type') == 'community']
    
    def get_reservoirs(self) -> List[int]:
        """Return list of reservoir node IDs."""
        return [n for n, d in self.G.nodes(data=True) if d.get('type') == 'reservoir']
    
    def get_pipeline_capacity(self, edge: Tuple[int, int]) -> float:
        """Get capacity of a pipeline edge."""
        if self.G.has_edge(*edge):
            return self.G[edge[0]][edge[1]]['capacity']
        return 0.0
    
    def get_pipeline_cost(self, edge: Tuple[int, int]) -> float:
        """Get delivery cost of a pipeline edge."""
        if self.G.has_edge(*edge):
            return self.G[edge[0]][edge[1]]['delivery_cost']
        return 0.0
    
    def total_demand(self) -> float:
        """Total water demand across all communities."""
        return sum(d.get('demand', 0) for n, d in self.G.nodes(data=True) if d.get('type') == 'community')
    
    def total_supply(self) -> float:
        """Total water supply from all reservoirs."""
        return sum(d.get('supply', 0) for n, d in self.G.nodes(data=True) if d.get('type') == 'reservoir')
    
    def check_feasibility(self, allocations: Dict[int, float]) -> bool:
        """Check if allocated water satisfies community demands within capacity constraints."""
        for node, alloc in allocations.items():
            if node in self.get_communities():
                demand = self.G.nodes[node]['demand']
                if alloc < demand:
                    return False  # unsatisfied demand
        return True


def create_sample_network() -> WaterNetwork:
    """Create a sample water distribution network for demonstration.
    
    Network: 2 reservoirs -> 5 communities via pipelines
    """
    net = WaterNetwork()
    
    # Add reservoirs
    net.add_reservoir(node_id=0, capacity=100.0, supply=100.0)
    net.add_reservoir(node_id=1, capacity=80.0, supply=80.0)
    
    # Add communities with different demands and priorities
    net.add_community(node_id=2, demand=20.0, priority=1)  # High priority
    net.add_community(node_id=3, demand=15.0, priority=2)  # Medium priority
    net.add_community(node_id=4, demand=25.0, priority=1)  # High priority
    net.add_community(node_id=5, demand=15.0, priority=3)  # Low priority
    net.add_community(node_id=6, demand=10.0, priority=2)  # Medium priority
    
    # Add pipelines with capacities and delivery costs
    # Reservoir 0 -> Communities
    net.add_pipeline(from_node=0, to_node=2, capacity=30.0, delivery_cost=1.0)  # to community 2
    net.add_pipeline(from_node=0, to_node=3, capacity=20.0, delivery_cost=1.5)  # to community 3
    net.add_pipeline(from_node=0, to_node=4, capacity=25.0, delivery_cost=1.2)  # to community 4
    
    # Reservoir 1 -> Communities
    net.add_pipeline(from_node=1, to_node=5, capacity=20.0, delivery_cost=1.3)  # to community 5
    net.add_pipeline(from_node=1, to_node=6, capacity=15.0, delivery_cost=1.1)  # to community 6
    
    # Cross-network pipelines (inter-community routing)
    net.add_pipeline(from_node=2, to_node=5, capacity=15.0, delivery_cost=2.0)
    net.add_pipeline(from_node=3, to_node=6, capacity=15.0, delivery_cost=2.0)
    net.add_pipeline(from_node=4, to_node=5, capacity=15.0, delivery_cost=1.8)
    net.add_pipeline(from_node=4, to_node=6, capacity=15.0, delivery_cost=1.9)
    
    return net