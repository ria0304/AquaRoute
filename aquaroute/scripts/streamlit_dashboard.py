from __future__ import annotations

import copy
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

from aquaroute.models.water_network import WaterNetwork, create_sample_network
from aquaroute.algorithms.optimizer import AllocationOptimizer
from aquaroute.algorithms.uncertainty import ScenarioGenerator, FailureAnalyzer
from aquaroute.algorithms.evaluation import EvaluationSuite, compare_approaches


st.set_page_config(
    page_title="AquaRoute — Algorithmic Optimization for Water Distribution",
    page_icon="💧",
    layout="wide",
)

st.title("💧 AquaRoute: Algorithmic Optimization for Reliable Community Water Distribution")
st.caption("Research prototype for fair and reliable community water allocation under uncertainty")


def sidebar_controls(network: WaterNetwork) -> dict:
    """Create sidebar controls for scenario configuration."""
    st.sidebar.header("Scenario Configuration")
    
    scenario_type = st.sidebar.selectbox(
        "Scenario Type",
        ["Normal", "Demand Spike", "Supply Reduction", "Pipeline Failure", "Multiple Disruptions"],
        help="Select the water distribution scenario to evaluate"
    )
    
    st.sidebar.header("Parameters")
    
    min_service = st.sidebar.checkbox(
        "Enable Minimum Service Guarantees",
        value=False,
        help="Priority communities receive at least min_allocation before optimization"
    )
    
    min_allocation = 0.0
    if min_service:
        min_allocation = st.sidebar.slider(
            "Minimum Allocation per Community (units)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=5.0,
            help="Minimum water units guaranteed for communities with this flag"
        )
    
    alpha = st.sidebar.slider("Alpha (shortage weight)", 0.0, 3.0, 1.0, 0.1)
    beta = st.sidebar.slider("Beta (fairness weight)", 0.0, 3.0, 1.0, 0.1)
    gamma = st.sidebar.slider("Gamma (delivery cost weight)", 0.0, 3.0, 1.0, 0.1)
    
    st.sidebar.header("Disruption Parameters")
    spike_factor = 1.0
    reduction_factor = 1.0
    n_failed_pipelines = 0
    
    if scenario_type == "Demand Spike":
        spike_factor = st.sidebar.slider("Spike Factor", 1.0, 3.0, 1.5, 0.1)
    elif scenario_type == "Supply Reduction":
        reduction_factor = st.sidebar.slider("Reduction Factor", 0.1, 1.0, 0.5, 0.1)
    elif scenario_type == "Pipeline Failure":
        n_failed_pipelines = st.sidebar.slider(
            "Number of Failed Pipelines",
            0, 6, 2,
            help="Select pipeline links to disable"
        )
    elif scenario_type == "Multiple Disruptions":
        spike_factor = st.sidebar.slider("Demand Spike Factor", 1.0, 2.0, 1.3, 0.1)
        reduction_factor = st.sidebar.slider("Supply Reduction Factor", 0.3, 1.0, 0.6, 0.1)
        n_failed_pipelines = st.sidebar.slider(
            "Number of Failed Pipelines",
            0, 6, 2,
            help="Select pipeline links to disable"
        )
    
    return {
        'scenario_type': scenario_type,
        'min_service': min_service,
        'min_allocation': min_allocation,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'spike_factor': spike_factor,
        'reduction_factor': reduction_factor,
        'n_failed_pipelines': n_failed_pipelines,
    }


def run_optimization(network, params):
    """Run the allocation optimization and return results."""
    with st.spinner("Solving allocation optimization..."):
        optimizer = AllocationOptimizer(
            network=network,
            alpha=params['alpha'],
            beta=params['beta'],
            gamma=params['gamma'],
            min_service_priority=list(range(len(network.get_communities()))) if params['min_service'] else [],
            min_allocation=params['min_allocation'],
        )
        
        import time
        start = time.time()
        solution = optimizer.solve()
        runtime = time.time() - start
        
        if solution is None:
            st.error("Optimization failed - no feasible solution found")
            return None
        
        results = optimizer.get_results()
        results['runtime'] = runtime
        results['solution'] = solution
        
        return results


def main():
    """Main dashboard application."""
    # Initialize network
    if 'network' not in st.session_state:
        st.session_state.network = create_sample_network()
    if 'params' not in st.session_state:
        st.session_state.params = sidebar_controls(st.session_state.network)
    
    original_network = st.session_state.network
    params = st.session_state.params
    
    # Work on a copy to avoid mutating the original network
    network = copy.deepcopy(original_network)
    
    # Update network based on scenario parameters
    demands = {}
    for c in network.get_communities():
        base_demand = network.G.nodes[c]['demand']
        if params['scenario_type'] == "Demand Spike":
            demands[c] = base_demand * params['spike_factor']
        elif params['scenario_type'] == "Supply Reduction":
            demands[c] = base_demand  # keep baseline, supply changes
        else:
            demands[c] = base_demand
        network.G.nodes[c]['demand'] = demands[c]
    
    supply = {}
    for r in network.get_reservoirs():
        base_supply = network.G.nodes[r].get('supply', float('inf'))
        if params['scenario_type'] == "Supply Reduction":
            supply[r] = base_supply * params['reduction_factor']
        else:
            supply[r] = base_supply
        network.G.nodes[r]['supply'] = supply[r]
    
    # Pipeline failures
    failed_edges = []
    if params['scenario_type'] == "Pipeline Failure":
        edges = list(network.G.edges())
        np_random = __import__('numpy').random
        np_random.seed(42)
        n_to_fail = min(params['n_failed_pipelines'], len(edges))
        failed_indices = np_random.choice(len(edges), size=n_to_fail, replace=False)
        failed_edges = [edges[i] for i in failed_indices]
        for u, v in failed_edges:
            network.G.remove_edge(u, v)
    elif params['scenario_type'] == "Multiple Disruptions":
        edges = list(network.G.edges())
        np_random = __import__('numpy').random
        np_random.seed(42)
        n_to_fail = min(params['n_failed_pipelines'], len(edges))
        failed_indices = np_random.choice(len(edges), size=n_to_fail, replace=False)
        failed_edges = [edges[i] for i in failed_indices]
        for u, v in failed_edges:
            network.G.remove_edge(u, v)
    
    # Run optimization
    results = run_optimization(network, params)
    
    # Display header info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Demand", f"{network.total_demand():.1f} units")
    with col2:
        st.metric("Total Supply", f"{network.total_supply():.1f} units")
    with col3:
        if results:
            st.metric("Shortage", f"{results['shortage']:.1f} units")
        else:
            st.metric("Shortage", "N/A")
    with col4:
        if results:
            st.metric("Fairness", f"{results['fairness']:.3f}")
        else:
            st.metric("Fairness", "N/A")
    
    # Allocation results
    st.divider()
    st.header("📊 Allocation Results")
    
    if results:
        communities = network.get_communities()
        alloc_data = []
        for c in communities:
            alloc = results['allocations'].get(c, 0.0)
            demand = network.G.nodes[c]['demand']
            shortfall = max(0, demand - alloc)
            pct_met = (alloc / demand * 100) if demand > 0 else 100.0
            alloc_data.append({
                'Community': c,
                'Allocated': alloc,
                'Demand': demand,
                'Shortfall': shortfall,
                '% Met': f"{pct_met:.1f}%",
                'Priority': network.G.nodes[c].get('priority', 0),
            })
        
        df = pd.DataFrame(alloc_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization: allocated vs demand
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Allocated",
            x=[f"Comm {c}" for c in communities],
            y=[results['allocations'].get(c, 0) for c in communities],
            marker_color='royalblue',
        ))
        fig.add_trace(go.Bar(
            name="Demand",
            x=[f"Comm {c}" for c in communities],
            y=[network.G.nodes[c]['demand'] for c in communities],
            marker_color='lightblue',
            opacity=0.6,
        ))
        fig.update_layout(
            title="Water Allocation vs Demand by Community",
            barmode='group',
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Total Allocated:** {results['total_allocated']:.1f} units")
            st.info(f"**Total Shortage:** {results['shortage']:.1f} units")
            st.info(f"**Feasible:** {'Yes' if results['shortage'] == 0 else 'No'}")
        with col2:
            st.info(f"**Fairness Index:** {results['fairness']:.3f}")
            st.info(f"**Delivery Cost:** {results['delivery_cost']:.1f}")
        with col3:
            st.info(f"**Runtime:** {results['runtime']:.2f}s")
            st.info(f"**Objective Value:** {results.get('objective_value', 'N/A')}")
            st.info(f"**Status:** {results.get('status', 'N/A')}")
    
    # Scenario comparison
    st.divider()
    st.header("🔄 Scenario Comparison")
    
    if st.button("Run Multi-Scenario Evaluation"):
        with st.spinner("Evaluating across scenarios..."):
            scenarios = [
                ScenarioGenerator.normal_scenario(network),
                ScenarioGenerator.demand_spike_scenario(network, params['spike_factor']),
                ScenarioGenerator.supply_reduction_scenario(network, params['reduction_factor']),
            ]
            if failed_edges:
                scenarios.append({
                    'demands': {c: network.G.nodes[c]['demand'] for c in network.get_communities()},
                    'supply': {r: network.G.nodes[r].get('supply', float('inf')) for r in network.get_reservoirs()},
                    'failures': failed_edges,
                    'scenario_type': 'pipeline_failure',
                })
            
            # Run comparison of default approaches
            approaches = [
                ("min_service_first", {
                    'alpha': params['alpha'], 'beta': params['beta'], 'gamma': params['gamma'],
                    'min_service_priority': [],
                    'min_allocation': params['min_allocation'],
                }),
                ("fairness_weighted", {
                    'alpha': 0.5, 'beta': 2.0, 'gamma': 1.0,
                    'min_service_priority': [],
                    'min_allocation': params['min_allocation'],
                }),
            ]
            
            suite = EvaluationSuite(network, approaches=approaches, scenarios=scenarios)
            suite.run_comparison()
            
            st.session_state.evaluation_results = suite
            st.session_state.scenarios = scenarios
    
    if 'evaluation_results' in st.session_state:
        st.subheader("Comparison: min_service_first vs fairness_weighted")
        summary = st.session_state.evaluation_results.compute_summary_metrics()
        
        for approach, metrics in summary.items():
            st.write(f"**{approach}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Shortage", f"{metrics['avg_shortage']:.1f}")
                st.metric("Feasibility Rate", f"{metrics['feasibility_rate']:.1%}")
            with col2:
                st.metric("Avg Fairness", f"{metrics['avg_fairness']:.3f}")
            with col3:
                st.metric("Avg Cost", f"{metrics['avg_cost']:.1f}")
                st.metric("Avg Runtime", f"{metrics['avg_runtime']:.2f}s")
        
        # Statistical test
        if len(summary) >= 2:
            approach_names = list(summary.keys())
            sig = st.session_state.evaluation_results.statistical_significance(
                metric='shortage', 
                approach_a=approach_names[0], 
                approach_b=approach_names[1]
            )
            if sig:
                st.caption(f"Statistical test: {sig['interpretation']} (p={sig['p_value']:.3f})")
    
    # Footer
    st.divider()
    st.caption("""
    **AquaRoute** — Research prototype for algorithmic water allocation optimization.
    Built with Python, NetworkX, Pyomo, and Streamlit.
    """)


if __name__ == "__main__":
    main()