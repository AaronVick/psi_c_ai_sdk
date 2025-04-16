#!/usr/bin/env python
"""
Schema Fork Detection Example
----------------------------

This example demonstrates how to use the SchemaForkDetector to monitor a ΨC agent
for potential identity forks during runtime.

The detector tracks schema embeddings over time and identifies when a schema mutation
has functionally forked the agent's identity, which is important for:
1. Understanding agent evolution patterns
2. Detecting unexpected identity shifts
3. Maintaining coherent agent behavior over long time periods
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# Add the parent directory to sys.path to import from the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the detector
from schema.fork_detector import SchemaForkDetector, analyze_agent_schema_fork


class SimpleAgent:
    """
    A simple agent with a schema graph that can be modified to demonstrate fork detection.
    """
    
    def __init__(self, num_nodes=30):
        """Initialize with a starter schema graph."""
        self.schema_graph = self._create_initial_schema(num_nodes)
        
    def _create_initial_schema(self, num_nodes):
        """Create an initial schema graph."""
        G = nx.Graph()
        
        # Add nodes with types
        node_types = ['belief', 'memory', 'concept', 'relation', 'rule']
        for i in range(num_nodes):
            node_type = node_types[i % len(node_types)]
            G.add_node(f"node_{i}", type=node_type, importance=np.random.uniform(0.3, 0.9))
        
        # Add edges with weights - creating a small-world network
        for i in range(num_nodes):
            # Connect to nearby nodes
            for j in range(1, 4):
                neighbor = (i + j) % num_nodes
                G.add_edge(f"node_{i}", f"node_{neighbor}", weight=np.random.uniform(0.6, 0.9))
            
            # Add a few random long-range connections
            if np.random.random() < 0.2:
                random_node = np.random.randint(0, num_nodes)
                if random_node != i and not G.has_edge(f"node_{i}", f"node_{random_node}"):
                    G.add_edge(f"node_{i}", f"node_{random_node}", 
                              weight=np.random.uniform(0.3, 0.7))
        
        return G
    
    def evolve_schema_gradually(self):
        """Evolve the schema with small, gradual changes."""
        # Add a new node
        node_id = f"node_{len(self.schema_graph)}"
        node_type = np.random.choice(['belief', 'memory', 'concept', 'relation', 'rule'])
        self.schema_graph.add_node(node_id, type=node_type, importance=np.random.uniform(0.3, 0.9))
        
        # Connect to existing nodes
        for _ in range(3):
            existing_node = np.random.choice(list(self.schema_graph.nodes()))
            if existing_node != node_id:
                self.schema_graph.add_edge(node_id, existing_node, weight=np.random.uniform(0.5, 0.9))
        
        # Slightly modify some existing edge weights
        for _ in range(5):
            if len(list(self.schema_graph.edges())) > 0:
                edge = np.random.choice(list(self.schema_graph.edges()))
                current_weight = self.schema_graph[edge[0]][edge[1]].get('weight', 0.5)
                new_weight = max(0.1, min(0.9, current_weight + np.random.uniform(-0.1, 0.1)))
                self.schema_graph[edge[0]][edge[1]]['weight'] = new_weight
    
    def cause_schema_fork(self):
        """Make dramatic changes to the schema that should trigger a fork detection."""
        # Strategy 1: Remove a significant number of nodes
        nodes = list(self.schema_graph.nodes())
        nodes_to_remove = np.random.choice(nodes, size=int(len(nodes) * 0.3), replace=False)
        for node in nodes_to_remove:
            self.schema_graph.remove_node(node)
            
        # Strategy 2: Add a large cluster of new nodes with a different structure
        num_new_nodes = int(len(self.schema_graph) * 0.4)
        base_id = len(self.schema_graph) + 1000  # Use an offset to avoid ID conflicts
        
        # Add the new nodes
        for i in range(num_new_nodes):
            node_id = f"fork_node_{base_id + i}"
            node_type = 'belief' if i % 2 == 0 else 'rule'  # Different distribution than original
            self.schema_graph.add_node(node_id, type=node_type, importance=np.random.uniform(0.7, 1.0))
        
        # Connect the new nodes in a dense cluster
        for i in range(num_new_nodes):
            for j in range(i+1, num_new_nodes):
                if np.random.random() < 0.7:  # Much denser connections
                    self.schema_graph.add_edge(
                        f"fork_node_{base_id + i}", 
                        f"fork_node_{base_id + j}", 
                        weight=np.random.uniform(0.8, 1.0)  # Higher weights
                    )
        
        # Strategy 3: Only minimally connect the new cluster to the old nodes
        # This creates a structural fork in the graph
        old_nodes = [n for n in self.schema_graph.nodes() if not n.startswith("fork_node_")]
        new_nodes = [n for n in self.schema_graph.nodes() if n.startswith("fork_node_")]
        
        # Add just a few bridge connections
        for _ in range(3):
            old_node = np.random.choice(old_nodes)
            new_node = np.random.choice(new_nodes)
            self.schema_graph.add_edge(old_node, new_node, weight=np.random.uniform(0.3, 0.5))


def demo_continuous_monitoring():
    """Demonstrate continuous monitoring of an agent for schema forks."""
    print("\n=== Continuous Monitoring Demo ===")
    
    # Create an agent and a detector
    agent = SimpleAgent(num_nodes=40)
    detector = SchemaForkDetector(
        history_window=10,
        drift_threshold=3.0,
        min_history_required=3
    )
    
    # Initialize the monitoring by taking a few snapshots of the initial state
    print("Taking initial snapshots...")
    for _ in range(4):
        detector.add_schema_snapshot(agent)
        # Simulate some time passing and small changes
        agent.evolve_schema_gradually()
        time.sleep(0.5)
    
    # Monitoring loop
    print("\nStarting continuous monitoring (press Ctrl+C to exit)...")
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Track drift and check for forks
            drift, is_fork = detector.monitor_agent(agent)
            
            # Print status
            print(f"Iteration {iteration}: Drift = {drift:.3f}", end="")
            if is_fork:
                print(" - FORK DETECTED!")
            else:
                print("")
            
            # Every 10 iterations, show more details and visualize
            if iteration % 10 == 0:
                print(f"\nDetailed metrics after {iteration} iterations:")
                metrics = detector.get_metrics()
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                
                # Visualize current state
                detector.visualize_trajectory(show=False)
                plt.savefig(f"drift_trajectory_{iteration}.png")
                plt.close()
                
                print(f"Drift trajectory saved to drift_trajectory_{iteration}.png")
            
            # For demo purposes, trigger a fork after 30 iterations
            if iteration == 30:
                print("\n[!] Triggering a dramatic schema change...")
                agent.cause_schema_fork()
            else:
                # Normal gradual evolution
                agent.evolve_schema_gradually()
            
            # Pause between iterations
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    # Final visualization
    print("\nGenerating final visualizations...")
    detector.visualize_trajectory(output_path="final_drift_trajectory.png", show=False)
    print("Drift trajectory saved to final_drift_trajectory.png")
    
    if len(detector.schema_embeddings) >= 3:
        detector.visualize_embedding_pca(output_path="schema_evolution_pca.png", show=False)
        print("PCA visualization saved to schema_evolution_pca.png")
    
    # Save detector state
    state_path = detector.save_state("fork_detector_final_state.json")
    print(f"Detector state saved to {state_path}")
    
    return detector


def demo_fork_analysis():
    """Demonstrate the analysis utility function."""
    print("\n=== Fork Analysis Utility Demo ===")
    
    # Create an agent
    agent = SimpleAgent(num_nodes=35)
    
    # First analysis with normal schema
    print("\nAnalyzing initial schema state...")
    analyze_agent_schema_fork(agent, visualize=False)
    
    # Evolve the schema normally for a while
    print("\nEvolving schema gradually...")
    for _ in range(5):
        agent.evolve_schema_gradually()
    
    # Second analysis after gradual evolution
    print("\nAnalyzing after gradual evolution...")
    analyze_agent_schema_fork(agent, visualize=False)
    
    # Cause a schema fork
    print("\nCausing dramatic schema changes that should trigger a fork...")
    agent.cause_schema_fork()
    
    # Final analysis after fork
    print("\nFinal analysis after schema disruption...")
    _, _, detector = analyze_agent_schema_fork(
        agent, 
        visualize=True,
        save_results=True,
        output_dir="fork_analysis_results"
    )
    
    return detector


if __name__ == "__main__":
    print("Schema Fork Detection Example\n")
    print("This demo shows how to use the SchemaForkDetector to monitor")
    print("an agent for identity forks as its schema evolves over time.")
    
    # Run the continuous monitoring demo
    detector1 = demo_continuous_monitoring()
    
    # Run the analysis utility demo
    detector2 = demo_fork_analysis()
    
    print("\nDemo completed. Check the output files for visualizations.") 