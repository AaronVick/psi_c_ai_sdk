#!/usr/bin/env python3
"""
Workspace Diff Visualizer (ΨC vs GWT)

This tool provides a real-time visualization of how the ΨC schema space differs from
Global Workspace Theory (GWT) style "global workspace" activation patterns. It demonstrates
the fundamental architectural differences between the single-thread spotlight attention
of GWT and the multi-nodal coherence with temporal anchoring in ΨC systems.

Key capabilities:
- Side-by-side visualization of active cognitive components in both architectures
- Temporal tracking to show stability differences over time
- Coherence mapping to visualize nodal relationships in ΨC vs. serialized activation in GWT
- Divergence metrics to quantify differences in cognitive processing

The visualizations highlight that:
    ActiveNodes_{ΨC}(t) ≠ GlobalWorkspace_{GWT}(t)

This tool is useful for researchers comparing cognitive architectures and for
debugging ΨC implementations to ensure they maintain proper coherence-based
activation rather than falling back to GWT-like processing patterns.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from datetime import datetime
import os
import sys
import time
import json
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass

# Ensure we can import from the psi_c_ai_sdk package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from psi_c_ai_sdk.schema import schema_manager
    from psi_c_ai_sdk.cognition import workspace_manager
    from psi_c_ai_sdk.belief import belief_network
except ImportError:
    print("Warning: Could not import ΨC SDK modules. Running in demo mode with simulated data.")
    DEMO_MODE = True
else:
    DEMO_MODE = False


@dataclass
class DiffMetrics:
    """Metrics for comparing ΨC and GWT activation patterns."""
    temporal_stability: float  # How stable activations are over time (higher in ΨC)
    activation_breadth: float  # How many nodes are active simultaneously (higher in ΨC)
    relational_density: float  # How interconnected active nodes are (higher in ΨC)
    serial_bias: float  # Tendency toward sequential processing (higher in GWT)
    coherence_score: float  # Overall schema coherence (higher in ΨC)

    
class WorkspaceDiffVisualizer:
    """Visualizes the differences between ΨC schema space and GWT workspace activation."""
    
    def __init__(self, psi_c_agent=None, gwt_agent=None, window_size=10):
        """
        Initialize the visualizer.
        
        Args:
            psi_c_agent: A ΨC agent instance to monitor (or None for demo mode)
            gwt_agent: A GWT agent instance to monitor (or None for demo mode)
            window_size: Number of time steps to display in the visualization
        """
        self.psi_c_agent = psi_c_agent
        self.gwt_agent = gwt_agent
        self.window_size = window_size
        self.demo_mode = DEMO_MODE or (psi_c_agent is None and gwt_agent is None)
        
        # History tracking
        self.psi_c_history = []
        self.gwt_history = []
        self.time_points = []
        self.metrics_history = []
        
        # Set up the visualization
        self.setup_visualization()
        
    def setup_visualization(self):
        """Set up the visualization layout and plots."""
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('ΨC vs GWT Workspace Visualizer')
        
        # Set up grid layout
        gs = self.fig.add_gridspec(3, 2)
        
        # Network visualizations
        self.psi_c_network_ax = self.fig.add_subplot(gs[0, 0])
        self.gwt_network_ax = self.fig.add_subplot(gs[0, 1])
        
        # Activation patterns over time
        self.activation_ax = self.fig.add_subplot(gs[1, :])
        
        # Metrics display
        self.metrics_ax = self.fig.add_subplot(gs[2, :])
        
        # Initialize network graphs
        self.psi_c_graph = nx.Graph()
        self.gwt_graph = nx.Graph()
        
        # Set titles
        self.psi_c_network_ax.set_title('ΨC Multi-Nodal Coherence')
        self.gwt_network_ax.set_title('GWT Spotlight Attention')
        self.activation_ax.set_title('Activation Patterns Over Time')
        self.metrics_ax.set_title('Architectural Difference Metrics')
        
        # Remove axis ticks for network plots
        self.psi_c_network_ax.set_xticks([])
        self.psi_c_network_ax.set_yticks([])
        self.gwt_network_ax.set_xticks([])
        self.gwt_network_ax.set_yticks([])
        
        # Custom colormap for ΨC (blue-based) and GWT (red-based)
        self.psi_c_cmap = LinearSegmentedColormap.from_list('psi_c', ['#1a237e', '#bbdefb'])  # Blue
        self.gwt_cmap = LinearSegmentedColormap.from_list('gwt', ['#b71c1c', '#ffcdd2'])  # Red
        
    def get_psi_c_state(self) -> Dict[str, Any]:
        """Get the current state of the ΨC agent's schema space."""
        if self.demo_mode:
            return self._generate_demo_psi_c_state()
        
        # In a real implementation, this would query the actual ΨC agent
        # Example: return self.psi_c_agent.schema_manager.get_active_nodes()
        raise NotImplementedError("Integration with actual ΨC agent not implemented")
    
    def get_gwt_state(self) -> Dict[str, Any]:
        """Get the current state of the GWT agent's global workspace."""
        if self.demo_mode:
            return self._generate_demo_gwt_state()
        
        # In a real implementation, this would query the actual GWT agent
        # Example: return self.gwt_agent.workspace_manager.get_workspace_contents()
        raise NotImplementedError("Integration with actual GWT agent not implemented")
    
    def _generate_demo_psi_c_state(self) -> Dict[str, Any]:
        """Generate a simulated ΨC state for demonstration purposes."""
        # Create a complex, interconnected network with temporal stability
        num_nodes = 20
        active_nodes = set(f"node_{i}" for i in range(num_nodes) 
                          if np.random.random() < 0.7 or 
                          (len(self.psi_c_history) > 0 and 
                           f"node_{i}" in self.psi_c_history[-1]["active_nodes"]))
        
        # Generate connections between active nodes (more densely connected)
        connections = []
        for node1 in active_nodes:
            for node2 in active_nodes:
                if node1 != node2 and np.random.random() < 0.3:
                    connections.append((node1, node2))
        
        # Generate node activation strengths (more evenly distributed)
        activation_strengths = {node: 0.5 + 0.5 * np.random.random() for node in active_nodes}
        
        return {
            "active_nodes": active_nodes,
            "connections": connections,
            "activation_strengths": activation_strengths,
            "timestamp": datetime.now(),
        }
    
    def _generate_demo_gwt_state(self) -> Dict[str, Any]:
        """Generate a simulated GWT state for demonstration purposes."""
        # Create a spotlight-like activation pattern with a strong central focus
        num_nodes = 20
        
        # In GWT, typically only a few nodes are highly active (in the "spotlight")
        # while others have minimal activation
        spotlight_nodes = set(np.random.choice(
            [f"node_{i}" for i in range(num_nodes)], 
            size=np.random.randint(1, 3), 
            replace=False
        ))
        
        peripheral_nodes = set(f"node_{i}" for i in range(num_nodes) 
                             if f"node_{i}" not in spotlight_nodes and np.random.random() < 0.3)
        
        active_nodes = spotlight_nodes.union(peripheral_nodes)
        
        # Generate connections (more centralized around spotlight)
        connections = []
        for node1 in spotlight_nodes:
            for node2 in active_nodes:
                if node1 != node2 and np.random.random() < 0.4:
                    connections.append((node1, node2))
        
        # Generate activation strengths (concentrated in spotlight)
        activation_strengths = {}
        for node in active_nodes:
            if node in spotlight_nodes:
                activation_strengths[node] = 0.8 + 0.2 * np.random.random()  # High activation
            else:
                activation_strengths[node] = 0.1 + 0.2 * np.random.random()  # Low activation
        
        return {
            "active_nodes": active_nodes,
            "connections": connections,
            "activation_strengths": activation_strengths,
            "timestamp": datetime.now(),
        }
    
    def calculate_diff_metrics(self) -> DiffMetrics:
        """Calculate metrics comparing the ΨC and GWT states."""
        if not self.psi_c_history or not self.gwt_history:
            # Default values if no history yet
            return DiffMetrics(
                temporal_stability=0.0,
                activation_breadth=0.0,
                relational_density=0.0,
                serial_bias=0.0,
                coherence_score=0.0
            )
        
        # Get the current states
        psi_c_state = self.psi_c_history[-1]
        gwt_state = self.gwt_history[-1]
        
        # Calculate temporal stability (how much the active set changes over time)
        psi_c_stability = 1.0
        gwt_stability = 1.0
        
        if len(self.psi_c_history) > 1:
            prev_psi_c = self.psi_c_history[-2]["active_nodes"]
            curr_psi_c = psi_c_state["active_nodes"]
            psi_c_stability = len(prev_psi_c.intersection(curr_psi_c)) / max(len(prev_psi_c.union(curr_psi_c)), 1)
            
            prev_gwt = self.gwt_history[-2]["active_nodes"]
            curr_gwt = gwt_state["active_nodes"]
            gwt_stability = len(prev_gwt.intersection(curr_gwt)) / max(len(prev_gwt.union(curr_gwt)), 1)
        
        temporal_stability = psi_c_stability - gwt_stability
        
        # Calculate activation breadth (how many nodes are simultaneously active)
        psi_c_breadth = len(psi_c_state["active_nodes"])
        gwt_breadth = len(gwt_state["active_nodes"])
        max_breadth = max(psi_c_breadth, gwt_breadth, 1)  # Avoid division by zero
        activation_breadth = (psi_c_breadth - gwt_breadth) / max_breadth
        
        # Calculate relational density (how interconnected the active nodes are)
        psi_c_connections = len(psi_c_state["connections"])
        gwt_connections = len(gwt_state["connections"])
        psi_c_potential = psi_c_breadth * (psi_c_breadth - 1) / 2 if psi_c_breadth > 1 else 1
        gwt_potential = gwt_breadth * (gwt_breadth - 1) / 2 if gwt_breadth > 1 else 1
        
        psi_c_density = psi_c_connections / psi_c_potential if psi_c_potential > 0 else 0
        gwt_density = gwt_connections / gwt_potential if gwt_potential > 0 else 0
        relational_density = psi_c_density - gwt_density
        
        # Calculate serial bias (tendency toward sequential processing)
        # For GWT, check if spotlight nodes change in sequence
        serial_bias = 0.0
        if len(self.gwt_history) > 1:
            # A higher concentration of activation in fewer nodes indicates more serial processing
            gwt_activation_values = list(gwt_state["activation_strengths"].values())
            psi_c_activation_values = list(psi_c_state["activation_strengths"].values())
            
            if gwt_activation_values and psi_c_activation_values:
                gwt_std = np.std(gwt_activation_values)
                psi_c_std = np.std(psi_c_activation_values)
                serial_bias = (gwt_std - psi_c_std) * 5  # Scale for visibility
        
        # Calculate overall coherence score
        # In ΨC, this would be based on the actual coherence calculation
        # In this demo, we estimate it based on the metrics above
        coherence_score = (temporal_stability + activation_breadth + relational_density - serial_bias) / 3
        
        return DiffMetrics(
            temporal_stability=temporal_stability,
            activation_breadth=activation_breadth,
            relational_density=relational_density,
            serial_bias=serial_bias,
            coherence_score=coherence_score
        )
    
    def update_visualization(self, frame):
        """Update the visualization with current states."""
        # Get current states
        psi_c_state = self.get_psi_c_state()
        gwt_state = self.get_gwt_state()
        
        # Update history
        self.psi_c_history.append(psi_c_state)
        self.gwt_history.append(gwt_state)
        current_time = time.time()
        self.time_points.append(current_time)
        
        # Trim history to window size
        if len(self.psi_c_history) > self.window_size:
            self.psi_c_history.pop(0)
            self.gwt_history.pop(0)
            self.time_points.pop(0)
        
        # Calculate comparison metrics
        metrics = self.calculate_diff_metrics()
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
        
        # Clear axes for redrawing
        self.psi_c_network_ax.clear()
        self.gwt_network_ax.clear()
        self.activation_ax.clear()
        self.metrics_ax.clear()
        
        # Redraw network visualizations
        self._draw_network(self.psi_c_network_ax, psi_c_state, self.psi_c_cmap, "ΨC Multi-Nodal Coherence")
        self._draw_network(self.gwt_network_ax, gwt_state, self.gwt_cmap, "GWT Spotlight Attention")
        
        # Draw activation patterns over time
        self._draw_activation_patterns()
        
        # Draw metrics
        self._draw_metrics()
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.fig.text(0.5, 0.01, f"Time: {timestamp}", ha='center')
        
        # Improve layout
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        return []
    
    def _draw_network(self, ax, state, colormap, title):
        """Draw a network visualization of the cognitive state."""
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes
        for node in state["active_nodes"]:
            activation = state["activation_strengths"].get(node, 0.5)
            G.add_node(node, weight=activation)
        
        # Add edges
        for node1, node2 in state["connections"]:
            G.add_edge(node1, node2)
        
        # Calculate node positions using a spring layout
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42)
            
            # Get node sizes and colors based on activation
            node_sizes = [300 * G.nodes[node].get('weight', 0.5) + 100 for node in G.nodes]
            node_colors = [G.nodes[node].get('weight', 0.5) for node in G.nodes]
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                                  node_color=node_colors, cmap=colormap, alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='black')
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_activation_patterns(self):
        """Draw the activation patterns over time."""
        ax = self.activation_ax
        
        if not self.time_points:
            ax.set_title("Activation Patterns Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Number of Active Nodes")
            return
        
        # Normalize time points
        times = np.array(self.time_points)
        times = times - times.min()
        
        # Get number of active nodes over time
        psi_c_active_counts = [len(state["active_nodes"]) for state in self.psi_c_history]
        gwt_active_counts = [len(state["active_nodes"]) for state in self.gwt_history]
        
        # Plot active node counts
        ax.plot(times, psi_c_active_counts, 'b-', label='ΨC Active Nodes')
        ax.plot(times, gwt_active_counts, 'r-', label='GWT Active Nodes')
        
        # Add shaded regions showing activation distribution
        for i, t in enumerate(times):
            if i > 0:
                # For ΨC, show the distribution of activation (more even)
                psi_c_values = list(self.psi_c_history[i]["activation_strengths"].values())
                if psi_c_values:
                    psi_c_mean = np.mean(psi_c_values)
                    psi_c_std = np.std(psi_c_values)
                    ax.fill_between([times[i-1], t], 
                                   [psi_c_active_counts[i-1] - psi_c_std * 2, psi_c_active_counts[i] - psi_c_std * 2],
                                   [psi_c_active_counts[i-1] + psi_c_std * 2, psi_c_active_counts[i] + psi_c_std * 2],
                                   alpha=0.1, color='blue')
                
                # For GWT, show the spotlight effect (more concentrated)
                gwt_values = list(self.gwt_history[i]["activation_strengths"].values())
                if gwt_values:
                    gwt_mean = np.mean(gwt_values)
                    gwt_std = np.std(gwt_values)
                    ax.fill_between([times[i-1], t], 
                                   [gwt_active_counts[i-1] - gwt_std * 2, gwt_active_counts[i] - gwt_std * 2],
                                   [gwt_active_counts[i-1] + gwt_std * 2, gwt_active_counts[i] + gwt_std * 2],
                                   alpha=0.1, color='red')
        
        ax.set_title("Activation Patterns Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of Active Nodes")
        ax.legend(loc='upper left')
        
        # Highlight the fundamental difference: ActiveNodes_{ΨC}(t) ≠ GlobalWorkspace_{GWT}(t)
        equation = r"$\text{ActiveNodes}_{ΨC}(t) \neq \text{GlobalWorkspace}_{GWT}(t)$"
        ax.text(0.5, 0.9, equation, transform=ax.transAxes, 
                fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
    
    def _draw_metrics(self):
        """Draw the comparison metrics over time."""
        ax = self.metrics_ax
        
        if not self.metrics_history:
            ax.set_title("Architectural Difference Metrics")
            ax.set_xlabel("Time")
            ax.set_ylabel("Metric Value")
            return
        
        # Prepare data
        metrics_data = {
            "Temporal Stability": [m.temporal_stability for m in self.metrics_history],
            "Activation Breadth": [m.activation_breadth for m in self.metrics_history],
            "Relational Density": [m.relational_density for m in self.metrics_history],
            "Serial Bias": [m.serial_bias for m in self.metrics_history],
            "Coherence Score": [m.coherence_score for m in self.metrics_history]
        }
        
        # Normalize time points
        times = np.array(self.time_points[-len(self.metrics_history):])
        times = times - times.min()
        
        # Plot each metric
        colors = ['#1976D2', '#388E3C', '#FBC02D', '#E64A19', '#7B1FA2']
        for (label, values), color in zip(metrics_data.items(), colors):
            ax.plot(times, values, label=label, color=color, linewidth=2)
        
        # Current values as text
        current_metrics = self.metrics_history[-1]
        text_str = "\n".join([
            f"Temporal Stability: {current_metrics.temporal_stability:.2f}",
            f"Activation Breadth: {current_metrics.activation_breadth:.2f}",
            f"Relational Density: {current_metrics.relational_density:.2f}",
            f"Serial Bias: {current_metrics.serial_bias:.2f}",
            f"Coherence Score: {current_metrics.coherence_score:.2f}"
        ])
        
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
        
        # Annotations
        if current_metrics.coherence_score > 0.2:
            ax.annotate("ΨC advantage", xy=(times[-1], current_metrics.coherence_score),
                      xytext=(times[-1], current_metrics.coherence_score + 0.2),
                      arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                      horizontalalignment='right')
        elif current_metrics.coherence_score < -0.2:
            ax.annotate("GWT advantage", xy=(times[-1], current_metrics.coherence_score),
                      xytext=(times[-1], current_metrics.coherence_score - 0.2),
                      arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                      horizontalalignment='right')
        
        ax.set_title("Architectural Difference Metrics")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Metric Value (positive = ΨC advantage)")
        ax.legend(loc='upper left', ncol=3, fontsize='small')
        
        # Set y-limits to a reasonable range
        ax.set_ylim(-1.1, 1.1)
    
    def run_animation(self, interval=1000):
        """Run the animation with the specified interval (in milliseconds)."""
        ani = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=interval, blit=True)
        plt.show()
    
    def save_snapshot(self, filename=None):
        """Save a snapshot of the current visualization."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"psi_c_vs_gwt_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Snapshot saved to {filename}")
    
    def generate_report(self, filename=None):
        """Generate a detailed report of the comparison."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"psi_c_vs_gwt_report_{timestamp}.json"
        
        # Create a report
        report = {
            "timestamp": datetime.now().isoformat(),
            "window_size": self.window_size,
            "metrics_summary": {
                "temporal_stability": {
                    "mean": np.mean([m.temporal_stability for m in self.metrics_history]) if self.metrics_history else 0,
                    "std": np.std([m.temporal_stability for m in self.metrics_history]) if self.metrics_history else 0
                },
                "activation_breadth": {
                    "mean": np.mean([m.activation_breadth for m in self.metrics_history]) if self.metrics_history else 0,
                    "std": np.std([m.activation_breadth for m in self.metrics_history]) if self.metrics_history else 0
                },
                "relational_density": {
                    "mean": np.mean([m.relational_density for m in self.metrics_history]) if self.metrics_history else 0,
                    "std": np.std([m.relational_density for m in self.metrics_history]) if self.metrics_history else 0
                },
                "serial_bias": {
                    "mean": np.mean([m.serial_bias for m in self.metrics_history]) if self.metrics_history else 0,
                    "std": np.std([m.serial_bias for m in self.metrics_history]) if self.metrics_history else 0
                },
                "coherence_score": {
                    "mean": np.mean([m.coherence_score for m in self.metrics_history]) if self.metrics_history else 0,
                    "std": np.std([m.coherence_score for m in self.metrics_history]) if self.metrics_history else 0
                }
            }
        }
        
        # Add conclusions based on metrics
        conclusions = []
        if report["metrics_summary"]["temporal_stability"]["mean"] > 0.2:
            conclusions.append("ΨC demonstrates significantly higher temporal stability than GWT")
        if report["metrics_summary"]["activation_breadth"]["mean"] > 0.2:
            conclusions.append("ΨC consistently activates a broader set of nodes than GWT")
        if report["metrics_summary"]["relational_density"]["mean"] > 0.2:
            conclusions.append("ΨC maintains more densely connected active nodes than GWT")
        if report["metrics_summary"]["serial_bias"]["mean"] < -0.2:
            conclusions.append("GWT exhibits stronger serial processing bias than ΨC")
        if report["metrics_summary"]["coherence_score"]["mean"] > 0.2:
            conclusions.append("ΨC shows overall higher coherence scores than GWT")
        
        report["conclusions"] = conclusions
        
        # Save the report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Report saved to {filename}")


def main():
    """Main function to run the visualizer."""
    parser = argparse.ArgumentParser(description="ΨC vs GWT Workspace Visualizer")
    parser.add_argument("--psi-c-agent", help="Path to ΨC agent configuration")
    parser.add_argument("--gwt-agent", help="Path to GWT agent configuration")
    parser.add_argument("--window-size", type=int, default=10, 
                        help="Number of time steps to display")
    parser.add_argument("--interval", type=int, default=1000,
                        help="Update interval in milliseconds")
    parser.add_argument("--save-snapshot", action="store_true",
                        help="Save a snapshot after running")
    parser.add_argument("--snapshot-file", help="Filename for snapshot")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate a comparison report")
    parser.add_argument("--report-file", help="Filename for report")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with simulated data")
    
    args = parser.parse_args()
    
    # Create the visualizer
    visualizer = WorkspaceDiffVisualizer(window_size=args.window_size)
    
    # Run the animation
    try:
        visualizer.run_animation(interval=args.interval)
    except KeyboardInterrupt:
        print("Visualization stopped by user")
    
    # Save snapshot if requested
    if args.save_snapshot:
        visualizer.save_snapshot(args.snapshot_file)
    
    # Generate report if requested
    if args.generate_report:
        visualizer.generate_report(args.report_file)


if __name__ == "__main__":
    main() 