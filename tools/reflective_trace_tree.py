#!/usr/bin/env python3
"""
Reflective Conflict Trace Tree

This tool builds a real-time tree visualization of how cognitive agents evolve through:
- Contradiction detection
- Belief mutations
- Identity shifts

The tree represents the agent's cognitive journey, where:
- Each node = cognitive state S_t
- Each edge = belief divergence ΔB_t
- Branch points = decision moments triggered by contradictions

This visualization helps researchers track emergent cognition by showing
how the resolution of contradictions leads to new stable states.

Mathematical model:
- Reflective utility: U_r = ∑(ΔΨC)_i - Contradiction_Penalty_i
- Where higher utility reflects productive cognitive development
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from datetime import datetime
import os
import sys
import time
import json
import uuid
import logging
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass, field
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure we can import from the psi_c_ai_sdk package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from psi_c_ai_sdk.belief import belief_network
    from psi_c_ai_sdk.schema import schema_manager 
    MOCK_MODE = False
except ImportError:
    logger.warning("Could not import ΨC SDK modules. Running in demo mode with simulated data.")
    MOCK_MODE = True


@dataclass
class Belief:
    """Represents a single belief in the system."""
    id: str
    content: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def divergence_from(self, other: 'Belief') -> float:
        """Calculate divergence from another belief."""
        # Simple implementation: if content differs completely, divergence = 1.0
        # In a real system, this would use semantic similarity
        if self.content == other.content:
            return 0.0
        return 1.0


@dataclass
class Contradiction:
    """Represents a contradiction between beliefs."""
    id: str
    belief_ids: List[str]
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class BeliefState:
    """Represents a belief state at a specific point in time."""
    id: str
    beliefs: Dict[str, Belief]
    contradictions: List[Contradiction] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    psi_c_score: float = 0.0
    contradiction_level: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def belief_divergence(self, other: 'BeliefState') -> float:
        """Calculate the belief divergence between states."""
        if not other.beliefs:
            return 1.0
        
        # Calculate divergence between common beliefs
        common_beliefs = set(self.beliefs.keys()) & set(other.beliefs.keys())
        if not common_beliefs:
            return 1.0
        
        total_divergence = 0.0
        for belief_id in common_beliefs:
            total_divergence += self.beliefs[belief_id].divergence_from(other.beliefs[belief_id])
        
        # Also consider beliefs present in one state but not the other
        only_in_self = len(set(self.beliefs.keys()) - common_beliefs)
        only_in_other = len(set(other.beliefs.keys()) - common_beliefs)
        
        # Calculate normalized divergence
        norm_factor = len(common_beliefs) + only_in_self + only_in_other
        return (total_divergence + only_in_self + only_in_other) / norm_factor
    
    def calculate_psi_c_score(self) -> float:
        """Calculate the ΨC score for this belief state."""
        if not self.beliefs:
            return 0.0
        
        # In a real system, this would do a coherence calculation
        # For the demo, we use a simple function of belief confidence and contradiction
        avg_confidence = np.mean([belief.confidence for belief in self.beliefs.values()])
        
        # Calculate contradiction penalty
        contradiction_penalty = self.contradiction_level
        
        # ΨC score calculation
        self.psi_c_score = avg_confidence * (1.0 - contradiction_penalty)
        return self.psi_c_score
    
    def calculate_contradiction_level(self) -> float:
        """Calculate the contradiction level for this belief state."""
        if not self.contradictions:
            return 0.0
        
        # Simple average of contradiction severities
        self.contradiction_level = np.mean([c.severity for c in self.contradictions])
        return self.contradiction_level
    
    def reflective_utility(self, parent: Optional['BeliefState'] = None) -> float:
        """Calculate the reflective utility of this state transition."""
        if parent is None:
            return self.psi_c_score
        
        # Calculate the change in ΨC
        delta_psi_c = self.psi_c_score - parent.psi_c_score
        
        # Calculate contradiction penalty
        contradiction_penalty = self.contradiction_level
        
        # Calculate the reflective utility
        return delta_psi_c - contradiction_penalty


class ConflictTraceTree:
    """
    The main class for tracking and visualizing cognitive evolution.
    
    This class maintains a tree of belief states, tracking how the agent's
    cognitive state evolves through contradictions and their resolutions.
    """
    
    def __init__(self):
        """Initialize the conflict trace tree."""
        self.states: Dict[str, BeliefState] = {}
        self.root_id: Optional[str] = None
        self.current_id: Optional[str] = None
        self.graph = nx.DiGraph()
        self.layout = None
        self.fig = None
        self.ax = None
        self.history: List[str] = []  # History of visited state IDs
        
        # Create custom colormaps
        self.psi_c_cmap = plt.cm.viridis
        self.contradiction_cmap = LinearSegmentedColormap.from_list(
            'contradiction', ['green', 'yellow', 'red'])
        
    def add_state(self, state: BeliefState) -> str:
        """
        Add a new belief state to the tree.
        
        Args:
            state: The belief state to add
            
        Returns:
            The ID of the added state
        """
        # Generate ID if needed
        if not state.id:
            state.id = str(uuid.uuid4())
        
        # Set as root if first state
        if not self.root_id:
            self.root_id = state.id
        
        # Add to states dictionary
        self.states[state.id] = state
        
        # Update current ID
        self.current_id = state.id
        
        # Add to history
        self.history.append(state.id)
        
        # Update graph
        self.graph.add_node(state.id)
        
        # Connect to parent if exists
        if state.parent_id and state.parent_id in self.states:
            parent = self.states[state.parent_id]
            parent.children_ids.append(state.id)
            
            # Calculate belief divergence for edge weight
            divergence = state.belief_divergence(parent)
            
            # Add edge to graph
            self.graph.add_edge(state.parent_id, state.id, weight=divergence)
        
        # Recalculate layout if needed
        if len(self.states) > 1:
            self.layout = None
        
        return state.id
    
    def add_child_state(self, parent_id: str, beliefs: Dict[str, Belief], 
                        contradictions: List[Contradiction] = None) -> str:
        """
        Add a child state to the given parent.
        
        Args:
            parent_id: The ID of the parent state
            beliefs: The beliefs for the new state
            contradictions: Any contradictions to include
            
        Returns:
            The ID of the new state
        """
        if parent_id not in self.states:
            raise ValueError(f"Parent state {parent_id} not found")
        
        # Create new state
        new_state = BeliefState(
            id=str(uuid.uuid4()),
            beliefs=beliefs,
            contradictions=contradictions or [],
            parent_id=parent_id,
            timestamp=time.time()
        )
        
        # Calculate metrics
        new_state.calculate_contradiction_level()
        new_state.calculate_psi_c_score()
        
        # Add to tree
        return self.add_state(new_state)
    
    def get_state(self, state_id: str) -> Optional[BeliefState]:
        """Get a state by ID."""
        return self.states.get(state_id)
    
    def get_current_state(self) -> Optional[BeliefState]:
        """Get the current state."""
        if self.current_id:
            return self.states.get(self.current_id)
        return None
    
    def set_current_state(self, state_id: str):
        """Set the current state."""
        if state_id in self.states:
            self.current_id = state_id
            self.history.append(state_id)
    
    def get_path_to_current(self) -> List[str]:
        """Get the path from root to current state."""
        if not self.current_id or not self.root_id:
            return []
        
        try:
            return nx.shortest_path(self.graph, self.root_id, self.current_id)
        except nx.NetworkXNoPath:
            return []
    
    def calculate_reflective_utility(self, state_id: str) -> float:
        """Calculate the reflective utility for a state."""
        state = self.states.get(state_id)
        if not state:
            return 0.0
        
        parent = self.states.get(state.parent_id) if state.parent_id else None
        return state.reflective_utility(parent)
    
    def _get_node_attributes(self):
        """Get node attributes for visualization."""
        node_colors = []
        node_sizes = []
        node_borders = []
        border_widths = []
        
        for node in self.graph.nodes():
            state = self.states[node]
            
            # Node color based on psi_c_score
            node_colors.append(state.psi_c_score)
            
            # Node size based on number of beliefs
            node_sizes.append(300 + 50 * len(state.beliefs))
            
            # Border color based on contradiction level
            node_borders.append(state.contradiction_level)
            
            # Border width: thicker for current/root nodes
            if node == self.current_id:
                border_widths.append(3.0)
            elif node == self.root_id:
                border_widths.append(2.0)
            else:
                border_widths.append(1.0)
        
        return node_colors, node_sizes, node_borders, border_widths
    
    def _get_edge_attributes(self):
        """Get edge attributes for visualization."""
        edge_widths = []
        
        for u, v, data in self.graph.edges(data=True):
            # Edge width based on belief divergence
            weight = data.get('weight', 0.5)
            edge_widths.append(1.0 + 5.0 * weight)
        
        return edge_widths
    
    def visualize(self, figsize=(12, 8), output_file=None):
        """
        Visualize the current state of the conflict trace tree.
        
        Args:
            figsize: Figure size in inches
            output_file: If provided, save visualization to this file
        """
        if not self.states:
            logger.warning("No states to visualize")
            return
        
        # Create figure if needed
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Clear the axes
        self.ax.clear()
        
        # Calculate layout if needed
        if self.layout is None:
            if len(self.states) < 20:
                self.layout = nx.spring_layout(self.graph, seed=42)
            else:
                # Use hierarchical layout for larger trees
                self.layout = nx.nx_pydot.graphviz_layout(self.graph, prog="dot")
        
        # Get node and edge attributes
        node_colors, node_sizes, node_borders, border_widths = self._get_node_attributes()
        edge_widths = self._get_edge_attributes()
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, self.layout, ax=self.ax,
            width=edge_widths, alpha=0.7,
            edge_color='gray'
        )
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            self.graph, self.layout, ax=self.ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=self.psi_c_cmap,
            edgecolors=[self.contradiction_cmap(c) for c in node_borders],
            linewidths=border_widths
        )
        
        # Add colorbar for ΨC score
        cbar = plt.colorbar(nodes, ax=self.ax)
        cbar.set_label('ΨC Score')
        
        # Add legend for contradiction level
        contradiction_legend = [
            mpatches.Patch(color=self.contradiction_cmap(0.0), label='No Contradictions'),
            mpatches.Patch(color=self.contradiction_cmap(0.5), label='Moderate Contradictions'),
            mpatches.Patch(color=self.contradiction_cmap(1.0), label='Severe Contradictions')
        ]
        self.ax.legend(handles=contradiction_legend, loc='upper right')
        
        # Add labels for selected nodes
        labels = {}
        for node in self.graph.nodes():
            if node == self.current_id or node == self.root_id:
                if node == self.current_id:
                    labels[node] = "Current"
                else:
                    labels[node] = "Root"
        
        nx.draw_networkx_labels(
            self.graph, self.layout, ax=self.ax,
            labels=labels, font_size=10
        )
        
        # Set title
        self.ax.set_title("Reflective Conflict Trace Tree")
        
        # Set axes limits to fit all nodes
        self.ax.axis('off')
        
        # If output file provided, save figure
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def visualize_path(self, figsize=(12, 8), output_file=None):
        """
        Visualize the path from root to current state.
        
        Args:
            figsize: Figure size in inches
            output_file: If provided, save visualization to this file
        """
        path = self.get_path_to_current()
        if not path or len(path) < 2:
            logger.warning("No valid path to visualize")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Extract states on the path
        path_states = [self.states[state_id] for state_id in path]
        
        # Calculate metrics along the path
        psi_c_scores = [state.psi_c_score for state in path_states]
        contradiction_levels = [state.contradiction_level for state in path_states]
        reflective_utilities = [self.calculate_reflective_utility(state_id) for state_id in path]
        
        # Plot ΨC score and contradiction level
        ax1 = axes[0]
        x = range(len(path))
        ax1.plot(x, psi_c_scores, 'b-', marker='o', label='ΨC Score')
        ax1.plot(x, contradiction_levels, 'r-', marker='x', label='Contradiction Level')
        ax1.set_xlabel('State Transitions')
        ax1.set_ylabel('Value')
        ax1.set_title('Cognitive Evolution Along Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot reflective utility
        ax2 = axes[1]
        ax2.bar(x[1:], reflective_utilities[1:], color='green', alpha=0.7)
        ax2.set_xlabel('State Transitions')
        ax2.set_ylabel('Reflective Utility')
        ax2.set_title('Reflective Utility of Transitions')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for key points
        for i in range(1, len(path)):
            if reflective_utilities[i] > 0.2:
                ax2.annotate(
                    f"+{reflective_utilities[i]:.2f}",
                    xy=(i, reflective_utilities[i]),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center'
                )
            elif reflective_utilities[i] < -0.2:
                ax2.annotate(
                    f"{reflective_utilities[i]:.2f}",
                    xy=(i, reflective_utilities[i]),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center'
                )
        
        # If output file provided, save figure
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Path visualization saved to {output_file}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def export_tree(self, output_file):
        """
        Export the tree to a JSON file.
        
        Args:
            output_file: File to write the exported tree to
        """
        data = {
            "states": {},
            "edges": [],
            "root_id": self.root_id,
            "current_id": self.current_id,
            "history": self.history
        }
        
        # Export states
        for state_id, state in self.states.items():
            data["states"][state_id] = {
                "id": state.id,
                "psi_c_score": state.psi_c_score,
                "contradiction_level": state.contradiction_level,
                "parent_id": state.parent_id,
                "children_ids": state.children_ids,
                "timestamp": state.timestamp,
                "belief_count": len(state.beliefs),
                "contradiction_count": len(state.contradictions),
                "metadata": state.metadata
            }
        
        # Export edges
        for u, v, data in self.graph.edges(data=True):
            data["edges"].append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 0.5)
            })
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Tree exported to {output_file}")
    
    def generate_dashboard(self, output_dir=None):
        """
        Generate a dashboard visualization of the conflict trace tree.
        
        Args:
            output_dir: Directory to save dashboard files to
        """
        if not self.states:
            logger.warning("No states to visualize")
            return
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save tree visualization
            tree_file = os.path.join(output_dir, "conflict_trace_tree.png")
            self.visualize(output_file=tree_file)
            
            # Save path visualization
            path_file = os.path.join(output_dir, "cognitive_evolution_path.png")
            self.visualize_path(output_file=path_file)
            
            # Export tree data
            data_file = os.path.join(output_dir, "conflict_trace_data.json")
            self.export_tree(data_file)
            
            # Generate HTML dashboard
            dashboard_file = os.path.join(output_dir, "conflict_trace_dashboard.html")
            
            # Create dashboard content
            with open(dashboard_file, 'w') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Reflective Conflict Trace Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #1a237e; color: white; padding: 20px; margin-bottom: 20px; }}
        .viz-container {{ margin-bottom: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; margin-right: 15px; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .state-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .state-table th, .state-table td {{ padding: 10px; border: 1px solid #ddd; }}
        .state-table th {{ background-color: #f2f2f2; }}
        .timestamp {{ font-size: 12px; color: #666; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reflective Conflict Trace Dashboard</h1>
            <p>Visualizing cognitive evolution through contradictions and mutations</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{len(self.states)}</div>
                <div class="metric-label">Total States</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(self.graph.edges())}</div>
                <div class="metric-label">State Transitions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.states[self.current_id].psi_c_score:.2f}</div>
                <div class="metric-label">Current ΨC Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.states[self.current_id].contradiction_level:.2f}</div>
                <div class="metric-label">Current Contradiction Level</div>
            </div>
        </div>
        
        <div class="viz-container">
            <h2>Conflict Trace Tree</h2>
            <img src="conflict_trace_tree.png" alt="Conflict Trace Tree" style="max-width: 100%;" />
        </div>
        
        <div class="viz-container">
            <h2>Cognitive Evolution Path</h2>
            <img src="cognitive_evolution_path.png" alt="Cognitive Evolution Path" style="max-width: 100%;" />
        </div>
        
        <div>
            <h2>Recent State Transitions</h2>
            <table class="state-table">
                <tr>
                    <th>State ID</th>
                    <th>ΨC Score</th>
                    <th>Contradiction Level</th>
                    <th>Beliefs</th>
                    <th>Reflective Utility</th>
                    <th>Timestamp</th>
                </tr>
                {"".join([f"<tr><td>{state_id}</td><td>{self.states[state_id].psi_c_score:.2f}</td><td>{self.states[state_id].contradiction_level:.2f}</td><td>{len(self.states[state_id].beliefs)}</td><td>{self.calculate_reflective_utility(state_id):.2f}</td><td>{datetime.fromtimestamp(self.states[state_id].timestamp).strftime('%Y-%m-%d %H:%M:%S')}</td></tr>" for state_id in self.history[-10:]])}
            </table>
        </div>
        
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>""")
            
            logger.info(f"Dashboard saved to {dashboard_file}")
        else:
            # Just show the visualizations
            self.visualize()
            self.visualize_path()


def create_mock_tree():
    """Create a mock tree for demonstration."""
    tree = ConflictTraceTree()
    
    # Create initial state
    initial_beliefs = {
        "b1": Belief("b1", "The sky is blue", 0.9, ["observation"]),
        "b2": Belief("b2", "Water is wet", 0.95, ["common knowledge"]),
        "b3": Belief("b3", "Fire is hot", 0.95, ["common knowledge"])
    }
    
    root_state = BeliefState(
        id="state_0",
        beliefs=initial_beliefs,
        timestamp=time.time() - 3600  # 1 hour ago
    )
    root_state.calculate_contradiction_level()
    root_state.calculate_psi_c_score()
    
    root_id = tree.add_state(root_state)
    
    # Create a contradiction
    contradiction = Contradiction(
        id="c1",
        belief_ids=["b1", "b4"],
        severity=0.7,
        description="New observation contradicts existing belief about sky color"
    )
    
    # Create a new state with the contradiction
    new_beliefs = initial_beliefs.copy()
    new_beliefs["b4"] = Belief("b4", "The sky appears red at sunset", 0.8, ["observation"])
    
    state_1 = BeliefState(
        id="state_1",
        beliefs=new_beliefs,
        contradictions=[contradiction],
        parent_id=root_id,
        timestamp=time.time() - 2400  # 40 minutes ago
    )
    state_1.calculate_contradiction_level()
    state_1.calculate_psi_c_score()
    
    state_1_id = tree.add_state(state_1)
    
    # Create two resolution paths
    
    # Path 1: Refine belief
    refined_beliefs = new_beliefs.copy()
    refined_beliefs["b1"] = Belief("b1", "The sky is usually blue but can be other colors", 0.85, ["updated model"])
    del refined_beliefs["b4"]  # Remove the contradictory belief
    
    state_2a = BeliefState(
        id="state_2a",
        beliefs=refined_beliefs,
        parent_id=state_1_id,
        timestamp=time.time() - 1800  # 30 minutes ago
    )
    state_2a.calculate_contradiction_level()
    state_2a.calculate_psi_c_score()
    
    state_2a_id = tree.add_state(state_2a)
    
    # Path 2: Add context
    context_beliefs = new_beliefs.copy()
    context_beliefs["b5"] = Belief("b5", "Sky color depends on time of day", 0.9, ["contextual understanding"])
    
    state_2b = BeliefState(
        id="state_2b",
        beliefs=context_beliefs,
        parent_id=state_1_id,
        timestamp=time.time() - 1800  # 30 minutes ago
    )
    state_2b.calculate_contradiction_level()
    state_2b.calculate_psi_c_score()
    
    state_2b_id = tree.add_state(state_2b)
    
    # Add another state to path 2
    integrated_beliefs = context_beliefs.copy()
    integrated_beliefs["b1"] = Belief("b1", "The sky color varies with atmospheric conditions", 0.92, ["integrated model"])
    
    state_3b = BeliefState(
        id="state_3b",
        beliefs=integrated_beliefs,
        parent_id=state_2b_id,
        timestamp=time.time() - 900  # 15 minutes ago
    )
    state_3b.calculate_contradiction_level()
    state_3b.calculate_psi_c_score()
    
    state_3b_id = tree.add_state(state_3b)
    
    # Set current state
    tree.set_current_state(state_3b_id)
    
    return tree


def main():
    """Main function to demonstrate the conflict trace tree."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reflective Conflict Trace Tree Visualization")
    parser.add_argument("--output-dir", type=str, help="Directory to save dashboard")
    parser.add_argument("--export-only", action="store_true", help="Export data without visualization")
    
    args = parser.parse_args()
    
    logger.info("Creating mock conflict trace tree...")
    tree = create_mock_tree()
    
    if args.output_dir:
        logger.info(f"Generating dashboard in {args.output_dir}...")
        tree.generate_dashboard(args.output_dir)
    elif not args.export_only:
        logger.info("Visualizing conflict trace tree...")
        tree.visualize()
        tree.visualize_path()
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 