"""
Shared Visualization Module for ΨC-AI SDK Multi-Agent System

This module provides tools for visualizing and analyzing shared schema graphs 
across multiple ΨC-AI agents. It enables visual exploration of belief overlaps,
contradictions, and conceptual alignment between agents.
"""

import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Set, Tuple, Optional, Any, Union

from psi_c_ai_sdk.schema.schema import SchemaGraph, NodeType
from psi_c_ai_sdk.multi_agent.shared_schema import SharedSchemaBuilder
from psi_c_ai_sdk.multi_agent.belief_negotiation import BeliefConsensus
from psi_c_ai_sdk.memory.memory import Memory


logger = logging.getLogger(__name__)


class AgentColorMap:
    """
    Manages color assignments for different agents in visualizations.
    
    This class ensures consistent color coding across different visualizations
    to help identify which elements belong to which agent.
    """
    
    # Predefined color set for agents
    DEFAULT_COLORS = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Teal
    ]
    
    def __init__(self):
        """Initialize the color map."""
        self.agent_colors = {}
        self.next_color_index = 0
    
    def get_agent_color(self, agent_id: str) -> str:
        """
        Get the color for an agent, assigning a new one if needed.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            Hex color code for the agent
        """
        if agent_id not in self.agent_colors:
            # Assign next color in the list
            color = self.DEFAULT_COLORS[self.next_color_index % len(self.DEFAULT_COLORS)]
            self.agent_colors[agent_id] = color
            self.next_color_index += 1
        
        return self.agent_colors[agent_id]
    
    def get_color_map(self) -> Dict[str, str]:
        """
        Get the complete agent color mapping.
        
        Returns:
            Dictionary mapping agent IDs to color codes
        """
        return self.agent_colors


class SchemaOverlapVisualizer:
    """
    Visualizes schema overlaps and relationships between multiple agents.
    
    This class provides tools for creating visual representations of shared 
    schemas, highlighting areas of agreement, conflict, and unique beliefs.
    """
    
    def __init__(self, shared_schema_builder: SharedSchemaBuilder):
        """
        Initialize with a shared schema builder.
        
        Args:
            shared_schema_builder: Builder containing agent schemas
        """
        self.shared_builder = shared_schema_builder
        self.color_map = AgentColorMap()
    
    def visualize_overlap_network(
        self, 
        figsize: Tuple[int, int] = (12, 10), 
        title: str = "Multi-Agent Schema Overlap",
        output_file: Optional[str] = None,
        show_agent_colors: bool = True,
        show_labels: bool = True,
        layout_method: str = "spring"
    ) -> plt.Figure:
        """
        Visualize the network of overlapping concepts across agents.
        
        Args:
            figsize: Size of the figure
            title: Title for the visualization
            output_file: Optional file path to save the visualization
            show_agent_colors: Whether to color nodes by agent
            show_labels: Whether to show node labels
            layout_method: Layout algorithm to use
            
        Returns:
            The matplotlib figure object
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the shared graph
        shared_graph = self.shared_builder.visualize_shared_schema()
        
        # Create position layout
        if layout_method == "spring":
            pos = nx.spring_layout(shared_graph, seed=42)
        elif layout_method == "kamada_kawai":
            pos = nx.kamada_kawai_layout(shared_graph)
        elif layout_method == "circular":
            pos = nx.circular_layout(shared_graph)
        else:
            pos = nx.spring_layout(shared_graph, seed=42)
        
        # Create node colors based on agent
        node_colors = []
        for node in shared_graph.nodes():
            agent_id = node.split(':')[0]
            node_colors.append(self.color_map.get_agent_color(agent_id))
        
        # Draw the network
        nx.draw_networkx_nodes(
            shared_graph, 
            pos, 
            node_color=node_colors if show_agent_colors else "#1f77b4",
            node_size=100,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            shared_graph, 
            pos, 
            alpha=0.5
        )
        
        if show_labels:
            # Create labels (use label attribute if available)
            labels = {}
            for node in shared_graph.nodes():
                node_data = shared_graph.nodes[node]
                label = node_data.get("label", node.split(':')[1])
                # Truncate long labels
                if len(label) > 20:
                    label = label[:17] + "..."
                labels[node] = label
            
            nx.draw_networkx_labels(
                shared_graph, 
                pos, 
                labels=labels,
                font_size=8,
                font_family="sans-serif"
            )
        
        # Add legend for agent colors
        if show_agent_colors:
            agent_handles = []
            agent_ids = set()
            
            # Get all agent IDs from nodes
            for node in shared_graph.nodes():
                agent_ids.add(node.split(':')[0])
            
            # Create legend handles
            for agent_id in sorted(agent_ids):
                color = self.color_map.get_agent_color(agent_id)
                agent_handles.append(plt.Line2D(
                    [0], [0], 
                    marker='o', 
                    color='w',
                    markerfacecolor=color, 
                    markersize=10, 
                    label=agent_id
                ))
            
            plt.legend(
                handles=agent_handles, 
                title="Agents",
                loc="upper right", 
                bbox_to_anchor=(1.1, 1)
            )
        
        # Add title and finalize layout
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Save to file if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_conceptual_alignment(
        self, 
        figsize: Tuple[int, int] = (10, 8),
        threshold: float = 0.3,
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a matrix visualization showing conceptual alignment between agents.
        
        Args:
            figsize: Size of the figure
            threshold: Similarity threshold for considering concepts aligned
            output_file: Optional file path to save the visualization
            
        Returns:
            The matplotlib figure object
        """
        # Get all agent IDs
        agent_ids = list(self.shared_builder.agent_graphs.keys())
        num_agents = len(agent_ids)
        
        if num_agents < 2:
            logger.warning("Need at least 2 agents to visualize conceptual alignment")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient agents for alignment visualization", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Create a similarity matrix
        similarity_matrix = np.zeros((num_agents, num_agents))
        
        # Find overlapping concepts
        overlap_nodes = self.shared_builder.find_overlap_nodes()
        
        # Calculate similarity based on overlapping concepts
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Perfect self-similarity
                    continue
                
                # Count concepts that both agents have
                agent1_schema = self.shared_builder.agent_graphs[agent1]
                agent2_schema = self.shared_builder.agent_graphs[agent2]
                
                agent1_concepts = set(
                    agent1_schema.graph.nodes[node].get("label", node) 
                    for node in agent1_schema.graph.nodes()
                )
                agent2_concepts = set(
                    agent2_schema.graph.nodes[node].get("label", node) 
                    for node in agent2_schema.graph.nodes()
                )
                
                # Calculate Jaccard similarity
                intersection = len(agent1_concepts.intersection(agent2_concepts))
                union = len(agent1_concepts.union(agent2_concepts))
                
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i, j] = similarity
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a custom colormap (from red to white to blue)
        cmap = LinearSegmentedColormap.from_list(
            'alignment_cmap', 
            ['#d62728', '#ffffff', '#1f77b4']
        )
        
        # Plot the matrix
        im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Conceptual Similarity')
        
        # Add labels and ticks
        ax.set_xticks(np.arange(num_agents))
        ax.set_yticks(np.arange(num_agents))
        ax.set_xticklabels(agent_ids)
        ax.set_yticklabels(agent_ids)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values to cells
        for i in range(num_agents):
            for j in range(num_agents):
                text_color = "black" if 0.3 <= similarity_matrix[i, j] <= 0.7 else "white"
                ax.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                       ha="center", va="center", color=text_color)
        
        # Add title
        ax.set_title("Agent Conceptual Alignment Matrix")
        
        # Add threshold line
        plt.axhline(y=-0.5, xmin=0, xmax=num_agents, color='black', linestyle='-', linewidth=2)
        plt.axvline(x=-0.5, ymin=0, ymax=num_agents, color='black', linestyle='-', linewidth=2)
        
        plt.tight_layout()
        
        # Save to file if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_belief_contradictions(
        self,
        figsize: Tuple[int, int] = (12, 10),
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize contradictions in beliefs between agents.
        
        Args:
            figsize: Size of the figure
            output_file: Optional file path to save the visualization
            
        Returns:
            The matplotlib figure object
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get all agent IDs
        agent_ids = list(self.shared_builder.agent_graphs.keys())
        
        # Create a contradiction graph
        contradiction_graph = nx.Graph()
        
        # Extract beliefs from schemas and find contradictions
        agent_beliefs = {}
        
        for agent_id, schema in self.shared_builder.agent_graphs.items():
            # Extract belief nodes
            beliefs = []
            for node_id, node_data in schema.graph.nodes(data=True):
                if node_data.get("type") == NodeType.BELIEF.value:
                    beliefs.append((node_id, node_data))
            
            agent_beliefs[agent_id] = beliefs
        
        # Add nodes to the contradiction graph
        for agent_id, beliefs in agent_beliefs.items():
            for node_id, node_data in beliefs:
                # Create a unique ID for the belief in the contradiction graph
                contradiction_node_id = f"{agent_id}:{node_id}"
                
                # Extract label and content
                label = node_data.get("label", node_id)
                content = node_data.get("content", "")
                
                # Add to graph
                contradiction_graph.add_node(
                    contradiction_node_id,
                    agent_id=agent_id,
                    label=label,
                    content=content,
                    original_id=node_id
                )
        
        # Find potential contradictions based on opposing statements
        # This is a simplified approach - in a real system, you'd use the 
        # ContradictionDetector from the schema module
        
        # For demonstration, we'll use a simple heuristic where beliefs with
        # similar labels but different agents are potentially contradictory
        
        # Group nodes by label
        nodes_by_label = {}
        for node in contradiction_graph.nodes():
            label = contradiction_graph.nodes[node].get("label", "")
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(node)
        
        # Add edges between potentially contradictory beliefs
        edge_count = 0
        for label, nodes in nodes_by_label.items():
            if len(nodes) > 1:
                # Add edges between all pairs with different agent IDs
                for i, node1 in enumerate(nodes):
                    agent1 = contradiction_graph.nodes[node1].get("agent_id")
                    for node2 in nodes[i+1:]:
                        agent2 = contradiction_graph.nodes[node2].get("agent_id")
                        if agent1 != agent2:
                            # Add edge representing potential contradiction
                            contradiction_graph.add_edge(node1, node2, weight=1)
                            edge_count += 1
        
        if edge_count == 0:
            ax.text(0.5, 0.5, "No contradictions detected", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=14)
            plt.axis('off')
        else:
            # Create position layout
            pos = nx.spring_layout(contradiction_graph, seed=42)
            
            # Create node colors based on agent
            node_colors = []
            for node in contradiction_graph.nodes():
                agent_id = contradiction_graph.nodes[node].get("agent_id")
                node_colors.append(self.color_map.get_agent_color(agent_id))
            
            # Draw nodes
            nx.draw_networkx_nodes(
                contradiction_graph, 
                pos, 
                node_color=node_colors,
                node_size=100,
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                contradiction_graph, 
                pos, 
                alpha=0.7,
                width=2,
                edge_color='red'
            )
            
            # Create labels
            labels = {}
            for node in contradiction_graph.nodes():
                label = contradiction_graph.nodes[node].get("label", "")
                # Truncate long labels
                if len(label) > 20:
                    label = label[:17] + "..."
                labels[node] = label
            
            nx.draw_networkx_labels(
                contradiction_graph, 
                pos, 
                labels=labels,
                font_size=8,
                font_family="sans-serif"
            )
            
            # Add legend for agent colors
            agent_handles = []
            for agent_id in sorted(agent_ids):
                color = self.color_map.get_agent_color(agent_id)
                agent_handles.append(plt.Line2D(
                    [0], [0], 
                    marker='o', 
                    color='w',
                    markerfacecolor=color, 
                    markersize=10, 
                    label=agent_id
                ))
            
            plt.legend(
                handles=agent_handles, 
                title="Agents",
                loc="upper right", 
                bbox_to_anchor=(1.1, 1)
            )
            
            plt.axis('off')
        
        # Add title
        plt.title("Multi-Agent Belief Contradictions")
        plt.tight_layout()
        
        # Save to file if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig


class CollectiveSchemaVisualizer:
    """
    Visualizes the collective schema and belief space of multiple agents.
    
    This class provides tools for creating visual representations of the
    merged schema of multiple agents, highlighting shared understandings.
    """
    
    def __init__(self, agent_schemas: Dict[str, SchemaGraph]):
        """
        Initialize with a dictionary of agent schemas.
        
        Args:
            agent_schemas: Dictionary mapping agent IDs to schema graphs
        """
        self.agent_schemas = agent_schemas
        self.color_map = AgentColorMap()
        self.shared_builder = SharedSchemaBuilder()
        
        # Add schemas to the shared builder
        for agent_id, schema in agent_schemas.items():
            self.shared_builder.add_agent_schema(agent_id, schema)
    
    def visualize_collective_schema(
        self,
        figsize: Tuple[int, int] = (16, 12),
        output_file: Optional[str] = None,
        highlight_overlaps: bool = True,
        layout_method: str = "spring"
    ) -> plt.Figure:
        """
        Visualize the collective schema of all agents.
        
        Args:
            figsize: Size of the figure
            output_file: Optional file path to save the visualization
            highlight_overlaps: Whether to highlight overlapping concepts
            layout_method: Layout algorithm to use
            
        Returns:
            The matplotlib figure object
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the shared graph
        shared_graph = self.shared_builder.visualize_shared_schema()
        
        # Create position layout
        if layout_method == "spring":
            pos = nx.spring_layout(shared_graph, seed=42)
        elif layout_method == "kamada_kawai":
            pos = nx.kamada_kawai_layout(shared_graph)
        elif layout_method == "circular":
            pos = nx.circular_layout(shared_graph)
        else:
            pos = nx.spring_layout(shared_graph, seed=42)
        
        # Find overlapping concepts
        overlap_nodes = self.shared_builder.find_overlap_nodes()
        
        # Create node colors and sizes based on agent and overlap status
        node_colors = []
        node_sizes = []
        
        for node in shared_graph.nodes():
            agent_id = node.split(':')[0]
            node_data = shared_graph.nodes[node]
            label = node_data.get("label", node.split(':')[1])
            
            # Determine if this is an overlap node
            is_overlap = label in overlap_nodes
            
            # Set color based on agent
            node_colors.append(self.color_map.get_agent_color(agent_id))
            
            # Set size - larger for overlap nodes
            node_sizes.append(200 if is_overlap and highlight_overlaps else 100)
        
        # Draw the network
        nx.draw_networkx_nodes(
            shared_graph, 
            pos, 
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw edges with colors based on source agent
        for u, v, data in shared_graph.edges(data=True):
            agent_id = u.split(':')[0]  # Use source node's agent
            color = self.color_map.get_agent_color(agent_id)
            nx.draw_networkx_edges(
                shared_graph, 
                pos, 
                edgelist=[(u, v)],
                alpha=0.5,
                edge_color=color
            )
        
        # Create labels
        labels = {}
        for node in shared_graph.nodes():
            node_data = shared_graph.nodes[node]
            label = node_data.get("label", node.split(':')[1])
            # Truncate long labels
            if len(label) > 20:
                label = label[:17] + "..."
            labels[node] = label
        
        nx.draw_networkx_labels(
            shared_graph, 
            pos, 
            labels=labels,
            font_size=8,
            font_family="sans-serif"
        )
        
        # Add legend for agent colors
        agent_handles = []
        for agent_id in sorted(self.agent_schemas.keys()):
            color = self.color_map.get_agent_color(agent_id)
            agent_handles.append(plt.Line2D(
                [0], [0], 
                marker='o', 
                color='w',
                markerfacecolor=color, 
                markersize=10, 
                label=agent_id
            ))
        
        # Add legend entry for overlap nodes if highlighting
        if highlight_overlaps:
            agent_handles.append(plt.Line2D(
                [0], [0], 
                marker='o', 
                color='w',
                markerfacecolor='gray', 
                markersize=15, 
                label='Overlapping Concept'
            ))
        
        plt.legend(
            handles=agent_handles, 
            title="Schema Legend",
            loc="upper right", 
            bbox_to_anchor=(1.1, 1)
        )
        
        # Add title
        plt.title("Collective Multi-Agent Schema")
        plt.axis('off')
        plt.tight_layout()
        
        # Save to file if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_shared_builder(self) -> SharedSchemaBuilder:
        """
        Get the shared schema builder.
        
        Returns:
            The shared schema builder
        """
        return self.shared_builder 