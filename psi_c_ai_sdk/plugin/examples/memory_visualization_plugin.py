"""
Memory Visualization Plugin for ΨC-AI SDK

This module provides a plugin for visualizing memory objects, their relationships,
and coherence metrics in the ΨC-AI SDK.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from psi_c_ai_sdk.plugin.base import (
    PluginBase,
    PluginInfo,
    PluginHook,
    PluginType,
    create_plugin_id
)
from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.schema.schema import SchemaGraph


class MemoryVisualizationPlugin(PluginBase):
    """
    Plugin for visualizing memory objects and their relationships.
    
    This plugin provides visualization capabilities for memory networks,
    schema graphs, contradiction heatmaps, and other visual analytics.
    """
    
    @classmethod
    def _get_plugin_info(cls) -> PluginInfo:
        """Get metadata about the plugin."""
        return PluginInfo(
            id=create_plugin_id("memory_visualization", "psi_c_example"),
            name="Memory Visualization Plugin",
            version="0.1.0",
            description="A plugin for visualizing memory objects and their relationships",
            author="ΨC-AI SDK Team",
            plugin_type=PluginType.VISUALIZATION,
            hooks={
                PluginHook.POST_REFLECTION,
                PluginHook.POST_SCHEMA_UPDATE
            },
            tags=["visualization", "memory", "schema", "example"]
        )
    
    def _register_hooks(self) -> Dict[PluginHook, Any]:
        """Register the hooks that this plugin implements."""
        return {
            PluginHook.POST_REFLECTION: self.post_reflection_handler,
            PluginHook.POST_SCHEMA_UPDATE: self.post_schema_update_handler
        }
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        self.logger.info("Initializing Memory Visualization Plugin")
        
        # Set default config if not provided
        if not self.config:
            self.config = {
                "output_dir": "visualizations",
                "min_coherence": 0.3,
                "show_plots": False,
                "file_format": "png",
                "max_nodes": 50,
                "color_scheme": "default"
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Track statistics
        self.stats = {
            "visualizations_created": 0,
            "files_saved": 0
        }
        
        self.logger.info("Memory Visualization Plugin initialized")
        return True
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.logger.info(
            f"Memory Visualization Plugin shutdown. Stats: {self.stats}"
        )
    
    def post_reflection_handler(
        self,
        result: Dict[str, Any],
        memories: Optional[List[Memory]] = None,
        schema: Optional[SchemaGraph] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-reflection events.
        
        This method is called after the reflection cycle completes
        and generates visualizations based on the results.
        
        Args:
            result: Results from the reflection cycle
            memories: Memories that were reflected upon
            schema: Updated schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        self.logger.info("Post-reflection: Generating visualizations")
        
        visualization_results = {}
        
        # Generate memory network visualization if memories are provided
        if memories:
            memory_network_path = self._visualize_memory_network(memories)
            visualization_results["memory_network"] = memory_network_path
        
        # Generate contradiction heatmap if enough memories are available
        if memories and len(memories) > 1:
            heatmap_path = self._create_contradiction_heatmap(memories)
            visualization_results["contradiction_heatmap"] = heatmap_path
        
        # Generate schema visualization if schema is provided
        if schema:
            schema_viz_path = self._visualize_schema(schema)
            visualization_results["schema_graph"] = schema_viz_path
        
        # Update stats
        self.stats["visualizations_created"] += len(visualization_results)
        
        return {
            "visualization_paths": visualization_results,
            "plugin_visualization_count": self.stats["visualizations_created"]
        }
    
    def post_schema_update_handler(
        self, 
        schema: SchemaGraph,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-schema-update events.
        
        This method is called after the schema graph is updated
        and generates a visualization of the updated schema.
        
        Args:
            schema: Updated schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        self.logger.info("Post-schema-update: Generating schema visualization")
        
        # Generate schema visualization
        schema_viz_path = self._visualize_schema(schema)
        
        # Update stats
        self.stats["visualizations_created"] += 1
        
        return {
            "schema_visualization": schema_viz_path,
            "plugin_visualization_count": self.stats["visualizations_created"]
        }
    
    def _visualize_memory_network(self, memories: List[Memory]) -> str:
        """
        Visualize memory objects as a network with coherence-weighted edges.
        
        Args:
            memories: List of memory objects to visualize
            
        Returns:
            Path to the saved visualization file
        """
        self.logger.info(f"Visualizing memory network with {len(memories)} memories")
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for each memory
        for i, memory in enumerate(memories):
            G.add_node(
                i, 
                label=self._get_short_content(memory),
                importance=getattr(memory, "importance", 0.5),
                memory_type=getattr(memory, "memory_type", "unknown")
            )
        
        # Add edges based on coherence (simulated here for demonstration)
        # In a real implementation, you would use actual coherence scores
        min_coherence = self.config.get("min_coherence", 0.3)
        
        for i in range(len(memories)):
            for j in range(i+1, len(memories)):
                # Simulate coherence based on embedding similarity
                # In real use, you would calculate actual coherence
                coherence = self._simulate_coherence(memories[i], memories[j])
                if coherence >= min_coherence:
                    G.add_edge(i, j, weight=coherence)
        
        # Limit graph size if needed
        max_nodes = self.config.get("max_nodes", 50)
        if len(G) > max_nodes:
            # Keep only the most important nodes
            importance_dict = nx.get_node_attributes(G, 'importance')
            nodes_to_keep = sorted(
                importance_dict.keys(), 
                key=lambda k: importance_dict[k], 
                reverse=True
            )[:max_nodes]
            G = G.subgraph(nodes_to_keep).copy()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Node positions using force-directed layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get node attributes for visualization
        node_importance = [G.nodes[n].get('importance', 0.5) * 500 for n in G.nodes()]
        
        # Get memory types for coloring
        memory_types = [G.nodes[n].get('memory_type', 'unknown') for n in G.nodes()]
        color_map = {
            'episodic': 'skyblue',
            'semantic': 'lightgreen',
            'procedural': 'orange',
            'unknown': 'gray'
        }
        node_colors = [color_map.get(t, 'gray') for t in memory_types]
        
        # Get edge weights for line thickness
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        
        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=node_importance,
            node_color=node_colors,
            alpha=0.8
        )
        nx.draw_networkx_edges(
            G, pos,
            width=edge_weights,
            alpha=0.5,
            edge_color='gray'
        )
        
        # Draw node labels
        labels = {n: G.nodes[n].get('label', str(n)) for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_family='sans-serif'
        )
        
        plt.title("Memory Network - Coherence Graph")
        plt.axis('off')
        
        # Save to file
        filename = f"{self.config['output_dir']}/memory_network_{self._get_timestamp()}.{self.config['file_format']}"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        # Show plot if configured
        if self.config.get("show_plots", False):
            plt.show()
        else:
            plt.close()
        
        self.stats["files_saved"] += 1
        return filename
    
    def _create_contradiction_heatmap(self, memories: List[Memory]) -> str:
        """
        Create a heatmap visualizing contradictions between memories.
        
        Args:
            memories: List of memory objects to analyze
            
        Returns:
            Path to the saved visualization file
        """
        self.logger.info(f"Creating contradiction heatmap for {len(memories)} memories")
        
        # Generate a contradiction matrix (simulated for demonstration)
        # In a real implementation, you would use actual contradiction detection
        n = len(memories)
        contradiction_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Simulate contradiction detection
                    # In real use, you would use the actual contradiction detection algorithm
                    # Using formula 11 from UnderlyingMath.md: Contradiction Heatmap Matrix
                    contradiction_matrix[i, j] = self._simulate_contradiction(memories[i], memories[j])
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        plt.imshow(contradiction_matrix, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Contradiction Score')
        
        # Labels
        plt.title("Memory Contradiction Heatmap")
        plt.xlabel("Memory Index")
        plt.ylabel("Memory Index")
        
        # Add ticks and labels
        plt.xticks(range(n), range(n))
        plt.yticks(range(n), range(n))
        
        # Save to file
        filename = f"{self.config['output_dir']}/contradiction_heatmap_{self._get_timestamp()}.{self.config['file_format']}"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        # Show plot if configured
        if self.config.get("show_plots", False):
            plt.show()
        else:
            plt.close()
        
        self.stats["files_saved"] += 1
        return filename
    
    def _visualize_schema(self, schema: SchemaGraph) -> str:
        """
        Visualize a schema graph.
        
        Args:
            schema: Schema graph to visualize
            
        Returns:
            Path to the saved visualization file
        """
        self.logger.info("Visualizing schema graph")
        
        # Extract the graph from the schema
        # In a real implementation, you would use the actual schema graph structure
        G = schema.get_graph() if hasattr(schema, 'get_graph') else nx.Graph()
        
        # If the graph is empty or not available, create a placeholder
        if not G or len(G.nodes()) == 0:
            self.logger.warning("Schema graph is empty, creating placeholder")
            G = nx.Graph()
            G.add_node("placeholder", label="Empty Schema")
        
        # Limit graph size if needed
        max_nodes = self.config.get("max_nodes", 50)
        if len(G) > max_nodes:
            # Keep only the most central nodes by betweenness centrality
            centrality = nx.betweenness_centrality(G)
            nodes_to_keep = sorted(
                centrality.keys(), 
                key=lambda k: centrality[k], 
                reverse=True
            )[:max_nodes]
            G = G.subgraph(nodes_to_keep).copy()
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Node positions using force-directed layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get node attributes for visualization (if available)
        node_size = []
        node_color = []
        
        for node in G.nodes():
            # Get node importance or use degree as fallback
            importance = G.nodes[node].get('importance', G.degree(node) / max(1, G.number_of_nodes()))
            node_size.append(300 * importance + 100)
            
            # Get node type for coloring
            node_type = G.nodes[node].get('type', 'unknown')
            if node_type == 'concept':
                node_color.append('lightblue')
            elif node_type == 'memory':
                node_color.append('lightgreen')
            else:
                node_color.append('lightgray')
        
        # Get edge weights for line thickness
        edge_weights = []
        for u, v in G.edges():
            # Get coherence weight or use default
            weight = G.edges[u, v].get('weight', 0.5)
            edge_weights.append(weight * 3)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=node_size,
            node_color=node_color,
            alpha=0.8
        )
        nx.draw_networkx_edges(
            G, pos,
            width=edge_weights,
            alpha=0.5,
            edge_color='gray'
        )
        
        # Draw node labels
        labels = {}
        for node in G.nodes():
            # Get node label or use node id as fallback
            label = G.nodes[node].get('label', str(node))
            # Truncate long labels
            if isinstance(label, str) and len(label) > 20:
                label = label[:17] + "..."
            labels[node] = label
            
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_family='sans-serif'
        )
        
        plt.title("Schema Graph Visualization")
        plt.axis('off')
        
        # Save to file
        filename = f"{self.config['output_dir']}/schema_graph_{self._get_timestamp()}.{self.config['file_format']}"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        # Show plot if configured
        if self.config.get("show_plots", False):
            plt.show()
        else:
            plt.close()
        
        self.stats["files_saved"] += 1
        return filename
    
    def _get_short_content(self, memory: Memory) -> str:
        """Get a short representation of memory content."""
        content = getattr(memory, "content", str(memory))
        if isinstance(content, str) and len(content) > 30:
            return content[:27] + "..."
        return str(content)
    
    def _simulate_coherence(self, memory1: Memory, memory2: Memory) -> float:
        """
        Simulate coherence between two memories (for demonstration).
        
        In a real implementation, you would use the actual coherence calculation
        from the system, using formula 2 from UnderlyingMath.md:
        C(A, B) = cosine(v_A, v_B) + λ · tag_overlap(A, B)
        """
        # This is just a simple simulation - in reality, you would use actual embeddings
        # Get importance values or use defaults
        imp1 = getattr(memory1, "importance", 0.5)
        imp2 = getattr(memory2, "importance", 0.5)
        
        # Get memory types
        type1 = getattr(memory1, "memory_type", "unknown")
        type2 = getattr(memory2, "memory_type", "unknown")
        
        # Higher coherence for memories of the same type
        type_factor = 0.3 if type1 == type2 else 0.1
        
        # Generate a base coherence score with some randomness
        base_coherence = 0.2 + 0.3 * (imp1 + imp2) / 2.0 + type_factor
        
        # Add some randomness
        import random
        random_factor = random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_coherence + random_factor))
    
    def _simulate_contradiction(self, memory1: Memory, memory2: Memory) -> float:
        """
        Simulate contradiction detection between two memories (for demonstration).
        
        In a real implementation, you would use the actual contradiction detection
        algorithm from the system, based on formula 3 from UnderlyingMath.md:
        Contradict(A, B) = 1 if ∃k ∈ K: k ∈ A ∧ semantic_match(B) else 0
        """
        # This is just a simple simulation for demonstration purposes
        # In reality, you would implement proper contradiction detection
        
        # Get memory types
        type1 = getattr(memory1, "memory_type", "unknown")
        type2 = getattr(memory2, "memory_type", "unknown")
        
        # Simulate higher chance of contradiction for different memory types
        if type1 != type2:
            base_probability = 0.15
        else:
            base_probability = 0.05
        
        # Add some randomness
        import random
        random_factor = random.uniform(0, 0.2)
        
        # Return binary contradiction (0 or 1) with probability
        if random.random() < base_probability + random_factor:
            return 1.0
        return 0.0
    
    def _get_timestamp(self) -> str:
        """Generate a timestamp for filenames."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# This allows the plugin to be loaded directly
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the plugin
    plugin = MemoryVisualizationPlugin()
    plugin.initialize()
    
    # Create some test memory objects
    test_memories = [
        Memory(id="m1", content="This is memory 1", importance=0.8, memory_type="episodic"),
        Memory(id="m2", content="This is memory 2", importance=0.6, memory_type="semantic"),
        Memory(id="m3", content="This is memory 3", importance=0.7, memory_type="procedural"),
        Memory(id="m4", content="This is memory 4", importance=0.5, memory_type="episodic"),
        Memory(id="m5", content="This is memory 5", importance=0.9, memory_type="semantic")
    ]
    
    # Create a simple schema graph for testing
    test_schema = SchemaGraph()
    
    # Test the visualization methods
    plugin.post_reflection_handler(
        result={"coherence_before": 0.5, "coherence_after": 0.7},
        memories=test_memories,
        schema=test_schema
    )
    
    plugin.shutdown() 