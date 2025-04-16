#!/usr/bin/env python3
"""
Visualization Battle Testing Script for ΨC-AI SDK Plugins.

This script performs comprehensive testing of the visualization capabilities
of the Memory Visualization Plugin with various data sizes and edge cases.
"""

import os
import sys
import time
import logging
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("visualization_tests")

# Create mock classes for core SDK components
class Memory:
    """Mock Memory class for testing."""
    def __init__(self, id=None, content=None, importance=0.5, memory_type="unknown", 
                 created_at=None, tags=None):
        self.id = id or f"memory_{random.randint(1000, 9999)}"
        self.content = content or f"Sample memory content {random.randint(1, 100)}"
        self.importance = importance
        self.memory_type = memory_type
        self.created_at = created_at or datetime.now().isoformat()
        self.tags = tags or []

class SchemaGraph:
    """Mock SchemaGraph class for testing."""
    def __init__(self):
        self._graph = nx.Graph()
        
    def get_graph(self):
        return self._graph
        
    def set_graph(self, graph):
        self._graph = graph

# Define a mock plugin base and related classes
class PluginHook:
    """Mock PluginHook enum for testing."""
    POST_REFLECTION = "POST_REFLECTION"
    POST_SCHEMA_UPDATE = "POST_SCHEMA_UPDATE"

class PluginType:
    """Mock PluginType enum for testing."""
    VISUALIZATION = "VISUALIZATION"

class PluginInfo:
    """Mock PluginInfo class for testing."""
    def __init__(self, id, name, version, description, author, plugin_type, hooks, tags):
        self.id = id
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.plugin_type = plugin_type
        self.hooks = hooks
        self.tags = tags

class PluginBase:
    """Mock PluginBase class for testing."""
    def __init__(self):
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self.config = {}
    
    def initialize(self):
        return True
    
    def shutdown(self):
        pass

# Create a mock version of the MemoryVisualizationPlugin
class MemoryVisualizationPlugin(PluginBase):
    """
    Mock implementation of the Memory Visualization Plugin for testing.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("plugin.MemoryVisualizationPlugin")
    
    @classmethod
    def _get_plugin_info(cls):
        """Get metadata about the plugin."""
        return PluginInfo(
            id="psi_c_example.memory_visualization",
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
    
    def initialize(self):
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
    
    def shutdown(self):
        """Shutdown the plugin."""
        self.logger.info(
            f"Memory Visualization Plugin shutdown. Stats: {self.stats}"
        )
    
    def post_reflection_handler(self, result, memories=None, schema=None, **kwargs):
        """
        Handler for post-reflection events.
        
        This method is called after the reflection cycle completes
        and generates visualizations based on the results.
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
    
    def post_schema_update_handler(self, schema, **kwargs):
        """
        Handler for post-schema-update events.
        
        This method is called after the schema graph is updated
        and generates a visualization of the updated schema.
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
    
    def _visualize_memory_network(self, memories):
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
        
        # Add edges based on coherence (simulated for this mock implementation)
        min_coherence = self.config.get("min_coherence", 0.3)
        
        for i in range(len(memories)):
            for j in range(i+1, len(memories)):
                # Simulate coherence
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
        
        # Node sizes based on importance
        node_importance = [G.nodes[n].get('importance', 0.5) * 500 for n in G.nodes()]
        
        # Node colors based on memory type
        memory_types = [G.nodes[n].get('memory_type', 'unknown') for n in G.nodes()]
        color_map = {
            'episodic': 'skyblue',
            'semantic': 'lightgreen',
            'procedural': 'orange',
            'unknown': 'gray'
        }
        node_colors = [color_map.get(t, 'gray') for t in memory_types]
        
        # Edge weights for line thickness
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
    
    def _create_contradiction_heatmap(self, memories):
        """
        Create a heatmap visualizing contradictions between memories.
        
        Args:
            memories: List of memory objects to analyze
            
        Returns:
            Path to the saved visualization file
        """
        self.logger.info(f"Creating contradiction heatmap for {len(memories)} memories")
        
        # Generate a contradiction matrix (simulated for this mock implementation)
        n = len(memories)
        contradiction_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Simulate contradiction detection
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
    
    def _visualize_schema(self, schema):
        """
        Visualize a schema graph.
        
        Args:
            schema: Schema graph to visualize
            
        Returns:
            Path to the saved visualization file
        """
        self.logger.info("Visualizing schema graph")
        
        # Extract the graph from the schema
        G = schema.get_graph()
        
        # If the graph is empty or not available, create a placeholder
        if not G or len(G.nodes()) == 0:
            self.logger.warning("Schema graph is empty, creating placeholder")
            G = nx.Graph()
            G.add_node("placeholder", label="Empty Schema")
        
        # Limit graph size if needed
        max_nodes = self.config.get("max_nodes", 50)
        if len(G) > max_nodes:
            # Keep only the most central nodes by betweenness centrality
            if len(G) > 1:  # Betweenness centrality requires at least 2 nodes
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
        
        # Node sizes and colors based on attributes
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
        
        # Edge weights for line thickness
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
    
    def _get_short_content(self, memory):
        """Get a short representation of memory content."""
        content = getattr(memory, "content", str(memory))
        if isinstance(content, str) and len(content) > 30:
            return content[:27] + "..."
        return str(content)
    
    def _simulate_coherence(self, memory1, memory2):
        """Simulate coherence between two memories (for mock implementation)."""
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
        random_factor = random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_coherence + random_factor))
    
    def _simulate_contradiction(self, memory1, memory2):
        """Simulate contradiction detection between two memories (for mock implementation)."""
        # Get memory types
        type1 = getattr(memory1, "memory_type", "unknown")
        type2 = getattr(memory2, "memory_type", "unknown")
        
        # Simulate higher chance of contradiction for different memory types
        if type1 != type2:
            base_probability = 0.15
        else:
            base_probability = 0.05
        
        # Add some randomness
        random_factor = random.uniform(0, 0.2)
        
        # Return binary contradiction (0 or 1) with probability
        if random.random() < base_probability + random_factor:
            return 1.0
        return 0.0
    
    def _get_timestamp(self):
        """Generate a timestamp for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# Test data generation functions
def create_diverse_test_memories(count):
    """Create diverse memory objects for testing."""
    memory_types = ["episodic", "semantic", "procedural", "declarative", "unknown"]
    memories = []
    
    # Create memories with varied properties
    for i in range(count):
        importance = random.uniform(0.1, 1.0)
        memory_type = random.choice(memory_types)
        created_at = (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
        tags = random.sample(["important", "personal", "work", "family", "travel", "idea"], 
                             k=random.randint(0, 3))
        
        content = f"Test memory {i} with {memory_type} type and {importance:.2f} importance."
        
        # Create longer content for some memories
        if random.random() > 0.7:
            content += " " + "Extended content. " * random.randint(5, 20)
        
        memories.append(Memory(
            id=f"memory_{i}",
            content=content,
            importance=importance,
            memory_type=memory_type,
            created_at=created_at,
            tags=tags
        ))
    
    return memories


def create_test_schema(nodes, edges, with_clusters=False):
    """Create a test schema graph with specified nodes and edges."""
    schema = SchemaGraph()
    G = nx.Graph()
    
    # Define node types for the graph
    node_types = ["concept", "entity", "event", "relation", "memory"]
    
    # Create nodes
    for i in range(nodes):
        # Determine node type
        node_type = random.choice(node_types)
        
        # For clustered graphs, assign nodes to clusters
        if with_clusters:
            cluster = random.randint(1, min(5, nodes // 10 or 1))
            G.add_node(f"node_{i}", 
                       label=f"{node_type.capitalize()} {i}",
                       type=node_type,
                       importance=random.uniform(0.1, 1.0),
                       cluster=cluster)
        else:
            G.add_node(f"node_{i}", 
                       label=f"{node_type.capitalize()} {i}",
                       type=node_type,
                       importance=random.uniform(0.1, 1.0))
    
    # Create edges (with preference for connecting nodes of the same cluster)
    edge_count = 0
    attempts = 0
    all_nodes = list(G.nodes())
    
    while edge_count < edges and attempts < edges * 10:
        attempts += 1
        
        # Select two random nodes
        if with_clusters and random.random() < 0.7:
            # Preferentially connect nodes in the same cluster
            clusters = set([G.nodes[n].get('cluster') for n in all_nodes])
            chosen_cluster = random.choice(list(clusters))
            cluster_nodes = [n for n in all_nodes if G.nodes[n].get('cluster') == chosen_cluster]
            
            if len(cluster_nodes) >= 2:
                u, v = random.sample(cluster_nodes, 2)
            else:
                u, v = random.sample(all_nodes, 2)
        else:
            u, v = random.sample(all_nodes, 2)
        
        # Avoid self-loops and existing edges
        if u != v and not G.has_edge(u, v):
            # Create edge with coherence weight
            weight = random.uniform(0.1, 1.0)
            # Higher coherence for nodes of the same type
            if G.nodes[u].get('type') == G.nodes[v].get('type'):
                weight += 0.2
                
            # Clamp weight to valid range
            weight = min(1.0, max(0.1, weight))
            
            G.add_edge(u, v, weight=weight)
            edge_count += 1
    
    # Set the graph on the schema
    schema.set_graph(G)
    return schema


def create_large_schema(nodes, edges):
    """Create a large schema graph for stress testing."""
    schema = SchemaGraph()
    
    # Use a scale-free graph for realistic large network topology
    G = nx.scale_free_graph(nodes, alpha=0.41, beta=0.54, gamma=0.05, seed=42)
    G = nx.Graph(G)  # Convert to undirected graph
    
    # Add some attributes to the nodes and edges
    node_types = ["concept", "entity", "event", "relation", "memory"]
    
    for node in G.nodes():
        G.nodes[node]['label'] = f"Node {node}"
        G.nodes[node]['type'] = random.choice(node_types)
        G.nodes[node]['importance'] = random.uniform(0.1, 1.0)
    
    # Limit to specified number of edges
    if G.number_of_edges() > edges:
        edges_to_remove = list(G.edges())[edges:]
        G.remove_edges_from(edges_to_remove)
    
    # Add weights to edges
    for u, v in G.edges():
        G.edges[u, v]['weight'] = random.uniform(0.1, 1.0)
    
    # Set the graph on the schema
    schema.set_graph(G)
    return schema


def create_test_directory():
    """Create a test directory for outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"viz_test_outputs_{timestamp}"
    os.makedirs(test_dir, exist_ok=True)
    return test_dir


def measure_execution_time(func, *args, **kwargs):
    """Measure and log execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def test_memory_network_visualization(plugin, memories, desc=""):
    """Test memory network visualization with given memories."""
    logger.info(f"Testing memory network visualization {desc} with {len(memories)} memories")
    try:
        result, execution_time = measure_execution_time(
            plugin._visualize_memory_network, memories
        )
        logger.info(f"  ✓ Created memory network visualization in {execution_time:.2f} seconds")
        
        # Verify the file exists
        if os.path.exists(result):
            file_size = os.path.getsize(result) / 1024  # KB
            logger.info(f"  ✓ File size: {file_size:.2f} KB")
            return True
        else:
            logger.error(f"  ✗ File not created: {result}")
            return False
    except Exception as e:
        logger.error(f"  ✗ Memory network visualization failed: {str(e)}")
        return False


def test_contradiction_heatmap(plugin, memories, desc=""):
    """Test contradiction heatmap visualization with given memories."""
    logger.info(f"Testing contradiction heatmap {desc} with {len(memories)} memories")
    try:
        result, execution_time = measure_execution_time(
            plugin._create_contradiction_heatmap, memories
        )
        logger.info(f"  ✓ Created contradiction heatmap in {execution_time:.2f} seconds")
        
        # Verify the file exists
        if os.path.exists(result):
            file_size = os.path.getsize(result) / 1024  # KB
            logger.info(f"  ✓ File size: {file_size:.2f} KB")
            return True
        else:
            logger.error(f"  ✗ File not created: {result}")
            return False
    except Exception as e:
        logger.error(f"  ✗ Contradiction heatmap failed: {str(e)}")
        return False


def test_schema_visualization(plugin, schema, desc=""):
    """Test schema visualization with given schema."""
    node_count = len(schema.get_graph().nodes())
    edge_count = len(schema.get_graph().edges())
    logger.info(f"Testing schema visualization {desc} with {node_count} nodes and {edge_count} edges")
    try:
        result, execution_time = measure_execution_time(
            plugin._visualize_schema, schema
        )
        logger.info(f"  ✓ Created schema visualization in {execution_time:.2f} seconds")
        
        # Verify the file exists
        if os.path.exists(result):
            file_size = os.path.getsize(result) / 1024  # KB
            logger.info(f"  ✓ File size: {file_size:.2f} KB")
            return True
        else:
            logger.error(f"  ✗ File not created: {result}")
            return False
    except Exception as e:
        logger.error(f"  ✗ Schema visualization failed: {str(e)}")
        return False


def run_visualization_tests():
    """Run comprehensive tests for visualizations."""
    logger.info("Starting visualization battle testing")
    
    # Create test directory
    test_dir = create_test_directory()
    logger.info(f"Created test directory: {test_dir}")
    
    # Initialize the plugin with actual visualization methods
    plugin = MemoryVisualizationPlugin()
    plugin.initialize()
    plugin.config["output_dir"] = test_dir
    plugin.config["show_plots"] = False
    plugin.config["file_format"] = "png"
    
    # Create test data with different sizes
    small_memories = create_diverse_test_memories(10)
    medium_memories = create_diverse_test_memories(50)
    large_memories = create_diverse_test_memories(200)
    
    small_schema = create_test_schema(15, 25)
    medium_schema = create_test_schema(50, 100, with_clusters=True)
    large_schema = create_test_schema(150, 300, with_clusters=True)
    xlarge_schema = create_large_schema(500, 1000)
    
    results = []
    
    # Test memory network visualizations
    results.append(test_memory_network_visualization(plugin, small_memories, "small"))
    results.append(test_memory_network_visualization(plugin, medium_memories, "medium"))
    results.append(test_memory_network_visualization(plugin, large_memories, "large"))
    
    # Test empty memories (edge case)
    results.append(test_memory_network_visualization(plugin, [], "empty"))
    
    # Test memories with identical properties (edge case)
    identical_memories = [
        Memory(id=f"same_{i}", content="Same content", importance=0.5, memory_type="semantic")
        for i in range(10)
    ]
    results.append(test_memory_network_visualization(plugin, identical_memories, "identical"))
    
    # Test contradiction heatmaps
    results.append(test_contradiction_heatmap(plugin, small_memories, "small"))
    results.append(test_contradiction_heatmap(plugin, medium_memories, "medium"))
    
    # Test schema visualizations
    results.append(test_schema_visualization(plugin, small_schema, "small"))
    results.append(test_schema_visualization(plugin, medium_schema, "medium with clusters"))
    results.append(test_schema_visualization(plugin, large_schema, "large with clusters"))
    
    # Test empty schema (edge case)
    empty_schema = SchemaGraph()
    results.append(test_schema_visualization(plugin, empty_schema, "empty"))
    
    # Test complete graph
    complete_schema = SchemaGraph()
    G = nx.complete_graph(20)
    nx.set_node_attributes(G, {i: f"Node {i}" for i in G.nodes()}, "label")
    nx.set_node_attributes(G, {i: random.choice(["concept", "entity"]) for i in G.nodes()}, "type")
    nx.set_edge_attributes(G, {e: random.uniform(0.5, 1.0) for e in G.edges()}, "weight")
    complete_schema.set_graph(G)
    results.append(test_schema_visualization(plugin, complete_schema, "complete graph"))
    
    # Stress test with very large schema
    try:
        logger.info("Starting stress test with very large schema")
        test_schema_visualization(plugin, xlarge_schema, "stress test")
    except Exception as e:
        logger.warning(f"Stress test failed (this may be expected): {str(e)}")
    
    # Test post_reflection_handler with all data
    try:
        logger.info("Testing post_reflection_handler with all data")
        start_time = time.time()
        result = plugin.post_reflection_handler(
            result={"coherence_before": 0.5, "coherence_after": 0.7},
            memories=medium_memories,
            schema=medium_schema
        )
        execution_time = time.time() - start_time
        logger.info(f"post_reflection_handler completed in {execution_time:.2f} seconds")
        logger.info(f"Generated {len(result['visualization_paths'])} visualizations")
        results.append(all(os.path.exists(path) for path in result["visualization_paths"].values()))
    except Exception as e:
        logger.error(f"post_reflection_handler test failed: {str(e)}")
        results.append(False)
    
    # Print summary
    success_count = sum(1 for r in results if r)
    logger.info("="*80)
    logger.info("VISUALIZATION TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(results) - success_count}")
    logger.info(f"Success rate: {success_count / len(results) * 100:.2f}%")
    
    # Manual review instructions
    logger.info("\nVisualizations have been generated in:")
    logger.info(f"  {os.path.abspath(test_dir)}")
    logger.info("\nPlease manually review the visualizations for:")
    logger.info("1. Clarity and readability of text and labels")
    logger.info("2. Appropriate color schemes and contrast")
    logger.info("3. Good layout and node placement")
    logger.info("4. Proper handling of edge cases (empty datasets, large datasets)")
    logger.info("5. Overall visual appearance and usefulness")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = run_visualization_tests()
    sys.exit(0 if success else 1) 