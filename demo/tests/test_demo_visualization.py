#!/usr/bin/env python3
"""
ΨC Demo Visualization Tests
---------------------------
Tests for the visualization components of the ΨC-AI SDK demonstration.
These tests validate that the visualization functions correctly generate
appropriate figures, metrics, and visual elements.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path for imports
script_dir = Path(__file__).parent.absolute()
demo_dir = script_dir.parent.absolute()
sys.path.append(str(demo_dir))
project_dir = demo_dir.parent.absolute()
sys.path.append(str(project_dir))

# Use non-interactive backend for testing
plt.switch_backend('Agg')

# Import visualization functions from web_interface
sys.path.append(str(demo_dir))
# We'll import specific functions below with try/except to handle Streamlit dependencies

class MockSessionState:
    """Mock class for Streamlit session state."""
    def __init__(self):
        self.coherence_history = [0.75, 0.8, 0.85, 0.83, 0.88]
        self.schema_complexity = [1, 3, 8, 12, 15]
        self.performance_history = [
            {'memory_time': 150.0, 'query_time': 200.0},
            {'memory_time': 148.5, 'query_time': 205.0},
            {'memory_time': 152.0, 'query_time': 198.0},
            {'memory_time': 145.0, 'query_time': 210.0},
        ]
        self.demo_runner = MagicMock()
        self.demo_runner.schema = MagicMock()
        self.demo_runner.schema.get_all_nodes.return_value = [f"node_{i}" for i in range(10)]
        self.demo_runner.schema.get_all_edges.return_value = [(f"node_{i}", f"node_{i+1}") for i in range(9)]
        self.demo_runner.get_schema_graph_data.return_value = {
            "nodes": [{"id": f"node_{i}", "label": f"Node {i}", "type": "concept", "importance": 0.5 + i/20} for i in range(10)],
            "edges": [{"source": f"node_{i}", "target": f"node_{i+1}", "weight": 0.5, "type": "relation"} for i in range(9)]
        }
        self.contradiction_rate = 0.15
        self.avg_memory_time = 150.0
        self.avg_query_time = 200.0
        self.delta_concepts = 2
        self.delta_connections = 3
        self.previous_schema_state = {
            "nodes": [{"id": f"node_{i}", "label": f"Node {i}", "type": "concept", "importance": 0.5 + i/20} for i in range(8)],
            "edges": [{"source": f"node_{i}", "target": f"node_{i+1}", "weight": 0.5, "type": "relation"} for i in range(7)]
        }
        self.last_action = "add_memory"
        self.detailed_log = [
            {
                "timestamp": "2023-07-01T12:00:00",
                "operation": "add_memory",
                "content": "Test memory",
                "processing_time_ms": 150.0,
                "before": {"coherence": 0.75, "node_count": 5, "edge_count": 4},
                "after": {"coherence": 0.8, "node_count": 7, "edge_count": 7},
                "delta": {"coherence": 0.05, "nodes": 2, "edges": 3},
                "contradictions": 0,
                "contradiction_rate": 0.0
            },
            {
                "timestamp": "2023-07-01T12:05:00",
                "operation": "process_query",
                "query": "Test query",
                "response": "Test response",
                "processing_time_ms": 200.0,
                "before": {"coherence": 0.8, "node_count": 7, "edge_count": 7},
                "after": {"coherence": 0.85, "node_count": 8, "edge_count": 8},
                "delta": {"coherence": 0.05, "nodes": 1, "edges": 1},
                "contradictions": 1,
                "contradiction_rate": 0.125
            }
        ]

class TestDemoVisualization(unittest.TestCase):
    """Test cases for demo visualization components."""
    
    def setUp(self):
        """Set up test environment with mocks."""
        self.mock_st = MagicMock()
        self.mock_session_state = MockSessionState()
        self.patches = []
        
        # Patch the streamlit module
        self.st_patch = patch.dict('sys.modules', {'streamlit': self.mock_st})
        self.st_patch.start()
        self.patches.append(self.st_patch)
        
        # Patch the session_state
        self.state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.state_patch.start()
        self.patches.append(self.state_patch)
        
        # Now import the visualization functions
        from web_interface_demo import create_schema_graph, create_schema_diff
        self.create_schema_graph = create_schema_graph
        self.create_schema_diff = create_schema_diff
    
    def tearDown(self):
        """Clean up patches."""
        for p in self.patches:
            p.stop()
    
    def test_create_schema_graph(self):
        """Test the schema graph visualization function."""
        # Test full graph
        fig = self.create_schema_graph(core_only=False)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test core-only graph
        fig = self.create_schema_graph(core_only=True)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test empty graph handling
        self.mock_session_state.demo_runner.get_schema_graph_data.return_value = {"nodes": [], "edges": []}
        fig = self.create_schema_graph()
        self.assertIsInstance(fig, plt.Figure)
    
    def test_create_schema_diff(self):
        """Test the schema diff visualization function."""
        # Test with changes
        fig = self.create_schema_diff()
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with no previous state
        self.mock_session_state.previous_schema_state = None
        fig = self.create_schema_diff()
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with identical states (no changes)
        self.mock_session_state.previous_schema_state = self.mock_session_state.demo_runner.get_schema_graph_data()
        fig = self.create_schema_diff()
        self.assertIsInstance(fig, plt.Figure)
    
    def test_render_metrics_data(self):
        """Test the metrics data generation for visualizations."""
        # Import relevant function
        from web_interface_demo import render_metrics
        
        # We can't directly test the rendering, but we can validate the data
        # that would be rendered by checking the mock calls to pyplot
        with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())) as mock_subplots:
            with patch('streamlit.pyplot') as mock_st_pyplot:
                # Call the function
                render_metrics()
                
                # Verify the appropriate data was used
                self.assertTrue(mock_subplots.called)
                self.assertTrue(mock_st_pyplot.called)

class TestGraphDataGeneration(unittest.TestCase):
    """Test cases for graph data generation and transformation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simple demo schema graph
        self.G = nx.Graph()
        for i in range(10):
            self.G.add_node(f"node_{i}", label=f"Node {i}", type="concept", importance=0.5 + i/20)
        for i in range(9):
            self.G.add_edge(f"node_{i}", f"node_{i+1}", weight=0.5, type="relation")
        
        # Convert to schema data format
        self.schema_data = {
            "nodes": [{"id": node, "label": data["label"], "type": data["type"], "importance": data["importance"]} 
                     for node, data in self.G.nodes(data=True)],
            "edges": [{"source": u, "target": v, "weight": data["weight"], "type": data["type"]} 
                     for u, v, data in self.G.edges(data=True)]
        }
    
    def test_graph_construction(self):
        """Test construction of NetworkX graph from schema data."""
        # This test verifies we can properly construct a NetworkX graph from our schema data format
        G = nx.Graph()
        
        # Add nodes with attributes
        for node in self.schema_data["nodes"]:
            G.add_node(
                node["id"], 
                label=node["label"], 
                type=node["type"],
                importance=node["importance"]
            )
        
        # Add edges with attributes
        for edge in self.schema_data["edges"]:
            G.add_edge(
                edge["source"], 
                edge["target"], 
                weight=edge["weight"],
                type=edge["type"]
            )
        
        # Verify the graph matches the original
        self.assertEqual(len(G.nodes), len(self.G.nodes))
        self.assertEqual(len(G.edges), len(self.G.edges))
        
        # Check node attributes
        for node, data in G.nodes(data=True):
            self.assertEqual(data["label"], self.G.nodes[node]["label"])
            self.assertEqual(data["type"], self.G.nodes[node]["type"])
            self.assertEqual(data["importance"], self.G.nodes[node]["importance"])
        
        # Check edge attributes
        for u, v, data in G.edges(data=True):
            self.assertEqual(data["weight"], self.G.edges[u, v]["weight"])
            self.assertEqual(data["type"], self.G.edges[u, v]["type"])
    
    def test_graph_difference(self):
        """Test computing the difference between two graph states."""
        # Create a modified version of the schema with new nodes and edges
        modified_schema = {
            "nodes": self.schema_data["nodes"] + [
                {"id": "node_10", "label": "Node 10", "type": "memory", "importance": 0.7},
                {"id": "node_11", "label": "Node 11", "type": "concept", "importance": 0.6}
            ],
            "edges": self.schema_data["edges"] + [
                {"source": "node_9", "target": "node_10", "weight": 0.5, "type": "relation"},
                {"source": "node_10", "target": "node_11", "weight": 0.6, "type": "relation"}
            ]
        }
        
        # Compute differences
        previous_nodes = {node["id"]: node for node in self.schema_data["nodes"]}
        current_nodes = {node["id"]: node for node in modified_schema["nodes"]}
        
        new_node_ids = set(current_nodes.keys()) - set(previous_nodes.keys())
        
        # Validate the differences
        self.assertEqual(len(new_node_ids), 2)
        self.assertTrue("node_10" in new_node_ids)
        self.assertTrue("node_11" in new_node_ids)
        
        # Check for new edges
        previous_edges = {(edge["source"], edge["target"]): edge for edge in self.schema_data["edges"]}
        current_edges = {(edge["source"], edge["target"]): edge for edge in modified_schema["edges"]}
        
        new_edge_pairs = set(current_edges.keys()) - set(previous_edges.keys())
        
        self.assertEqual(len(new_edge_pairs), 2)
        self.assertTrue(("node_9", "node_10") in new_edge_pairs)
        self.assertTrue(("node_10", "node_11") in new_edge_pairs)

if __name__ == '__main__':
    unittest.main() 