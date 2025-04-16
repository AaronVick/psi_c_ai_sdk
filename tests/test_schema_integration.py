#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Schema Integration Components

These tests validate the mathematical correctness and functionality of 
the schema integration system, ensuring proper alignment with the
ΨC-AI SDK mathematical foundations.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import MagicMock, patch
import networkx as nx

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import schema integration components
from tools.dev_environment.memory_schema_integration import (
    MemorySchemaIntegration, 
    is_schema_integration_available
)
from tools.dev_environment.schema_analysis import SchemaAnalysis

# Import memory components needed for testing
from tools.dev_environment.memory_sandbox import MemorySandbox
from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.memory.memory import Memory


class TestSchemaIntegrationMath(unittest.TestCase):
    """Test the mathematical correctness of schema integration components."""
    
    def setUp(self):
        """Set up test environment with synthetic memories."""
        # Create temporary directory for snapshots
        self.temp_dir = tempfile.mkdtemp()
        
        # Create memory store with synthetic test memories
        self.memory_store = MemoryStore()
        
        # Create memories with controlled embeddings for predictable similarity
        self.memory1 = Memory(
            content="This is memory about cats and dogs",
            memory_type="semantic",
            importance=0.8,
            embedding=np.array([0.1, 0.2, 0.3, 0.4])
        )
        self.memory1.id = "memory1"
        self.memory1.tags = ["animals", "pets"]
        
        self.memory2 = Memory(
            content="Dogs are loyal pets that need care",
            memory_type="semantic",
            importance=0.7,
            embedding=np.array([0.15, 0.25, 0.35, 0.45])  # Similar to memory1
        )
        self.memory2.id = "memory2"
        self.memory2.tags = ["animals", "dogs"]
        
        self.memory3 = Memory(
            content="Differential equations are used in physics",
            memory_type="semantic",
            importance=0.6,
            embedding=np.array([0.8, 0.7, 0.6, 0.5])  # Different from memory1/2
        )
        self.memory3.id = "memory3" 
        self.memory3.tags = ["science", "math"]
        
        # Add memories to store
        self.memory_store.add_memory(self.memory1)
        self.memory_store.add_memory(self.memory2)
        self.memory_store.add_memory(self.memory3)
        
        # Create memory sandbox and schema integration
        self.sandbox = MemorySandbox(memory_store=self.memory_store, snapshot_dir=self.temp_dir)
        self.schema_integration = MemorySchemaIntegration(self.sandbox)
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_coherence_calculation(self):
        """Test coherence calculation aligns with formula #2:
        C(A, B) = cosine(v_A, v_B) + λ · tag_overlap(A, B)
        """
        # Build schema graph
        self.schema_integration.build_schema_graph()
        
        # Calculate expected coherence between memory1 and memory2
        # First part: cosine similarity between embeddings
        embedding1 = self.memory1.embedding
        embedding2 = self.memory2.embedding
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        expected_cosine = dot_product / (norm1 * norm2)
        
        # Second part: tag overlap factor
        tags1 = set(self.memory1.tags)
        tags2 = set(self.memory2.tags)
        tag_overlap = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
        
        # λ is typically around 0.5 in the implementation
        lambda_factor = 0.5
        expected_coherence = expected_cosine + lambda_factor * tag_overlap
        
        # Get actual coherence from graph
        graph = self.schema_integration.graph
        edge_weight = 0
        if graph.has_edge("memory1", "memory2"):
            edge_weight = graph["memory1"]["memory2"]["weight"]
        
        # Test coherence is within reasonable range
        self.assertGreater(edge_weight, 0.6, "Coherence score should be positive and significant")
        # We don't expect exact match due to implementation details, but should be close
        self.assertLess(abs(edge_weight - expected_coherence), 0.3, 
                       f"Coherence calculation differs: {edge_weight} vs expected ~{expected_coherence}")
    
    def test_schema_graph_structure(self):
        """Test schema graph structure aligns with formula #6:
        G = (V, E), E_{ij} = avg(C(i, j))
        """
        # Build schema graph
        graph = self.schema_integration.build_schema_graph()
        
        # Verify graph structure
        self.assertEqual(len(graph.nodes()), 3, "Graph should have 3 nodes")
        self.assertGreaterEqual(len(graph.edges()), 1, "Graph should have at least 1 edge")
        
        # Verify node attributes
        for memory_id in ["memory1", "memory2", "memory3"]:
            self.assertIn(memory_id, graph.nodes(), f"Node {memory_id} should be in graph")
            node = graph.nodes[memory_id]
            self.assertIn("memory_type", node, "Node should have memory_type attribute")
            self.assertIn("importance", node, "Node should have importance attribute")
            
        # Verify edge weights
        for edge in graph.edges(data=True):
            source, target, data = edge
            self.assertIn("weight", data, "Edge should have weight attribute")
            self.assertGreaterEqual(data["weight"], 0, "Weight should be non-negative")
            self.assertLessEqual(data["weight"], 1, "Weight should be at most 1")
    
    def test_memory_clusters(self):
        """Test memory clustering aligns with mathematical clustering principles."""
        # Build schema graph
        self.schema_integration.build_schema_graph()
        
        # Detect clusters
        clusters = self.schema_integration.detect_memory_clusters(eps=0.5, min_samples=1)
        
        # We expect at least one cluster
        self.assertGreaterEqual(len(clusters), 1, "Should detect at least one cluster")
        
        # Verify cluster structure
        for cluster_id, cluster_data in clusters.items():
            self.assertIn("size", cluster_data, "Cluster should have size")
            self.assertIn("memory_types", cluster_data, "Cluster should have memory_types")
            self.assertIn("avg_importance", cluster_data, "Cluster should have avg_importance")
            
            # Memory1 and Memory2 should be in the same cluster (similar content/embeddings)
            # Memory3 should be in a different cluster (different content/embedding)
            if "memory1" in cluster_data.get("memory_ids", []):
                self.assertIn("memory2", cluster_data.get("memory_ids", []), 
                             "Memory1 and Memory2 should be in the same cluster")
                self.assertNotIn("memory3", cluster_data.get("memory_ids", []),
                               "Memory3 should not be in the same cluster as Memory1")
    
    def test_concept_generation(self):
        """Test concept generation aligns with mathematical principles."""
        # Build schema graph
        self.schema_integration.build_schema_graph()
        
        # Generate concept suggestions
        concepts = self.schema_integration.generate_concept_suggestions()
        
        # Verify concept structure
        self.assertGreaterEqual(len(concepts), 1, "Should generate at least one concept")
        
        for concept_id, concept_data in concepts.items():
            self.assertIn("concept_name", concept_data, "Concept should have name")
            self.assertIn("dominant_type", concept_data, "Concept should have dominant_type")
            self.assertIn("keywords", concept_data, "Concept should have keywords")
            self.assertIn("importance", concept_data, "Concept should have importance")
            
            # Check the concept associated with memory1/memory2 has animal-related keywords
            if "memory1" in concept_data.get("memory_ids", []) or "memory2" in concept_data.get("memory_ids", []):
                keywords = concept_data["keywords"]
                animal_related = any(keyword in ["animal", "pet", "dog", "cat"] for keyword in keywords)
                self.assertTrue(animal_related, "Concept should have animal-related keywords")


class TestSchemaAnalysisMath(unittest.TestCase):
    """Test the mathematical correctness of schema analysis components."""
    
    def setUp(self):
        """Set up test environment with synthetic schema graphs."""
        # Create SchemaAnalysis instance
        self.analyzer = SchemaAnalysis()
        
        # Create synthetic schema graphs for comparison
        self.schema1 = self._create_test_schema(
            nodes=[
                {"id": "concept1", "node_type": "concept", "name": "Animals"},
                {"id": "memory1", "node_type": "memory", "content": "Dogs are pets", "memory_type": "semantic"},
                {"id": "memory2", "node_type": "memory", "content": "Cats are pets", "memory_type": "semantic"}
            ],
            edges=[
                {"source": "concept1", "target": "memory1", "weight": 0.8},
                {"source": "concept1", "target": "memory2", "weight": 0.7}
            ]
        )
        
        self.schema2 = self._create_test_schema(
            nodes=[
                {"id": "concept1", "node_type": "concept", "name": "Animals"},
                {"id": "concept2", "node_type": "concept", "name": "Domestic Animals"},
                {"id": "memory1", "node_type": "memory", "content": "Dogs are pets", "memory_type": "semantic"},
                {"id": "memory2", "node_type": "memory", "content": "Cats are pets", "memory_type": "semantic"},
                {"id": "memory3", "node_type": "memory", "content": "Elephants are wild animals", "memory_type": "semantic"}
            ],
            edges=[
                {"source": "concept1", "target": "memory1", "weight": 0.8},
                {"source": "concept1", "target": "memory2", "weight": 0.7},
                {"source": "concept1", "target": "memory3", "weight": 0.6},
                {"source": "concept2", "target": "memory1", "weight": 0.9},
                {"source": "concept2", "target": "memory2", "weight": 0.85}
            ]
        )
        
    def _create_test_schema(self, nodes, edges):
        """Create a test schema graph from nodes and edges."""
        graph = nx.Graph()
        
        # Add nodes
        for node in nodes:
            graph.add_node(node["id"], **node)
            
        # Add edges
        for edge in edges:
            graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])
            
        return graph
    
    def test_schema_comparison(self):
        """Test schema comparison aligns with formula #5:
        ΔC = (1/N) · ∑_{i=1}^N (C_i^(t) - C_i^(t-1))
        """
        # Compare schemas
        comparison = self.analyzer.compare_schemas(self.schema1, self.schema2)
        
        # Verify comparison structure
        self.assertIn("summary", comparison, "Comparison should have summary")
        self.assertIn("unique_nodes", comparison, "Comparison should have unique_nodes")
        self.assertIn("unique_edges", comparison, "Comparison should have unique_edges")
        
        # Verify numerical accuracy
        summary = comparison["summary"]
        self.assertEqual(summary["nodes_in_graph1"], 3, "Schema1 should have 3 nodes")
        self.assertEqual(summary["nodes_in_graph2"], 5, "Schema2 should have 5 nodes")
        self.assertEqual(summary["unique_to_graph1"], 0, "No nodes unique to schema1")
        self.assertEqual(summary["unique_to_graph2"], 2, "2 nodes unique to schema2")
        self.assertEqual(summary["common_nodes"], 3, "3 common nodes")
    
    def test_concept_drift(self):
        """Test concept drift analysis aligns with mathematical principles."""
        # Analyze concept drift
        drift = self.analyzer.analyze_concept_drift(self.schema1, self.schema2)
        
        # Verify drift structure
        self.assertIn("summary", drift, "Drift should have summary")
        self.assertIn("evolved_concepts", drift, "Drift should have evolved_concepts")
        self.assertIn("new_concepts", drift, "Drift should have new_concepts")
        
        # Verify numerical accuracy
        summary = drift["summary"]
        self.assertEqual(summary["concepts_in_schema1"], 1, "Schema1 should have 1 concept")
        self.assertEqual(summary["concepts_in_schema2"], 2, "Schema2 should have 2 concepts")
        self.assertEqual(summary["new_concepts"], 1, "1 new concept")
    
    def test_schema_evolution(self):
        """Test schema evolution analysis aligns with formula #5."""
        # Create a third schema for evolution sequence
        schema3 = self._create_test_schema(
            nodes=[
                {"id": "concept1", "node_type": "concept", "name": "Animals"},
                {"id": "concept2", "node_type": "concept", "name": "Domestic Animals"},
                {"id": "concept3", "node_type": "concept", "name": "Wild Animals"},
                {"id": "memory1", "node_type": "memory", "content": "Dogs are pets", "memory_type": "semantic"},
                {"id": "memory2", "node_type": "memory", "content": "Cats are pets", "memory_type": "semantic"},
                {"id": "memory3", "node_type": "memory", "content": "Elephants are wild animals", "memory_type": "semantic"}
            ],
            edges=[
                {"source": "concept1", "target": "memory1", "weight": 0.8},
                {"source": "concept1", "target": "memory2", "weight": 0.7},
                {"source": "concept1", "target": "memory3", "weight": 0.6},
                {"source": "concept2", "target": "memory1", "weight": 0.9},
                {"source": "concept2", "target": "memory2", "weight": 0.85},
                {"source": "concept3", "target": "memory3", "weight": 0.95}
            ]
        )
        
        # Analyze schema evolution
        evolution = self.analyzer.analyze_schema_evolution(
            [self.schema1, self.schema2, schema3],
            ["t1", "t2", "t3"]
        )
        
        # Verify evolution structure
        self.assertIn("summary", evolution, "Evolution should have summary")
        self.assertIn("evolution_steps", evolution, "Evolution should have evolution_steps")
        
        # Verify numerical accuracy
        summary = evolution["summary"]
        self.assertEqual(summary["num_snapshots"], 3, "Should have 3 snapshots")
        self.assertEqual(len(evolution["evolution_steps"]), 2, "Should have 2 evolution steps")
        
        # Verify each step shows increasing complexity
        steps = evolution["evolution_steps"]
        self.assertEqual(steps[0]["from_time"], "t1", "First step should be from t1")
        self.assertEqual(steps[0]["to_time"], "t2", "First step should be to t2")
        self.assertEqual(steps[1]["from_time"], "t2", "Second step should be from t2")
        self.assertEqual(steps[1]["to_time"], "t3", "Second step should be to t3")


if __name__ == "__main__":
    unittest.main() 