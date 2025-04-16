#!/usr/bin/env python3
"""
Unit tests for the Memory Schema Integration module.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path

from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.memory.memory import Memory
from tools.dev_environment.memory_sandbox import MemorySandbox
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration

class TestMemorySchemaIntegration(unittest.TestCase):
    """Test cases for the Memory Schema Integration functionality."""
    
    def setUp(self):
        """Set up test environment with temp dir and initialize objects."""
        self.temp_dir = tempfile.mkdtemp()
        self.snapshot_dir = Path(self.temp_dir) / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Initialize memory store and sandbox
        self.memory_store = MemoryStore()
        self.sandbox = MemorySandbox(memory_store=self.memory_store, 
                                     snapshot_dir=str(self.snapshot_dir))
        
        # Initialize schema integration
        self.schema = MemorySchemaIntegration(self.sandbox)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialize_schema(self):
        """Test initialization of schema integration."""
        self.assertIsNotNone(self.schema)
        self.assertEqual(self.schema.sandbox, self.sandbox)
        self.assertIsNotNone(self.schema.graph)
    
    def test_create_memories_for_schema(self):
        """Test creating memories that will be used for schema testing."""
        # Create test memories with clear relationships
        self.sandbox.create_synthetic_memory(
            content="Cats are mammals with fur",
            memory_type="semantic",
            importance=0.8,
            tags=["animal", "cat", "mammal"]
        )
        
        self.sandbox.create_synthetic_memory(
            content="Dogs are loyal pets",
            memory_type="semantic",
            importance=0.7,
            tags=["animal", "dog", "pet"]
        )
        
        self.sandbox.create_synthetic_memory(
            content="I saw a black cat yesterday",
            memory_type="episodic",
            importance=0.5,
            tags=["experience", "cat"]
        )
        
        # Verify memories were created
        self.assertEqual(len(self.sandbox.memory_store.memories), 3)
    
    def test_build_schema_graph(self):
        """Test building a schema graph from memories."""
        # Create test memories
        self.test_create_memories_for_schema()
        
        # Build schema graph
        self.schema.build_schema_graph()
        
        # Verify graph has nodes
        self.assertGreater(len(self.schema.graph.nodes), 0)
        
        # Verify graph has edges
        self.assertGreater(len(self.schema.graph.edges), 0)
    
    def test_detect_memory_clusters(self):
        """Test detection of memory clusters."""
        # Create test memories
        self.test_create_memories_for_schema()
        
        # Build schema graph
        self.schema.build_schema_graph()
        
        # Detect clusters
        clusters = self.schema.detect_memory_clusters()
        
        # Verify clusters were detected
        self.assertIsNotNone(clusters)
        self.assertGreater(len(clusters), 0)
    
    def test_generate_concept_suggestions(self):
        """Test generating concept suggestions from memory clusters."""
        # Create test memories
        self.test_create_memories_for_schema()
        
        # Build schema graph
        self.schema.build_schema_graph()
        
        # Detect clusters
        self.schema.detect_memory_clusters()
        
        # Generate concept suggestions
        concepts = self.schema.generate_concept_suggestions()
        
        # Verify concepts were generated
        self.assertIsNotNone(concepts)
        self.assertGreater(len(concepts), 0)
    
    def test_find_related_memories(self):
        """Test finding related memories."""
        # Create test memories
        self.test_create_memories_for_schema()
        
        # Build schema graph
        self.schema.build_schema_graph()
        
        # Get the first memory
        first_memory_id = list(self.sandbox.memory_store.memories.keys())[0]
        
        # Find related memories
        related = self.schema.find_related_memories(first_memory_id)
        
        # Verify related memories were found
        self.assertIsNotNone(related)
    
    def test_export_import_schema(self):
        """Test exporting and importing schema graph."""
        # Create test memories
        self.test_create_memories_for_schema()
        
        # Build schema graph
        self.schema.build_schema_graph()
        
        # Export schema
        export_path = os.path.join(self.temp_dir, "schema_export.json")
        self.schema.export_schema_graph(export_path)
        
        # Verify export file exists
        self.assertTrue(os.path.exists(export_path))
        
        # Create new schema integration
        new_schema = MemorySchemaIntegration(self.sandbox)
        
        # Import schema
        new_schema.import_schema_graph(export_path)
        
        # Verify imported graph has same number of nodes
        self.assertEqual(len(new_schema.graph.nodes), len(self.schema.graph.nodes))
    
    def test_generate_knowledge_report(self):
        """Test generating knowledge report."""
        # Create test memories
        self.test_create_memories_for_schema()
        
        # Build schema graph
        self.schema.build_schema_graph()
        
        # Generate report
        report = self.schema.generate_knowledge_report()
        
        # Verify report was generated
        self.assertIsNotNone(report)
        self.assertIn("concepts", report)
        self.assertIn("clusters", report)
    
    def test_schema_statistics(self):
        """Test schema statistics calculation."""
        # Create test memories
        self.test_create_memories_for_schema()
        
        # Build schema graph
        self.schema.build_schema_graph()
        
        # Get statistics
        stats = self.schema.calculate_schema_statistics()
        
        # Verify statistics were calculated
        self.assertIsNotNone(stats)
        self.assertIn("node_count", stats)
        self.assertIn("edge_count", stats)
        self.assertIn("density", stats)

if __name__ == "__main__":
    unittest.main() 