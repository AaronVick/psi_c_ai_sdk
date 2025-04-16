#!/usr/bin/env python3
"""
Unit tests for the Memory Sandbox tool.

This module contains unit tests for the core functionality
of the Memory Sandbox from the ΨC-AI SDK Development Environment.
"""

import os
import sys
import unittest
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path to import the MemorySandbox
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tools.dev_environment.memory_sandbox import MemorySandbox
    from psi_c_ai_sdk.memory.memory_store import MemoryStore
    from psi_c_ai_sdk.memory.memory_types import Memory, MemoryType
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure the ΨC-AI SDK is properly installed")
    sys.exit(1)

class TestMemorySandbox(unittest.TestCase):
    """Test cases for Memory Sandbox."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for snapshots
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a new memory store
        self.memory_store = MemoryStore()
        
        # Initialize the sandbox
        self.sandbox = MemorySandbox(
            memory_store=self.memory_store,
            snapshot_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_synthetic_memory(self):
        """Test creating a synthetic memory."""
        # Create a memory
        memory = self.sandbox.create_synthetic_memory(
            "Test memory content",
            MemoryType.EPISODIC,
            importance=0.7
        )
        
        # Verify memory was created
        self.assertEqual(memory.content, "Test memory content")
        self.assertEqual(memory.memory_type, MemoryType.EPISODIC)
        self.assertEqual(memory.importance, 0.7)
        
        # Verify memory was added to the store
        memories = self.memory_store.get_all_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].content, "Test memory content")
    
    def test_batch_create_memories(self):
        """Test creating a batch of memories."""
        # Create a batch of memories
        memories = self.sandbox.batch_create_memories(
            count=10,
            memory_type=MemoryType.SEMANTIC,
            template="Template memory {index}",
            importance_range=(0.3, 0.8)
        )
        
        # Verify memories were created
        self.assertEqual(len(memories), 10)
        for i, memory in enumerate(memories, 1):
            self.assertEqual(memory.content, f"Template memory {i}")
            self.assertEqual(memory.memory_type, MemoryType.SEMANTIC)
            self.assertTrue(0.3 <= memory.importance <= 0.8)
        
        # Verify memories were added to the store
        store_memories = self.memory_store.get_all_memories()
        self.assertEqual(len(store_memories), 10)
    
    def test_snapshot_functionality(self):
        """Test snapshot creation and loading."""
        # Create some memories
        self.sandbox.create_synthetic_memory(
            "Memory 1",
            MemoryType.EPISODIC,
            importance=0.5
        )
        self.sandbox.create_synthetic_memory(
            "Memory 2",
            MemoryType.SEMANTIC,
            importance=0.6
        )
        
        # Take a snapshot
        snapshot = self.sandbox.take_snapshot("test_snapshot")
        
        # Verify snapshot was created
        self.assertEqual(snapshot["memory_count"], 2)
        self.assertEqual(len(snapshot["memories"]), 2)
        
        # Verify snapshot file exists
        snapshot_files = os.listdir(self.temp_dir)
        self.assertTrue(any("test_snapshot" in f for f in snapshot_files))
        
        # Clear memory store
        self.memory_store.clear()
        self.assertEqual(len(self.memory_store.get_all_memories()), 0)
        
        # Load snapshot
        self.sandbox.load_snapshot("test_snapshot")
        
        # Verify memories were restored
        memories = self.memory_store.get_all_memories()
        self.assertEqual(len(memories), 2)
        contents = [m.content for m in memories]
        self.assertTrue("Memory 1" in contents)
        self.assertTrue("Memory 2" in contents)
    
    def test_memory_recall(self):
        """Test memory recall functionality."""
        # Create memories with specific content
        self.sandbox.create_synthetic_memory(
            "Artificial intelligence uses machine learning algorithms",
            MemoryType.SEMANTIC,
            importance=0.8
        )
        self.sandbox.create_synthetic_memory(
            "Deep learning is a subset of machine learning",
            MemoryType.SEMANTIC,
            importance=0.7
        )
        self.sandbox.create_synthetic_memory(
            "Python is a programming language",
            MemoryType.SEMANTIC,
            importance=0.6
        )
        
        # Test recall with different queries
        ml_results = self.sandbox.test_memory_recall("machine learning", top_k=2)
        python_results = self.sandbox.test_memory_recall("python programming", top_k=1)
        
        # Since the recall is based on random embeddings in this test, we can't
        # guarantee specific results, but we can check basic properties
        self.assertLessEqual(len(ml_results), 2)
        self.assertLessEqual(len(python_results), 1)
    
    def test_memory_deletion(self):
        """Test memory deletion functionality."""
        # Create some memories
        self.sandbox.create_synthetic_memory(
            "Memory to keep",
            MemoryType.EPISODIC,
            importance=0.5
        )
        self.sandbox.create_synthetic_memory(
            "Memory to delete",
            MemoryType.SEMANTIC,
            importance=0.6
        )
        
        # Verify initial state
        self.assertEqual(len(self.memory_store.get_all_memories()), 2)
        
        # Delete by index
        self.sandbox.delete_memory(index=1)
        
        # Verify memory was deleted
        memories = self.memory_store.get_all_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].content, "Memory to keep")
    
    def test_memory_simulation(self):
        """Test memory simulation functionality."""
        # Run a short simulation
        result = self.sandbox.simulate_memory_dynamics(
            duration_days=5,
            memory_decay_rate=0.1,
            memory_creation_rate=2,
            importance_drift=0.05
        )
        
        # Check that memories were created
        memories = self.memory_store.get_all_memories()
        self.assertGreater(len(memories), 0)
        
        # Check that snapshots were created
        snapshot_files = os.listdir(self.temp_dir)
        self.assertTrue(any("before_simulation" in f for f in snapshot_files))
        self.assertTrue(any("after_simulation" in f for f in snapshot_files))
    
    def test_memory_statistics(self):
        """Test memory statistics functionality."""
        # Create memories of different types
        self.sandbox.create_synthetic_memory(
            "Episodic memory",
            MemoryType.EPISODIC,
            importance=0.5
        )
        self.sandbox.create_synthetic_memory(
            "Semantic memory",
            MemoryType.SEMANTIC,
            importance=0.6
        )
        self.sandbox.create_synthetic_memory(
            "Procedural memory",
            MemoryType.PROCEDURAL,
            importance=0.7
        )
        
        # Calculate statistics
        stats = self.sandbox.calculate_memory_statistics()
        
        # Verify basic statistics
        self.assertEqual(stats["basic"]["memory_count"], 3)
        self.assertAlmostEqual(stats["basic"]["avg_importance"], (0.5 + 0.6 + 0.7) / 3, places=5)
        
        # Verify memory type counts
        self.assertEqual(stats["basic"]["memory_types"].get("EPISODIC", 0), 1)
        self.assertEqual(stats["basic"]["memory_types"].get("SEMANTIC", 0), 1)
        self.assertEqual(stats["basic"]["memory_types"].get("PROCEDURAL", 0), 1)
        
        # Verify importance stats
        self.assertEqual(stats["importance"]["min"], 0.5)
        self.assertEqual(stats["importance"]["max"], 0.7)
    
    def test_snapshot_comparison(self):
        """Test snapshot comparison functionality."""
        # Create initial memories
        self.sandbox.create_synthetic_memory(
            "Initial memory 1",
            MemoryType.EPISODIC,
            importance=0.5
        )
        self.sandbox.create_synthetic_memory(
            "Initial memory 2",
            MemoryType.SEMANTIC,
            importance=0.6
        )
        
        # Take first snapshot
        self.sandbox.take_snapshot("snapshot1")
        
        # Modify memories
        self.sandbox.delete_memory(index=0)  # Delete first memory
        self.sandbox.create_synthetic_memory(
            "New memory",
            MemoryType.PROCEDURAL,
            importance=0.7
        )
        
        # Take second snapshot
        self.sandbox.take_snapshot("snapshot2")
        
        # Compare snapshots
        comparison = self.sandbox.compare_snapshots("snapshot1", "snapshot2")
        
        # Verify comparison results
        self.assertEqual(comparison["snapshot1"]["memory_count"], 2)
        self.assertEqual(comparison["snapshot2"]["memory_count"], 2)
        self.assertEqual(comparison["comparison"]["common_memory_count"], 1)
        self.assertEqual(comparison["comparison"]["added_memory_count"], 1)
        self.assertEqual(comparison["comparison"]["removed_memory_count"], 1)

if __name__ == "__main__":
    unittest.main() 