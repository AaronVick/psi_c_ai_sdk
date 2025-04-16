"""
Unit tests for memory compression functionality.
"""

import unittest
from typing import List

from psi_c_ai_sdk.memory import (
    Memory,
    MemoryStore,
    MemoryCompressor,
    CompressionLevel,
    CompressedMemory
)


class TestMemoryCompression(unittest.TestCase):
    """Test cases for memory compression functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.memory_store = MemoryStore()
        self.compressor = MemoryCompressor(self.memory_store, similarity_threshold=0.6)
        
        # Create some test memories
        self.similar_memories = [
            Memory(
                content="I learned about artificial intelligence today.",
                importance=0.8,
                tags=["AI", "learning"]
            ),
            Memory(
                content="Today I studied artificial intelligence and found it fascinating.",
                importance=0.7,
                tags=["AI", "learning"]
            ),
            Memory(
                content="Artificial intelligence is an important field to understand.",
                importance=0.9,
                tags=["AI", "important"]
            )
        ]
        
        # Add memories to store for testing
        for memory in self.similar_memories:
            self.memory_store.add_memory(memory)
    
    def test_memory_similarity_detection(self):
        """Test that similar memories are correctly identified."""
        # Check first two memories are similar
        self.assertTrue(
            self.compressor._are_memories_similar(
                self.similar_memories[0], 
                self.similar_memories[1]
            )
        )
        
        # Add a dissimilar memory
        dissimilar = Memory(
            content="I went for a walk in the park today.",
            importance=0.5,
            tags=["exercise", "nature"]
        )
        
        # Check it's not similar to the AI memories
        self.assertFalse(
            self.compressor._are_memories_similar(
                self.similar_memories[0],
                dissimilar
            )
        )
    
    def test_identify_compressible_clusters(self):
        """Test identification of compressible memory clusters."""
        clusters = self.compressor.identify_compressible_clusters(min_cluster_size=2)
        
        # Should find at least one cluster
        self.assertGreaterEqual(len(clusters), 1)
        
        # First cluster should have at least 2 memories
        self.assertGreaterEqual(len(clusters[0]), 2)
    
    def test_compression_levels(self):
        """Test compression at different levels."""
        for level in CompressionLevel:
            compressed = self.compressor.compress_memories(
                self.similar_memories,
                compression_level=level,
                preserve_originals=True
            )
            
            if level == CompressionLevel.NONE:
                # For NONE level, all contents should be included
                for memory in self.similar_memories:
                    self.assertIn(memory.content, compressed.content)
            elif level == CompressionLevel.HIGH:
                # For HIGH level, should be very concise
                self.assertIn("Highly compressed", compressed.content)
                # Content should be shorter than originals
                combined_length = sum(len(m.content) for m in self.similar_memories)
                self.assertLess(len(compressed.content), combined_length)
    
    def test_compression_and_decompression(self):
        """Test complete compression and decompression cycle."""
        # Compress memories
        compressed = self.compressor.compress_memories(
            self.similar_memories,
            compression_level=CompressionLevel.MEDIUM,
            preserve_originals=True
        )
        
        self.assertIsNotNone(compressed)
        self.assertEqual(len(compressed.original_uuids), len(self.similar_memories))
        
        # Verify compression was tracked
        self.assertEqual(self.compressor.stats["total_compressions"], 1)
        self.assertEqual(
            self.compressor.stats["total_memories_compressed"], 
            len(self.similar_memories)
        )
        
        # Test decompression
        decompressed = self.compressor.decompress_memory(compressed.uuid)
        
        self.assertIsNotNone(decompressed)
        self.assertEqual(len(decompressed), len(self.similar_memories))
        
        # Check original UUIDs are preserved
        original_uuids = [m.uuid for m in self.similar_memories]
        decompressed_uuids = [m.uuid for m in decompressed]
        
        for uuid in original_uuids:
            self.assertIn(uuid, decompressed_uuids)
    
    def test_restoration_to_store(self):
        """Test restoring compressed memories to the store."""
        # Get initial count
        initial_count = len(self.memory_store.get_all_memories())
        
        # Compress memories
        compressed = self.compressor.compress_memories(
            self.similar_memories,
            compression_level=CompressionLevel.MEDIUM
        )
        
        # After compression, should have 1 compressed memory instead of originals
        compressed_count = len(self.memory_store.get_all_memories())
        self.assertEqual(compressed_count, initial_count - len(self.similar_memories) + 1)
        
        # Restore memories
        success = self.compressor.restore_memories(compressed.uuid)
        self.assertTrue(success)
        
        # After restoration, should have original count again
        restored_count = len(self.memory_store.get_all_memories())
        self.assertEqual(restored_count, initial_count)
    
    def test_compression_cycle(self):
        """Test running a full compression cycle."""
        # Add some more similar memories
        more_memories = [
            Memory(
                content="Machine learning is a subset of artificial intelligence.",
                importance=0.8,
                tags=["AI", "ML"]
            ),
            Memory(
                content="Deep learning is an advanced technique in machine learning.",
                importance=0.7,
                tags=["ML", "DL"]
            ),
            Memory(
                content="Neural networks are the foundation of deep learning systems.",
                importance=0.9,
                tags=["ML", "DL", "neural networks"]
            )
        ]
        
        for memory in more_memories:
            self.memory_store.add_memory(memory)
        
        # Run compression cycle
        results = self.compressor.run_compression_cycle(
            compression_level=CompressionLevel.MEDIUM,
            min_cluster_size=2
        )
        
        # Should have found and compressed at least one cluster
        self.assertGreaterEqual(results["clusters_found"], 1)
        self.assertGreaterEqual(results["clusters_compressed"], 1)
        self.assertGreaterEqual(results["memories_compressed"], 2)
        
        # Check compression stats were updated
        stats = self.compressor.get_compression_stats()
        self.assertGreaterEqual(stats["total_compressions"], 1)
        self.assertGreaterEqual(stats["total_memories_compressed"], 2)
    
    def test_memory_reconstruction(self):
        """Test reconstruction when original memories aren't available."""
        # Compress memories without preserving originals
        compressed = self.compressor.compress_memories(
            self.similar_memories,
            compression_level=CompressionLevel.MEDIUM,
            preserve_originals=False
        )
        
        # Force original_memories to None to simulate missing originals
        compressed.original_memories = None
        
        # Try to decompress
        reconstructed = self.compressor.decompress_memory(compressed.uuid)
        
        # Should get reconstructed memories with warnings
        self.assertIsNotNone(reconstructed)
        self.assertGreaterEqual(len(reconstructed), 1)
        
        # First memory should be a warning
        self.assertIn("WARNING", reconstructed[0].content)
        self.assertIn("reconstructed", reconstructed[0].tags)


if __name__ == "__main__":
    unittest.main() 