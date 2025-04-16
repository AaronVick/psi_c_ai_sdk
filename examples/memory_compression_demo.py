#!/usr/bin/env python
"""
Memory Compression Demo

This script demonstrates the memory compression capabilities of the ΨC-AI SDK,
including semantic compression of redundant memories, compression level tracking,
and decompression for inspection.
"""

import time
import random
import logging
from typing import List

from psi_c_ai_sdk.memory import (
    Memory, 
    MemoryStore, 
    MemoryCompressor, 
    CompressionLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_similar_memories(topic: str, count: int = 5, variation: float = 0.3) -> List[Memory]:
    """
    Generate a set of similar memories about a topic.
    
    Args:
        topic: The main topic for the memories
        count: Number of memories to generate
        variation: How much variation to include (0-1)
        
    Returns:
        List of similar Memory objects
    """
    base_templates = [
        f"I learned about {topic} today and found it interesting.",
        f"Today I studied {topic} and gained some insights.",
        f"I read an article about {topic} that was informative.",
        f"{topic} is an important concept worth understanding.",
        f"I spent some time thinking about {topic} today."
    ]
    
    adjectives = ["interesting", "important", "complex", "fascinating", "useful", 
                 "relevant", "essential", "noteworthy", "significant"]
    
    verbs = ["learned", "studied", "read about", "explored", "investigated", 
            "researched", "analyzed", "examined"]
    
    memories = []
    for i in range(count):
        # Select a base template
        template = random.choice(base_templates)
        
        # Add variation if needed
        if random.random() < variation:
            adj = random.choice(adjectives)
            verb = random.choice(verbs)
            template = template.replace("interesting", adj)
            
            for v in verbs:
                if v in template:
                    template = template.replace(v, verb)
        
        # Add some additional details for variety
        details = [
            f" It relates to several other concepts.",
            f" I think this could be applied in various contexts.",
            f" This knowledge will be useful in the future.",
            f" I should explore this further.",
            ""  # Sometimes no additional detail
        ]
        
        content = template + random.choice(details)
        
        # Create memory with appropriate tags
        memory = Memory(
            content=content,
            importance=random.uniform(0.6, 1.0),
            tags=[topic, "learning"] + (["important"] if random.random() > 0.7 else []),
            metadata={
                "source": random.choice(["article", "book", "conversation", "thought"]),
                "confidence": random.uniform(0.7, 1.0)
            }
        )
        
        memories.append(memory)
    
    return memories


def demonstrate_compression_levels(memory_store: MemoryStore, compressor: MemoryCompressor):
    """Show compression at different levels."""
    logger.info("Demonstrating different compression levels...")
    
    # Generate memories about AI for each compression level
    memories = generate_similar_memories("artificial intelligence", count=5)
    for memory in memories:
        memory_store.add_memory(memory)
    
    logger.info(f"Added {len(memories)} memories about AI")
    
    # Compress memories at different levels
    for level in [CompressionLevel.LOW, CompressionLevel.MEDIUM, CompressionLevel.HIGH]:
        logger.info(f"\nCompressing memories at {level.name} level:")
        compressed = compressor.compress_memories(memories, compression_level=level)
        
        if compressed:
            logger.info(f"Compressed {len(memories)} memories into: \n{compressed.content}\n")
            
            # Demonstrate decompression
            logger.info(f"Decompressing {level.name} compression:")
            decompressed = compressor.decompress_memory(compressed.uuid)
            
            if decompressed:
                logger.info(f"Successfully decompressed into {len(decompressed)} memories")
                for i, memory in enumerate(decompressed[:2]):  # Show first few
                    logger.info(f"  Memory {i+1}: {memory.content[:100]}...")
                if len(decompressed) > 2:
                    logger.info(f"  ... and {len(decompressed) - 2} more")
            
            # Restore only the last compression to avoid conflicts
            if level == CompressionLevel.HIGH:
                logger.info("\nRestoring original memories to store:")
                success = compressor.restore_memories(compressed.uuid)
                logger.info(f"Restoration {'successful' if success else 'failed'}")
        else:
            logger.error(f"Compression at {level.name} level failed")


def demonstrate_auto_compression(memory_store: MemoryStore, compressor: MemoryCompressor):
    """Demonstrate automatic detection and compression of similar memories."""
    logger.info("\nDemonstrating automatic memory compression...")
    
    # Add groups of similar memories on different topics
    topics = ["quantum physics", "machine learning", "philosophy", "neuroscience"]
    
    for topic in topics:
        memories = generate_similar_memories(topic, count=random.randint(3, 7))
        for memory in memories:
            memory_store.add_memory(memory)
        logger.info(f"Added {len(memories)} memories about {topic}")
    
    # Let's add some unique memories too
    unique_topics = ["cooking", "gardening", "exercise", "meditation", "travel"]
    for topic in unique_topics[:3]:
        memory = Memory(
            content=f"I spent time {topic} today and enjoyed it thoroughly.",
            importance=0.8,
            tags=[topic, "personal"],
        )
        memory_store.add_memory(memory)
    
    logger.info(f"\nMemory store now contains {len(memory_store.get_all_memories())} memories")
    
    # Run a compression cycle
    logger.info("\nRunning automatic compression cycle...")
    results = compressor.run_compression_cycle(
        compression_level=CompressionLevel.MEDIUM,
        min_cluster_size=3
    )
    
    logger.info(f"Compression cycle results:")
    logger.info(f"  Found {results['clusters_found']} compressible clusters")
    logger.info(f"  Compressed {results['clusters_compressed']} clusters")
    logger.info(f"  Total of {results['memories_compressed']} memories were compressed")
    
    # Display compression stats
    stats = compressor.get_compression_stats()
    logger.info("\nCompression statistics:")
    logger.info(f"  Total compressions: {stats['total_compressions']}")
    logger.info(f"  Total memories compressed: {stats['total_memories_compressed']}")
    logger.info(f"  Estimated space saved: {stats['space_saved']} bytes")
    
    # Show compressed memories
    logger.info("\nCompressed memories currently in store:")
    compressed_count = 0
    
    for memory in memory_store.get_all_memories():
        if memory.metadata.get("is_compressed"):
            compressed_count += 1
            orig_count = memory.metadata.get("original_count", "?")
            logger.info(f"  Compressed memory containing {orig_count} originals: {memory.content[:100]}...")
    
    logger.info(f"Total compressed memory objects: {compressed_count}")
    

def main():
    logger.info("Memory Compression Demo")
    
    # Create memory store
    memory_store = MemoryStore()
    
    # Create memory compressor
    compressor = MemoryCompressor(memory_store, similarity_threshold=0.7)
    
    # Demonstrate compression at different levels
    demonstrate_compression_levels(memory_store, compressor)
    
    # Clear memory store for next demonstration
    memory_store.clear()
    
    # Demonstrate automatic compression
    demonstrate_auto_compression(memory_store, compressor)
    
    logger.info("\nDemo complete!")


if __name__ == "__main__":
    main() 