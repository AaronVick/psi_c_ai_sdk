#!/usr/bin/env python
"""
Memory Pinning Demo: Demonstrates the memory pinning functionality.

This example shows how pinned memories are preserved during decay and culling operations,
even when their importance would otherwise cause them to be removed.
"""

import time
import random
import sys
import os
from typing import List

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psi_c_ai_sdk.memory.memory import MemoryStore, Memory


def print_memories(memories: List[Memory], title: str) -> None:
    """Print a formatted list of memories with their importance and pinned status."""
    print(f"\n--- {title} ---")
    print(f"{'UUID':<10} | {'Pinned':<8} | {'Importance':<10} | Content")
    print("-" * 80)
    
    for memory in sorted(memories, key=lambda m: m.importance, reverse=True):
        uuid_short = memory.uuid[:8]  # First 8 chars of UUID
        print(f"{uuid_short:<10} | {str(memory.is_pinned):<8} | {memory.importance:.4f} | {memory.content[:50]}")


def main() -> None:
    print("Memory Pinning Demonstration")
    print("============================")
    
    # Create a memory store with faster decay for demonstration
    memory_store = MemoryStore(decay_constant=0.1)
    
    # Add some regular memories
    regular_memories = [
        "This is a regular memory with medium importance",
        "Another standard memory that might be forgotten",
        "A third memory with nothing special about it",
        "Some information that isn't critical to system function"
    ]
    
    regular_ids = []
    for i, content in enumerate(regular_memories):
        memory_id = memory_store.add(
            content=content,
            importance=random.uniform(0.5, 0.9),  # Medium importance
            tags=["regular"]
        )
        regular_ids.append(memory_id)
    
    # Add some critical memories that should be pinned
    critical_memories = [
        "CRITICAL: Core system configuration parameters",
        "CRITICAL: User authentication credentials",
        "CRITICAL: Security policy rules",
        "CRITICAL: Emergency shutdown protocol"
    ]
    
    critical_ids = []
    for content in critical_memories:
        # Add with similar importance as regular memories
        memory_id = memory_store.add(
            content=content,
            importance=random.uniform(0.5, 0.9),  # Same importance range as regular memories
            tags=["critical"],
            is_pinned=True  # PIN these memories
        )
        critical_ids.append(memory_id)
    
    # Display initial state
    print_memories(memory_store.get_all_memories(), "Initial Memories")
    
    # Apply importance decay to simulate passage of time
    print("\nSimulating time passing and importance decay...")
    
    # Manually reduce importance of non-pinned memories for demonstration
    for uuid, memory in memory_store.memories.items():
        if not memory.is_pinned:
            # Explicitly set some memories to very low importance
            memory.importance *= 0.1  # Reduce to 10% of original value
    
    # Show state after decay
    print_memories(memory_store.get_all_memories(), "After Manual Importance Reduction")
    print("\nNote that pinned memories maintain their original importance values!")
    
    # Now cull memories with low importance
    importance_threshold = 0.4  # Set threshold to be between decayed and original values
    
    # Duplicate the MemoryStore.cull_memories logic here for demonstration
    memories_before = len(memory_store.memories)
    
    # Manually remove low-importance memories
    low_importance_memories = []
    for uuid, memory in list(memory_store.memories.items()):
        if not memory.is_pinned and memory.importance < importance_threshold:
            print(f"Removing memory {uuid[:8]} with importance {memory.importance:.4f}")
            low_importance_memories.append(uuid)
    
    # Remove memories
    for uuid in low_importance_memories:
        memory_store.delete_memory(uuid)
    
    removed = memories_before - len(memory_store.memories)
    
    # Show result
    print(f"\nManually culled {removed} memories with importance below {importance_threshold}")
    print_memories(memory_store.get_all_memories(), "After Culling Low-Importance Memories")
    print("\nNote that pinned memories are preserved even if they have low importance!")
    
    # Demonstrate pinning/unpinning
    print("\nDemonstrating dynamic pinning/unpinning...")
    
    # Unpin one critical memory
    unpin_id = critical_ids[0]
    memory_store.unpin_memory(unpin_id)
    print(f"Unpinned memory: {unpin_id[:8]}")
    
    # Pin one regular memory (if any are left)
    remaining_regular_ids = [uuid for uuid in regular_ids if uuid in memory_store.memories]
    if remaining_regular_ids:
        pin_id = remaining_regular_ids[0]
        memory_store.pin_memory(pin_id)
        print(f"Pinned memory: {pin_id[:8]}")
    
    # Show final state
    print_memories(memory_store.get_all_memories(), "After Pin/Unpin Operations")
    
    # Show just pinned memories
    print_memories(memory_store.get_pinned_memories(), "All Pinned Memories")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 