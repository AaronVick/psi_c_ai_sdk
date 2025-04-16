#!/usr/bin/env python
"""
Memory Archive Demo: Demonstrates the memory archiving functionality.

This example shows how memories can be archived instead of being permanently deleted
during culling operations, and how they can be restored from the archive if needed.
"""

import time
import random
import sys
import os
import tempfile
from typing import List

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psi_c_ai_sdk.memory.memory import MemoryStore, Memory


def print_memories(memories: List[Memory], title: str) -> None:
    """Print a formatted list of memories with their importance and content."""
    print(f"\n--- {title} ({len(memories)}) ---")
    if not memories:
        print("  No memories found")
        return
        
    print(f"{'UUID':<10} | {'Importance':<10} | Content")
    print("-" * 70)
    
    for memory in sorted(memories, key=lambda m: m.importance, reverse=True):
        uuid_short = memory.uuid[:8]  # First 8 chars of UUID
        print(f"{uuid_short:<10} | {memory.importance:.4f} | {memory.content[:50]}")


def main() -> None:
    print("Memory Archive Demonstration")
    print("===========================")
    
    # Create a temporary directory for the archive file
    archive_dir = tempfile.mkdtemp()
    archive_path = os.path.join(archive_dir, "memory_archive.json")
    print(f"Using archive path: {archive_path}")
    
    # Create a memory store with archive support
    memory_store = MemoryStore(decay_constant=0.01, archive_path=archive_path)
    
    # Add some high-importance memories
    high_importance_memories = [
        "Important fact: The Earth orbits around the Sun",
        "Important fact: Water is composed of hydrogen and oxygen",
        "Important fact: Humans need oxygen to survive",
        "Important fact: Gravity pulls objects toward each other"
    ]
    
    high_ids = []
    for content in high_importance_memories:
        memory_id = memory_store.add(
            content=content,
            importance=random.uniform(0.7, 0.9),
            tags=["important", "fact"]
        )
        high_ids.append(memory_id)
    
    # Add some low-importance memories
    low_importance_memories = [
        "Trivial fact: I saw a red car today",
        "Trivial fact: Someone sneezed during the meeting",
        "Trivial fact: The office has white walls",
        "Trivial fact: The coffee was a bit cold this morning",
        "Trivial fact: The neighbor's dog barked at 7am",
        "Trivial fact: Someone wore a blue shirt yesterday"
    ]
    
    low_ids = []
    for content in low_importance_memories:
        memory_id = memory_store.add(
            content=content,
            importance=random.uniform(0.2, 0.4),
            tags=["trivial", "everyday"]
        )
        low_ids.append(memory_id)
    
    # Display initial state
    all_memories = memory_store.get_all_memories()
    print_memories(all_memories, "Initial Memories")
    
    # 1. Manually archive a memory
    print("\nManually archiving one high-importance memory...")
    memory_to_archive = high_ids[0]
    memory_store.archive_memory(memory_to_archive)
    print(f"Archived memory: {memory_to_archive[:8]}")
    
    # Show active memories after manual archiving
    print_memories(memory_store.get_all_memories(), "Active Memories After Manual Archive")
    
    # Show archive contents
    print_memories(memory_store.archive.get_all(), "Archive Contents")
    
    # 2. Cull low-importance memories with archiving
    print("\nCulling low-importance memories (with archiving)...")
    removed, archived = memory_store.cull_memories(importance_threshold=0.5, archive=True)
    
    print(f"Culled {removed} memories, archived {archived}")
    
    # Show active memories after culling
    print_memories(memory_store.get_all_memories(), "Active Memories After Culling")
    
    # Show archive contents after culling
    print_memories(memory_store.archive.get_all(), "Archive Contents After Culling")
    
    # 3. Save the archive to disk
    print("\nSaving archive to disk...")
    memory_store.save_archive()
    print(f"Archive saved to: {archive_path}")
    
    # 4. Create a new memory store and load the archive
    print("\nCreating new memory store and loading archive...")
    new_store = MemoryStore(archive_path=archive_path)
    new_store.load_archive()
    
    # Show the loaded archive contents
    print_memories(new_store.archive.get_all(), "Archive Contents in New Store")
    
    # 5. Restore a memory from the archive
    if new_store.archive.count() > 0:
        restore_uuid = new_store.archive.get_all()[0].uuid
        print(f"\nRestoring memory {restore_uuid[:8]} from archive to active store...")
        new_store.restore_from_archive(restore_uuid)
        
        # Show active memories after restoration
        print_memories(new_store.get_all_memories(), "Active Memories After Restoration")
        
        # Show archive contents after restoration
        print_memories(new_store.archive.get_all(), "Archive Contents After Restoration")
    
    # 6. Search the archive
    print("\nSearching archive for 'Trivial'...")
    search_results = new_store.search_archive("Trivial")
    print_memories(search_results, "Search Results")
    
    print("\nArchive statistics:")
    stats = new_store.get_archive_stats()
    print(f"  Total memories in archive: {stats['count']}")
    if stats['last_archived_at']:
        print(f"  Last archived at: {time.ctime(stats['last_archived_at'])}")
    
    # Clean up the temporary directory
    try:
        os.remove(archive_path)
        os.rmdir(archive_dir)
        print(f"\nRemoved temporary archive file: {archive_path}")
    except Exception as e:
        print(f"Error cleaning up: {e}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 