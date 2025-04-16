"""
Simple Test for the ΨC-AI SDK Structure

This script simply tests that the package structure is correct
and that imports are working correctly.
"""

import os
import sys
import json

# Add parent directory to path to import from psi_c_ai_sdk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the classes (but don't instantiate them)
from psi_c_ai_sdk.memory.memory import Memory, MemoryStore

def main():
    # Create a memory store
    memory_store = MemoryStore(decay_constant=0.05)
    
    # Add some memories without using external dependencies
    memory_ids = []
    memory_ids.append(memory_store.add("This is test memory 1", tags=["test"]))
    memory_ids.append(memory_store.add("This is test memory 2", tags=["test"]))
    
    print(f"Added {len(memory_ids)} memories.")
    
    # Get all memories
    memories = [memory_store.get_memory(mid) for mid in memory_ids]
    
    # Print memory content
    for i, memory in enumerate(memories):
        print(f"Memory {i+1}:")
        print(f"  UUID: {memory.uuid}")
        print(f"  Content: {memory.content}")
        print(f"  Importance: {memory.importance}")
        print(f"  Tags: {memory.tags}")
    
    # Export memories to a file
    export_file = "test_memories.json"
    memory_store.export(export_file)
    print(f"Exported memories to {export_file}")
    
    # Load the exported file to verify
    with open(export_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['memories'])} memories from exported file.")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    main() 