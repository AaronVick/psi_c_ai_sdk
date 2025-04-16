#!/usr/bin/env python3
"""
Memory Sandbox Demo Script

This script demonstrates the key features of the Memory Sandbox tool
from the ΨC-AI SDK Development Environment.

Run this script to see the Memory Sandbox in action with a guided walkthrough
of its capabilities, from creating memories to simulating memory dynamics.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directories to path to import the MemorySandbox
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from tools.dev_environment.memory_sandbox import MemorySandbox
    from psi_c_ai_sdk.memory.memory_store import MemoryStore
    from psi_c_ai_sdk.memory.memory_types import Memory, MemoryType
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure the ΨC-AI SDK is properly installed")
    sys.exit(1)

class MemorySandboxDemo:
    """Demonstration of Memory Sandbox capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.memory_store = MemoryStore()
        
        # Create sandbox directory
        self.snapshot_dir = os.path.join(os.getcwd(), "demo_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        self.sandbox = MemorySandbox(
            memory_store=self.memory_store,
            snapshot_dir=self.snapshot_dir
        )
        
        # Configure matplotlib for better display in terminals
        plt.style.use('ggplot')
    
    def print_header(self, text):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)
    
    def wait_for_user(self):
        """Wait for user to press Enter to continue."""
        input("\nPress Enter to continue...\n")
    
    def demo_memory_creation(self):
        """Demonstrate memory creation capabilities."""
        self.print_header("MEMORY CREATION DEMO")
        
        print("Creating individual synthetic memories...")
        
        # Create an episodic memory
        episodic = self.sandbox.create_synthetic_memory(
            "Met with the development team to discuss the new project roadmap",
            MemoryType.EPISODIC,
            importance=0.8
        )
        print(f"Created EPISODIC memory with importance {episodic.importance:.2f}")
        
        # Create a semantic memory
        semantic = self.sandbox.create_synthetic_memory(
            "Neural networks leverage gradient descent to optimize weights during training",
            MemoryType.SEMANTIC,
            importance=0.7
        )
        print(f"Created SEMANTIC memory with importance {semantic.importance:.2f}")
        
        # Create a procedural memory
        procedural = self.sandbox.create_synthetic_memory(
            "To initialize a git repository, use 'git init' followed by 'git add .'",
            MemoryType.PROCEDURAL,
            importance=0.6
        )
        print(f"Created PROCEDURAL memory with importance {procedural.importance:.2f}")
        
        # Create an emotional memory
        emotional = self.sandbox.create_synthetic_memory(
            "Felt excited when the team successfully deployed the new feature",
            MemoryType.EMOTIONAL,
            importance=0.9
        )
        print(f"Created EMOTIONAL memory with importance {emotional.importance:.2f}")
        
        self.wait_for_user()
        
        print("Creating a batch of memories...")
        memories = self.sandbox.batch_create_memories(
            count=15,
            memory_type=MemoryType.EPISODIC,
            template="Observed {index} interesting patterns in the data analysis",
            importance_range=(0.3, 0.7)
        )
        print(f"Created {len(memories)} EPISODIC memories with varying importance")
        
        print("\nCurrent memory count:", len(self.memory_store.get_all_memories()))
        
        # Take a snapshot for later comparison
        self.sandbox.take_snapshot("initial_state")
        print("Took snapshot 'initial_state' for later comparison")
        
        self.wait_for_user()
    
    def demo_memory_visualization(self):
        """Demonstrate memory visualization capabilities."""
        self.print_header("MEMORY VISUALIZATION DEMO")
        
        print("Visualizing memory activation patterns...")
        print("(Importance vs Recency for all memories)")
        
        # Visualize memory activation
        self.sandbox.visualize_memory_activation()
        
        self.wait_for_user()
        
        print("Analyzing memory distribution by type...")
        distribution = self.sandbox.analyze_memory_distribution()
        
        # Visualize distribution as pie chart
        labels = list(distribution.keys())
        sizes = list(distribution.values())
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Memory Type Distribution')
        plt.tight_layout()
        plt.show()
        
        self.wait_for_user()
    
    def demo_memory_recall(self):
        """Demonstrate memory recall capabilities."""
        self.print_header("MEMORY RECALL DEMO")
        
        # Add some specific memories for improved recall demo
        self.sandbox.create_synthetic_memory(
            "Machine learning models require regular retraining to avoid concept drift",
            MemoryType.SEMANTIC,
            importance=0.85
        )
        
        self.sandbox.create_synthetic_memory(
            "Gradient boosting combines multiple weak learners to create a strong model",
            MemoryType.SEMANTIC,
            importance=0.75
        )
        
        print("Testing memory recall with different queries...")
        
        queries = [
            "machine learning",
            "development team",
            "git repository",
            "excited about project"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            memories = self.sandbox.test_memory_recall(query, top_k=3)
            
            # Results are already printed by test_memory_recall
        
        self.wait_for_user()
    
    def demo_memory_simulation(self):
        """Demonstrate memory simulation capabilities."""
        self.print_header("MEMORY SIMULATION DEMO")
        
        print("Taking a snapshot before simulation...")
        self.sandbox.take_snapshot("pre_simulation")
        
        print("\nSimulating memory dynamics over 60 days...")
        print("This simulates:")
        print("- Memory creation with realistic patterns")
        print("- Memory importance decay and strengthening")
        print("- Memory recall effects")
        print("- Forgetting of low-importance memories")
        
        self.sandbox.simulate_memory_dynamics(
            duration_days=60,
            memory_decay_rate=0.03,
            memory_creation_rate=3,
            importance_drift=0.01
        )
        
        print("\nSimulation complete!")
        print(f"Current memory count: {len(self.memory_store.get_all_memories())}")
        
        print("\nCalculating memory statistics...")
        stats = self.sandbox.calculate_memory_statistics()
        
        print("\nMemory Statistics:")
        print(f"Total memories: {stats['basic']['memory_count']}")
        print(f"Average importance: {stats['basic']['avg_importance']:.4f}")
        print(f"Time span: {stats['basic']['time_span_days']:.1f} days")
        
        print("\nMemory Types:")
        for memory_type, count in stats['basic']['memory_types'].items():
            percentage = (count / stats['basic']['memory_count']) * 100
            print(f"  {memory_type}: {count} ({percentage:.1f}%)")
        
        print("\nVisualizing post-simulation memory activation...")
        self.sandbox.visualize_memory_activation()
        
        print("\nTaking a snapshot after simulation...")
        self.sandbox.take_snapshot("post_simulation")
        
        self.wait_for_user()
    
    def demo_memory_comparison(self):
        """Demonstrate memory comparison capabilities."""
        self.print_header("MEMORY COMPARISON DEMO")
        
        print("Comparing pre-simulation and post-simulation snapshots...")
        
        comparison = self.sandbox.compare_snapshots("pre_simulation", "post_simulation")
        
        print("\nSnapshot Comparison:")
        print(f"Pre-simulation memory count: {comparison['snapshot1']['memory_count']}")
        print(f"Post-simulation memory count: {comparison['snapshot2']['memory_count']}")
        print(f"Common memories: {comparison['comparison']['common_memory_count']}")
        print(f"Added memories: {comparison['comparison']['added_memory_count']}")
        print(f"Removed memories: {comparison['comparison']['removed_memory_count']}")
        print(f"Average importance change: {comparison['comparison']['importance_change']:.4f}")
        
        print("\nType Distribution Comparison:")
        print("Pre-simulation:")
        for memory_type, count in comparison['snapshot1']['type_distribution'].items():
            print(f"  {memory_type}: {count}")
        
        print("\nPost-simulation:")
        for memory_type, count in comparison['snapshot2']['type_distribution'].items():
            print(f"  {memory_type}: {count}")
        
        # Create a bar chart comparing memory types
        pre_types = comparison['snapshot1']['type_distribution']
        post_types = comparison['snapshot2']['type_distribution']
        
        # Combine all memory types
        all_types = set(pre_types.keys()).union(set(post_types.keys()))
        types = sorted(list(all_types))
        
        # Get counts for each type
        pre_counts = [pre_types.get(t, 0) for t in types]
        post_counts = [post_types.get(t, 0) for t in types]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(types))
        width = 0.35
        
        ax.bar(x - width/2, pre_counts, width, label='Pre-Simulation')
        ax.bar(x + width/2, post_counts, width, label='Post-Simulation')
        
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.legend()
        
        plt.title('Memory Type Distribution Comparison')
        plt.xlabel('Memory Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
        
        self.wait_for_user()
    
    def run_demo(self):
        """Run the complete demo sequence."""
        self.print_header("MEMORY SANDBOX DEMONSTRATION")
        print("This demo will walk through the key features of the Memory Sandbox tool")
        print("from the ΨC-AI SDK Development Environment.")
        
        self.wait_for_user()
        
        try:
            self.demo_memory_creation()
            self.demo_memory_visualization()
            self.demo_memory_recall()
            self.demo_memory_simulation()
            self.demo_memory_comparison()
            
            self.print_header("DEMO COMPLETE")
            print("Explored key Memory Sandbox features:")
            print("- Memory creation (individual and batch)")
            print("- Memory visualization")
            print("- Memory recall and retrieval")
            print("- Memory simulation")
            print("- Memory comparison")
            
            print("\nSnapshot directory:", self.snapshot_dir)
            print("\nTo explore more features, check out the Memory Sandbox documentation")
            print("or use the web interface with:")
            print("  python -m tools.dev_environment.launcher web")
            print("\nThank you for trying the Memory Sandbox!")
            
        except Exception as e:
            print(f"Error during demo: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    demo = MemorySandboxDemo()
    demo.run_demo() 