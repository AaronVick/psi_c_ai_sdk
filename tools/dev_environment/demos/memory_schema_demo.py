#!/usr/bin/env python3
"""
Memory Schema Integration Demo for the ΨC-AI SDK Development Environment.

This demo shows how to use the Memory Schema Integration module to:
1. Create memory clusters with thematic content
2. Build a schema graph from memories
3. Detect memory clusters automatically
4. Generate concept suggestions
5. Analyze memory relationships

Run this demo directly: python -m tools.dev_environment.demos.memory_schema_demo
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.memory.memory import Memory
from tools.dev_environment.memory_sandbox import MemorySandbox
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration

class MemorySchemaDemo:
    """Demonstrates the Memory Schema Integration functionality."""
    
    def __init__(self):
        """Initialize the demo environment."""
        self.memory_store = MemoryStore()
        self.sandbox = MemorySandbox(memory_store=self.memory_store)
        self.schema_integration = MemorySchemaIntegration(memory_store=self.memory_store)
        
        # For storing snapshots
        self.snapshots = {}
        
    def create_themed_memory_clusters(self):
        """Create memory clusters with thematic content for testing."""
        print("Creating themed memory clusters...")
        
        # Theme 1: Travel memories
        travel_embeddings = np.random.rand(5, 128) * 0.2 + np.ones((5, 128)) * 0.4
        travel_memories = [
            ("I visited Paris last summer and saw the Eiffel Tower.", "episodic", 0.8),
            ("The best way to travel around Europe is by train.", "semantic", 0.7),
            ("Remember to pack light when traveling internationally.", "procedural", 0.6),
            ("I felt excited when I booked my trip to Japan.", "emotional", 0.9),
            ("Rome has amazing historical sites and delicious food.", "episodic", 0.75)
        ]
        
        # Theme 2: Work-related memories
        work_embeddings = np.random.rand(5, 128) * 0.2 + np.ones((5, 128)) * 0.6
        work_memories = [
            ("The quarterly meeting is scheduled for next Monday.", "episodic", 0.6),
            ("Effective project management requires clear communication.", "semantic", 0.8),
            ("To create a new project, click on File > New > Project.", "procedural", 0.7),
            ("I was proud when our team completed the project ahead of schedule.", "emotional", 0.85),
            ("Our company's mission is to innovate and inspire.", "semantic", 0.65)
        ]
        
        # Theme 3: Learning memories
        learning_embeddings = np.random.rand(5, 128) * 0.2 + np.ones((5, 128)) * 0.8
        learning_memories = [
            ("I attended a workshop on machine learning yesterday.", "episodic", 0.7),
            ("Neural networks consist of layers of interconnected nodes.", "semantic", 0.85),
            ("To train a model, first split your data into training and test sets.", "procedural", 0.75),
            ("I felt accomplished after solving a difficult programming problem.", "emotional", 0.8),
            ("Regular practice is essential for mastering any skill.", "semantic", 0.9)
        ]
        
        # Create all memories with their respective embeddings
        memory_sets = [
            (travel_memories, travel_embeddings),
            (work_memories, work_embeddings),
            (learning_memories, learning_embeddings)
        ]
        
        all_memory_ids = []
        
        for memory_set, embeddings in memory_sets:
            for i, (content, memory_type, importance) in enumerate(memory_set):
                memory = Memory(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    creation_time=datetime.now() - timedelta(days=i),
                    embedding=embeddings[i]
                )
                self.memory_store.add_memory(memory)
                all_memory_ids.append(memory.id)
                
                # Randomly connect some memories
                if all_memory_ids and np.random.random() > 0.7:
                    memory.related_memories.append(np.random.choice(all_memory_ids))
        
        print(f"Created {len(all_memory_ids)} memories in 3 themed clusters")
        
    def build_schema_graph(self):
        """Build a schema graph from the memories and visualize it."""
        print("\nBuilding schema graph from memories...")
        
        # Build the schema graph
        schema_graph = self.schema_integration.build_schema_graph()
        
        print(f"Schema graph built with {len(schema_graph.nodes)} nodes and {len(schema_graph.edges)} edges")
        
        # Visualize the graph
        fig = self.schema_integration.visualize_schema_graph()
        plt.savefig("memory_schema_graph.png")
        plt.close(fig)
        
        print("Schema graph visualization saved to 'memory_schema_graph.png'")
        
    def detect_memory_clusters(self):
        """Detect memory clusters automatically."""
        print("\nDetecting memory clusters...")
        
        # The clusters are already detected during build_schema_graph(),
        # but we can also access them directly
        memory_clusters = self.schema_integration.get_memory_clusters()
        
        print(f"Detected {len(memory_clusters)} memory clusters:")
        for cluster_id, cluster_data in memory_clusters.items():
            memory_ids = cluster_data["memory_ids"]
            suggestion = cluster_data["concept_suggestion"]
            
            print(f"  - {cluster_id}: {len(memory_ids)} memories")
            if suggestion:
                print(f"    Suggested concept: {suggestion.get('concept_name', 'Unknown')}")
                print(f"    Keywords: {', '.join(suggestion.get('keywords', []))}")
                print(f"    Dominant type: {suggestion.get('dominant_type', 'Unknown')}")
                
    def add_concept_nodes(self):
        """Add concept nodes to the schema graph based on suggestions."""
        print("\nAdding concept nodes based on suggestions...")
        
        memory_clusters = self.schema_integration.get_memory_clusters()
        added_concepts = []
        
        for cluster_id, cluster_data in memory_clusters.items():
            memory_ids = cluster_data["memory_ids"]
            suggestion = cluster_data["concept_suggestion"]
            
            if suggestion:
                concept_name = suggestion.get("concept_name", f"Concept {cluster_id}")
                concept_id = self.schema_integration.add_concept_node(
                    name=concept_name,
                    memory_ids=memory_ids
                )
                added_concepts.append((concept_id, concept_name, len(memory_ids)))
                
        print(f"Added {len(added_concepts)} concept nodes to the schema graph:")
        for concept_id, name, count in added_concepts:
            print(f"  - {name} ({concept_id}): connected to {count} memories")
            
        # Visualize the updated graph
        fig = self.schema_integration.visualize_schema_graph()
        plt.savefig("memory_schema_graph_with_concepts.png")
        plt.close(fig)
        
        print("Updated schema graph visualization saved to 'memory_schema_graph_with_concepts.png'")
        
    def export_import_schema(self):
        """Demonstrate schema export and import functionality."""
        print("\nDemonstrating schema export and import...")
        
        # Export the schema graph
        schema_data = self.schema_integration.export_schema_graph()
        
        print(f"Exported schema with {len(schema_data['nodes'])} nodes and {len(schema_data['edges'])} edges")
        
        # Create a new integration instance
        new_integration = MemorySchemaIntegration()
        
        # Import the schema
        new_integration.import_schema_graph(schema_data)
        
        print("Schema successfully imported into new integration instance")
        print(f"Imported schema has {len(new_integration.schema_graph.nodes)} nodes and {len(new_integration.schema_graph.edges)} edges")
        
    def run_demo(self):
        """Run the full demonstration."""
        print("=" * 80)
        print("Memory Schema Integration Demo".center(80))
        print("=" * 80)
        
        # Create memory clusters
        self.create_themed_memory_clusters()
        
        # Build and visualize schema graph
        self.build_schema_graph()
        
        # Detect memory clusters
        self.detect_memory_clusters()
        
        # Add concept nodes
        self.add_concept_nodes()
        
        # Export and import schema
        self.export_import_schema()
        
        print("\nDemo completed! Check the generated visualization files.")
        print("=" * 80)
        
if __name__ == "__main__":
    demo = MemorySchemaDemo()
    demo.run_demo() 