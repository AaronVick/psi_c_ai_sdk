#!/usr/bin/env python3
"""
Schema Integration Demo for the ΨC-AI SDK Memory Sandbox

This demo demonstrates the capabilities of the memory schema integration,
including building knowledge graphs from memories, detecting clusters,
generating concept suggestions, and visualizing the relationships between memories.

Usage:
    python -m tools.dev_environment.demos.schema_integration_demo

The demo will:
1. Create a set of related memories in different categories
2. Build a schema graph representing memory relationships
3. Detect clusters of related memories
4. Generate concept suggestions from the clusters
5. Visualize the schema graph
6. Export and import the schema graph
7. Generate a knowledge report
"""

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import Memory Sandbox components
from tools.dev_environment.memory_sandbox import MemorySandbox, MemoryStore
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration

class SchemaIntegrationDemo:
    """Demonstration of the Memory Schema Integration capabilities."""
    
    def __init__(self, output_dir="./demo_output"):
        """
        Initialize the schema integration demo.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize memory store and sandbox
        self.memory_store = MemoryStore()
        self.sandbox = MemorySandbox(memory_store=self.memory_store, 
                                     snapshot_dir=os.path.join(output_dir, "snapshots"))
        
        # Initialize schema integration
        self.schema = MemorySchemaIntegration(self.sandbox)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def create_test_memories(self):
        """Create test memories for the schema integration demo."""
        print("\n[1/7] Creating test memories in different categories...")
        
        # Create memories about machine learning
        self.create_memory_group(
            base_content="Machine learning is a subset of artificial intelligence.",
            variations=[
                "Neural networks are a powerful technique in machine learning.",
                "Supervised learning requires labeled training data.",
                "Reinforcement learning involves reward-based training.",
                "Deep learning uses multiple layers of neural networks.",
                "GPT models have revolutionized natural language processing."
            ],
            memory_type="semantic",
            importance_range=(0.7, 0.9),
            tags=["AI", "machine learning", "technology"]
        )
        
        # Create memories about nature
        self.create_memory_group(
            base_content="I went hiking in the mountains last weekend.",
            variations=[
                "The forest was full of beautiful pine trees.",
                "We saw a deer drinking from a stream.",
                "The view from the mountain top was breathtaking.",
                "The wildflowers were in full bloom on the trail.",
                "We had a picnic by a waterfall in the afternoon."
            ],
            memory_type="episodic",
            importance_range=(0.5, 0.8),
            tags=["nature", "hiking", "experience"]
        )
        
        # Create memories about cooking
        self.create_memory_group(
            base_content="To make a good pasta sauce, start with quality tomatoes.",
            variations=[
                "Adding basil at the end preserves its flavor.",
                "Cooking garlic too long can make it bitter.",
                "A pinch of sugar can balance acidity in tomato sauce.",
                "Fresh herbs are generally added after cooking.",
                "Use pasta water to help sauce stick to the noodles."
            ],
            memory_type="procedural",
            importance_range=(0.6, 0.8),
            tags=["cooking", "recipe", "food"]
        )
        
        # Create some emotional memories
        self.create_memory_group(
            base_content="I was so happy when I received the good news.",
            variations=[
                "The concert was an amazing experience that made me feel alive.",
                "I felt a profound sense of joy at the graduation ceremony.",
                "The surprise party made me feel so appreciated.",
                "Watching the sunset on the beach filled me with peace."
            ],
            memory_type="emotional",
            importance_range=(0.7, 0.95),
            tags=["emotion", "happiness", "experience"]
        )
        
        # Create some memories that bridge categories
        self.sandbox.create_synthetic_memory(
            content="I learned a new machine learning technique while hiking.",
            memory_type="episodic",
            importance=0.75,
            tags=["hiking", "machine learning", "experience"]
        )
        
        self.sandbox.create_synthetic_memory(
            content="The AI model predicted which cooking recipes would make me happy.",
            memory_type="semantic",
            importance=0.8,
            tags=["AI", "cooking", "happiness"]
        )
        
        # Print summary
        print(f"Created {len(self.sandbox.memory_store.memories)} memories for schema analysis")
    
    def create_memory_group(self, base_content, variations, memory_type, importance_range, tags):
        """
        Create a group of related memories with similar embeddings.
        
        Args:
            base_content: The base content for the memories
            variations: List of content variations
            memory_type: Type of memory to create
            importance_range: Range of importance values (min, max)
            tags: List of tags to apply to the memories
        """
        # Create a base embedding for this group
        base_embedding = np.random.random(128)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Create the base memory
        base_memory_id = self.sandbox.create_synthetic_memory(
            content=base_content,
            memory_type=memory_type,
            importance=random.uniform(*importance_range),
            tags=tags,
            embedding=base_embedding
        )
        
        # Create variations with similar embeddings
        variation_ids = []
        for content in variations:
            # Create a similar but slightly different embedding
            noise = np.random.normal(0, 0.1, 128)
            embedding = base_embedding + noise
            embedding = embedding / np.linalg.norm(embedding)
            
            # Create the memory
            memory_id = self.sandbox.create_synthetic_memory(
                content=content,
                memory_type=memory_type,
                importance=random.uniform(*importance_range),
                tags=tags,
                embedding=embedding
            )
            
            variation_ids.append(memory_id)
        
        # Add some relationships between the memories
        base_memory = self.sandbox.memory_store.memories[base_memory_id]
        for var_id in variation_ids[:2]:  # Only connect to some variations
            base_memory.related_memories.append(var_id)
            self.sandbox.memory_store.memories[var_id].related_memories.append(base_memory_id)
    
    def build_schema_graph(self):
        """Build and visualize the schema graph."""
        print("\n[2/7] Building schema graph from memories...")
        
        # Build the schema graph
        self.schema.build_schema_graph()
        
        # Print graph statistics
        stats = self.schema.calculate_schema_statistics()
        print(f"Schema graph created with {stats['node_count']} nodes and {stats['edge_count']} edges")
        
        # Get memory type distribution
        if 'memory_type_distribution' in stats:
            print("\nMemory type distribution:")
            for memory_type, count in stats['memory_type_distribution'].items():
                print(f"  - {memory_type}: {count}")
    
    def detect_clusters(self):
        """Detect memory clusters and print results."""
        print("\n[3/7] Detecting memory clusters...")
        
        # Detect clusters
        clusters = self.schema.detect_memory_clusters(eps=0.6, min_samples=2)
        
        # Print cluster info
        print(f"Detected {len(clusters)} clusters of related memories:")
        
        for cluster_id, cluster_data in clusters.items():
            memory_types = cluster_data['memory_types']
            dominant_type = max(memory_types.items(), key=lambda x: x[1])[0]
            
            print(f"\nCluster {cluster_id}:")
            print(f"  - Size: {cluster_data['size']} memories")
            print(f"  - Dominant type: {dominant_type}")
            print(f"  - Average importance: {cluster_data['avg_importance']:.2f}")
            print(f"  - Sample contents:")
            
            # Print up to 3 sample contents
            for content in cluster_data['contents'][:3]:
                print(f"    * {content[:50]}..." if len(content) > 50 else f"    * {content}")
    
    def generate_concepts(self):
        """Generate concept suggestions from memory clusters."""
        print("\n[4/7] Generating concept suggestions...")
        
        # Generate concepts
        concepts = self.schema.generate_concept_suggestions()
        
        # Print concept info
        print(f"Generated {len(concepts)} concept suggestions:")
        
        for concept_id, concept_data in concepts.items():
            print(f"\nConcept: {concept_data['concept_name']}")
            print(f"  - Memory type: {concept_data['dominant_type']}")
            print(f"  - Keywords: {', '.join(concept_data['keywords'])}")
            print(f"  - Associated with {len(concept_data['memory_ids'])} memories")
            print(f"  - Importance: {concept_data['importance']:.2f}")
    
    def visualize_schema(self):
        """Visualize the schema graph."""
        print("\n[5/7] Visualizing schema graph...")
        
        # Visualize the schema graph and save to file
        output_path = os.path.join(self.output_dir, "schema_graph.png")
        self.schema.visualize_schema_graph(output_path=output_path, show=False)
        
        print(f"Schema graph visualization saved to {output_path}")
    
    def export_import_schema(self):
        """Export and import the schema graph."""
        print("\n[6/7] Exporting and importing schema graph...")
        
        # Export schema graph
        export_path = os.path.join(self.output_dir, "schema_export.json")
        self.schema.export_schema_graph(export_path)
        print(f"Schema graph exported to {export_path}")
        
        # Create a new schema integration
        new_schema = MemorySchemaIntegration(self.sandbox)
        
        # Import schema graph
        new_schema.import_schema_graph(export_path)
        print("Schema graph successfully imported")
        
        # Verify the graphs match
        orig_nodes = len(self.schema.graph.nodes)
        new_nodes = len(new_schema.graph.nodes)
        
        print(f"Original graph: {orig_nodes} nodes")
        print(f"Imported graph: {new_nodes} nodes")
        
        if orig_nodes == new_nodes:
            print("✓ Graphs match exactly")
        else:
            print("✗ Graphs do not match")
    
    def generate_knowledge_report(self):
        """Generate a knowledge report."""
        print("\n[7/7] Generating knowledge report...")
        
        # Generate report
        report = self.schema.generate_knowledge_report()
        
        # Print report summary
        print("Knowledge report summary:")
        print(f"  - Total memories: {report['summary']['total_memories']}")
        print(f"  - Total concepts: {report['summary']['total_concepts']}")
        print(f"  - Total clusters: {report['summary']['total_clusters']}")
        
        # Print some concept analysis
        if report['concepts']:
            print("\nTop concepts by importance:")
            sorted_concepts = sorted(
                report['concepts'].items(), 
                key=lambda x: x[1]['importance'], 
                reverse=True
            )[:3]
            
            for _, concept in sorted_concepts:
                print(f"  - {concept['concept_name']}: {concept['importance']:.2f}")
        
        # Print stats
        if 'statistics' in report and 'memory_type_distribution' in report['statistics']:
            print("\nMemory type distribution:")
            for memory_type, count in report['statistics']['memory_type_distribution'].items():
                print(f"  - {memory_type}: {count}")
        
        # Save report to file
        report_path = os.path.join(self.output_dir, "knowledge_report.json")
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        print(f"Full knowledge report saved to {report_path}")
    
    def run_demo(self):
        """Run the full schema integration demo."""
        print("========================================")
        print("Memory Schema Integration Demo")
        print("========================================")
        print("This demo showcases the ability to build knowledge graphs from memories,")
        print("detect memory clusters, generate concepts, and analyze memory relationships.")
        
        # Run all demo steps
        self.create_test_memories()
        self.build_schema_graph()
        self.detect_clusters()
        self.generate_concepts()
        self.visualize_schema()
        self.export_import_schema()
        self.generate_knowledge_report()
        
        print("\n========================================")
        print("Schema Integration Demo Complete")
        print("========================================")
        print(f"Demo output files can be found in: {os.path.abspath(self.output_dir)}")


def run_demo():
    """Run the schema integration demo."""
    demo = SchemaIntegrationDemo()
    demo.run_demo()


if __name__ == "__main__":
    run_demo() 