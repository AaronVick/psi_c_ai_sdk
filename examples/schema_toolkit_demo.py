#!/usr/bin/env python3
"""
Schema Toolkit Demo

This script demonstrates how to use the Schema Toolkit to create, update, 
and visualize schema graphs from memories. It shows how to:
- Build a schema graph from a set of memories
- Visualize the schema graph
- Perform schema mutations
- Extract related memories and concepts
"""

import logging
import time
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

from psi_c_ai_sdk.memory.memory import Memory, MemoryType, MemoryStore
from psi_c_ai_sdk.coherence.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.embedding.engine import EmbeddingEngine
from psi_c_ai_sdk.schema.toolkit import SchemaToolkit
from psi_c_ai_sdk.schema.mutation import MutationType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('schema_demo')

# Create visualization directory
VIZ_DIR = "schema_visualizations"
if not os.path.exists(VIZ_DIR):
    os.makedirs(VIZ_DIR)

def create_sample_memories() -> List[Memory]:
    """Create a set of sample memories for the demo."""
    
    # Create memory categories with related content
    categories = {
        "technology": [
            "Artificial intelligence is transforming how we interact with computers.",
            "Machine learning models require large amounts of training data.",
            "Neural networks are inspired by the human brain's structure.",
            "Deep learning has revolutionized computer vision and natural language processing.",
            "Cloud computing allows for scalable and flexible resource allocation.",
            "Edge computing brings processing power closer to data sources.",
            "Quantum computing may solve problems impossible for classical computers.",
            "Blockchain technology provides a decentralized way to record transactions.",
            "Augmented reality overlays digital information on the physical world.",
            "Virtual reality creates immersive digital environments for users."
        ],
        "nature": [
            "Rainforests are home to more than half of the world's plant and animal species.",
            "The Great Barrier Reef is the world's largest coral reef system.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "Ecosystems maintain themselves through the balance of producer and consumer organisms.",
            "Climate change is affecting weather patterns across the globe.",
            "The water cycle describes how water moves through the Earth's systems.",
            "Biodiversity is essential for ecosystem resilience and stability.",
            "National parks help preserve natural habitats for future generations.",
            "Endangered species require conservation efforts to prevent extinction.",
            "Forests act as carbon sinks, absorbing carbon dioxide from the atmosphere."
        ],
        "history": [
            "The Renaissance was a period of cultural rebirth in Europe.",
            "The Industrial Revolution transformed manufacturing processes.",
            "World War II was the deadliest conflict in human history.",
            "The Ancient Egyptian civilization lasted for over 3,000 years.",
            "The Silk Road connected trade routes across Europe and Asia.",
            "The Declaration of Independence was signed in 1776.",
            "The fall of the Berlin Wall symbolized the end of the Cold War.",
            "The printing press revolutionized the spread of information.",
            "The Apollo 11 mission landed the first humans on the Moon.",
            "The Roman Empire expanded across Europe, Africa, and Asia."
        ],
        "mixed": [
            "Reading books improves vocabulary and critical thinking skills.",
            "Exercise helps maintain physical and mental health.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "Effective communication is essential for teamwork and collaboration.",
            "Different cultures have unique traditions and customs.",
            "The internet has transformed how people access information.",
            "Music can evoke powerful emotional responses in listeners.",
            "Sustainable agriculture practices help preserve natural resources.",
            "Learning a new language expands cognitive abilities.",
            "Architecture reflects the values and technology of a society."
        ]
    }
    
    # Create memories with appropriate metadata
    memories = []
    base_time = datetime.now() - timedelta(days=30)
    
    for category, contents in categories.items():
        for i, content in enumerate(contents):
            # Create memory with incremental timestamp
            created_at = base_time + timedelta(days=i, hours=random.randint(0, 12))
            
            # Generate tags based on content
            words = content.lower().replace(".", "").replace(",", "").split()
            potential_tags = [word for word in words if len(word) > 5]
            tags = random.sample(potential_tags, min(3, len(potential_tags)))
            
            # Set importance based on position in list (later items more important)
            importance = 0.4 + (i / len(contents)) * 0.5
            
            memory = Memory(
                uuid=uuid.uuid4(),
                content=content,
                memory_type=MemoryType.DECLARATIVE,
                importance=importance,
                created_at=created_at,
                tags=tags,
                metadata={
                    "category": category,
                    "source": f"demo_{category}",
                    "confidence": random.uniform(0.7, 0.95)
                }
            )
            memories.append(memory)
    
    return memories

def main():
    """Run the schema toolkit demo."""
    
    logger.info("Initializing components...")
    
    # Initialize components
    memory_store = MemoryStore()
    embedding_engine = EmbeddingEngine()
    coherence_scorer = BasicCoherenceScorer()
    
    # Create Schema Toolkit
    schema_toolkit = SchemaToolkit(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer,
        min_edge_weight=0.3,
        mutation_threshold=0.5,
        auto_prune=True,
        max_nodes=100,
        visualize_dir=VIZ_DIR
    )
    
    # Create sample memories
    logger.info("Creating sample memories...")
    memories = create_sample_memories()
    
    # Add memories to store and schema
    logger.info("Adding memories to schema...")
    for i, memory in enumerate(memories):
        # Generate embedding for memory
        embedding = embedding_engine.get_embedding(memory.content)
        memory.embedding = embedding
        
        # Add to memory store
        memory_store.add_memory(memory)
        
        # Add to schema - disable auto-update for most memories for performance
        update_schema = (i % 10 == 0)
        check_mutation = (i % 10 == 0)
        node_id, mutation_info = schema_toolkit.add_memory(
            memory,
            update_schema=update_schema,
            check_mutation=check_mutation
        )
        
        if mutation_info:
            logger.info(f"Mutation occurred: {mutation_info['mutation_type']}")
    
    # Visualize initial schema
    logger.info("Visualizing initial schema...")
    initial_viz = schema_toolkit.visualize_schema(
        filename=os.path.join(VIZ_DIR, "initial_schema.png"),
        max_nodes=50
    )
    logger.info(f"Initial schema visualization saved to {initial_viz}")
    
    # Update the schema (process all memories)
    logger.info("Updating schema with all memories...")
    update_stats = schema_toolkit.update_schema(max_memories=len(memories))
    logger.info(f"Schema update stats: {update_stats}")
    
    # Get schema stats
    stats = schema_toolkit.get_stats()
    logger.info(f"Schema graph has {stats['graph']['node_count']} nodes and {stats['graph']['edge_count']} edges")
    
    # Visualize updated schema
    logger.info("Visualizing updated schema...")
    updated_viz = schema_toolkit.visualize_schema(
        filename=os.path.join(VIZ_DIR, "updated_schema.png"),
        max_nodes=50
    )
    
    # Perform various mutations
    logger.info("Performing schema mutations...")
    
    # Try each mutation type
    mutation_types = [
        MutationType.MERGE.value,
        MutationType.SPLIT.value,
        MutationType.ADD_CONCEPT.value,
        MutationType.CONSOLIDATE.value,
        MutationType.RESTRUCTURE.value
    ]
    
    for mutation_type in mutation_types:
        logger.info(f"Attempting {mutation_type} mutation...")
        mutation_result = schema_toolkit.mutate_schema(mutation_type=mutation_type)
        
        if mutation_result:
            logger.info(f"Mutation {mutation_type} successful")
            
            # Visualize after each successful mutation
            viz_path = os.path.join(VIZ_DIR, f"after_{mutation_type}_mutation.png")
            schema_toolkit.visualize_schema(
                filename=viz_path,
                max_nodes=50,
                highlight_nodes=mutation_result.get("affected_nodes", [])
            )
        else:
            logger.info(f"Mutation {mutation_type} not applicable or failed")
    
    # Add manual concept
    logger.info("Adding manual concept node...")
    tech_memories = [m for m in memories if m.metadata.get("category") == "technology"]
    
    tech_concept_id = schema_toolkit.add_concept(
        label="Technology Concept",
        importance=0.8,
        tags=["technology", "computing", "digital"],
        metadata={"manually_created": True},
        related_memories=tech_memories[:5]
    )
    
    logger.info(f"Created technology concept node: {tech_concept_id}")
    
    # Visualize with highlighted concept
    schema_toolkit.visualize_schema(
        filename=os.path.join(VIZ_DIR, "with_concept_highlighted.png"),
        highlight_nodes=[tech_concept_id],
        max_nodes=50
    )
    
    # Get related memories for a specific memory
    sample_memory = random.choice(memories)
    logger.info(f"Finding memories related to: {sample_memory.content[:50]}...")
    
    related = schema_toolkit.get_related_memories(
        memory=sample_memory,
        min_coherence=0.4,
        max_results=5
    )
    
    logger.info(f"Found {len(related)} related memories:")
    for i, rel in enumerate(related):
        logger.info(f"{i+1}. {rel['content'][:50]}... (coherence: {rel['coherence']:.2f})")
    
    # Get concept hierarchy
    hierarchy = schema_toolkit.get_concept_hierarchy()
    logger.info(f"Schema has {len(hierarchy['concepts'])} concept nodes")
    
    # Calculate schema fingerprint
    fingerprint = schema_toolkit.get_schema_fingerprint()
    logger.info(f"Schema fingerprint: {fingerprint}")
    
    # Get mutation history
    mutation_history = schema_toolkit.get_mutation_history()
    logger.info(f"Schema underwent {len(mutation_history)} mutations")
    
    # Final visualization
    logger.info("Creating final schema visualization...")
    final_viz = schema_toolkit.visualize_schema(
        filename=os.path.join(VIZ_DIR, "final_schema.png"),
        max_nodes=75
    )
    
    logger.info("Schema Toolkit demo completed successfully!")
    logger.info(f"Visualizations saved to {VIZ_DIR} directory")

if __name__ == "__main__":
    main() 