"""
Justification Engine Demo: Demonstrates the belief explanation and source tracing capabilities.

This example shows how to:
1. Generate explanations for beliefs and memories
2. Visualize ancestry chains to understand how memories evolved
3. Trace the sources of information to understand where knowledge came from
"""

import logging
import os
import time
from typing import List, Dict, Any

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.embedding.engine import BasicEmbeddingEngine
from psi_c_ai_sdk.coherence.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.contradiction.detector import ContradictionDetector
from psi_c_ai_sdk.beliefs.revision import BeliefRevisionSystem, DecisionStrategy
from psi_c_ai_sdk.justification.justification_engine import (
    JustificationEngine, ExplanationFormat, SourceReference
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_memories() -> List[Dict[str, Any]]:
    """Create a set of test memories with different sources and relationships."""
    return [
        {
            "content": "Paris is the capital of France.",
            "source": "geography_textbook",
            "source_type": "document",
            "trust_level": 0.9
        },
        {
            "content": "The Eiffel Tower is located in Paris, France.",
            "source": "travel_guide",
            "source_type": "document",
            "trust_level": 0.8
        },
        {
            "content": "The Seine River flows through Paris.",
            "source": "geography_textbook",
            "source_type": "document",
            "trust_level": 0.9
        },
        {
            "content": "Paris has a population of approximately 2.2 million people.",
            "source": "wikipedia",
            "source_type": "web",
            "trust_level": 0.7
        },
        {
            "content": "Paris is known as the 'City of Light'.",
            "source": "travel_guide",
            "source_type": "document",
            "trust_level": 0.8
        },
        {
            "content": "The Mona Lisa is displayed in the Louvre Museum in Paris.",
            "source": "art_encyclopedia",
            "source_type": "document",
            "trust_level": 0.85
        },
        {
            "content": "Paris is located in northern France.",
            "source": "geography_textbook",
            "source_type": "document",
            "trust_level": 0.9
        },
        {
            "content": "Berlin is the capital of Germany.",
            "source": "geography_textbook",
            "source_type": "document",
            "trust_level": 0.9
        },
        {
            "content": "Madrid is the capital of Spain.",
            "source": "geography_textbook",
            "source_type": "document",
            "trust_level": 0.9
        },
        # Add a contradiction
        {
            "content": "Paris has a population of 3 million people.",
            "source": "outdated_almanac",
            "source_type": "document",
            "trust_level": 0.5
        }
    ]


def create_memory_store() -> MemoryStore:
    """Create and populate a memory store with test memories."""
    # Create components
    embedding_engine = BasicEmbeddingEngine(dim=64)  # Smaller dim for faster demo
    memory_store = MemoryStore(embedding_engine=embedding_engine)
    
    # Add memories
    memory_ids = []
    test_memories = create_test_memories()
    
    for i, mem_data in enumerate(test_memories):
        # Create memory
        memory = Memory(
            content=mem_data["content"],
            metadata={
                "source": mem_data["source"],
                "source_type": mem_data["source_type"],
                "trust_level": mem_data["trust_level"]
            }
        )
        
        # Add to memory store
        memory_id = memory_store.add_memory(memory)
        memory_ids.append(memory_id)
        
        logger.info(f"Added memory {i+1}: {mem_data['content']} (ID: {memory_id})")
    
    return memory_store


def demonstrate_belief_explanation(justification_engine: JustificationEngine, memory_id: str) -> None:
    """Demonstrate the belief explanation capabilities."""
    logger.info("\n=== DEMONSTRATING BELIEF EXPLANATION ===")
    
    # Get the memory
    memory = justification_engine.memory_store.get_memory_by_id(memory_id)
    if not memory:
        logger.error(f"Memory with ID {memory_id} not found")
        return
        
    logger.info(f"Explaining belief: '{memory.content}'")
    
    # Get explanation in text format
    explanation_text = justification_engine.explain_memory(
        memory_id, 
        format=ExplanationFormat.TEXT
    )
    logger.info(f"\nExplanation (TEXT format):\n{explanation_text}")
    
    # Get explanation in JSON format
    explanation_json = justification_engine.explain_memory(
        memory_id, 
        format=ExplanationFormat.JSON
    )
    logger.info(f"\nExplanation (JSON format - keys only): {list(explanation_json.keys())}")
    
    # Visualize the explanation
    visualization_path = "explanation_visualization.png"
    justification_engine.explain_memory(
        memory_id, 
        format=ExplanationFormat.VISUALIZATION,
        visualization_path=visualization_path
    )
    logger.info(f"Visualization saved to: {visualization_path}")


def demonstrate_ancestry_tracking(
    justification_engine: JustificationEngine, 
    memory_store: MemoryStore
) -> None:
    """Demonstrate the ancestry tracking capabilities."""
    logger.info("\n=== DEMONSTRATING ANCESTRY TRACKING ===")
    
    # Create a derived memory from existing memories
    paris_memories = []
    for memory in memory_store.get_all_memories():
        if "Paris" in memory.content:
            paris_memories.append(memory)
    
    # Create a derived memory
    derived_content = "Paris is a major cultural center in Europe with famous landmarks like the Eiffel Tower and Louvre Museum."
    derived_memory = Memory(
        content=derived_content,
        metadata={
            "derived": True,
            "operation": "merge",
            "parent_count": len(paris_memories)
        }
    )
    
    derived_id = memory_store.add_memory(derived_memory)
    logger.info(f"Created derived memory: {derived_content} (ID: {derived_id})")
    
    # Register the ancestry relationship
    parent_ids = [mem.id for mem in paris_memories]
    source_refs = [
        SourceReference(
            source_id="reflection_engine",
            source_type="system",
            source_name="Reflection Engine",
            trust_level=0.8
        )
    ]
    
    justification_engine.register_memory_operation(
        memory_id=derived_id,
        operation="merge",
        parent_ids=parent_ids,
        source_references=source_refs
    )
    
    logger.info(f"Registered ancestry relationship with {len(parent_ids)} parent memories")
    
    # Visualize the ancestry
    ancestry_path = "ancestry_visualization.png"
    justification_engine.visualize_ancestry(
        memory_id=derived_id,
        save_path=ancestry_path
    )
    logger.info(f"Ancestry visualization saved to: {ancestry_path}")
    
    # Get the ancestry chain as text
    chain = justification_engine.get_ancestry_chain(derived_id)
    logger.info(f"Ancestry chain for derived memory ({len(chain)} nodes):")
    for i, node in enumerate(chain):
        logger.info(f"  {i+1}. {node.content[:50]}...")
    
    return derived_id


def demonstrate_source_tracing(justification_engine: JustificationEngine) -> None:
    """Demonstrate the source tracing capabilities."""
    logger.info("\n=== DEMONSTRATING SOURCE TRACING ===")
    
    # Register some sources
    sources = [
        SourceReference(
            source_id="geography_textbook",
            source_type="document",
            source_name="World Geography Textbook (2022)",
            trust_level=0.9,
            metadata={"publisher": "Academic Press", "year": 2022}
        ),
        SourceReference(
            source_id="travel_guide",
            source_type="document",
            source_name="Europe Travel Guide",
            trust_level=0.8,
            metadata={"publisher": "Travel Media", "year": 2023}
        ),
        SourceReference(
            source_id="wikipedia",
            source_type="web",
            source_name="Wikipedia",
            trust_level=0.7,
            metadata={"url": "https://wikipedia.org"}
        ),
        SourceReference(
            source_id="art_encyclopedia",
            source_type="document",
            source_name="Encyclopedia of Art History",
            trust_level=0.85,
            metadata={"publisher": "Arts Publishing", "year": 2020}
        ),
        SourceReference(
            source_id="outdated_almanac",
            source_type="document",
            source_name="World Almanac 2010",
            trust_level=0.5,
            metadata={"publisher": "Reference Publishing", "year": 2010}
        )
    ]
    
    # Register sources in the engine
    for source in sources:
        justification_engine.source_registry[source.source_id] = source
    
    logger.info(f"Registered {len(sources)} sources")
    
    # Trace sources for all memories
    logger.info("\nSource analysis for memories:")
    
    # Group memories by source
    memories_by_source = {}
    for memory in justification_engine.memory_store.get_all_memories():
        if "source" in memory.metadata:
            source_id = memory.metadata["source"]
            if source_id not in memories_by_source:
                memories_by_source[source_id] = []
            memories_by_source[source_id].append(memory)
    
    # Print source analysis
    for source_id, memories in memories_by_source.items():
        source = justification_engine.source_registry.get(source_id)
        source_name = source.source_name if source else source_id
        trust_level = source.trust_level if source else "unknown"
        
        logger.info(f"\nSource: {source_name} (Trust level: {trust_level})")
        logger.info(f"Contributed {len(memories)} memories:")
        
        for i, memory in enumerate(memories, 1):
            logger.info(f"  {i}. {memory.content}")


def demonstrate_contradiction_resolution(
    justification_engine: JustificationEngine,
    belief_system: BeliefRevisionSystem
) -> None:
    """Demonstrate resolving contradictions and explaining decisions."""
    logger.info("\n=== DEMONSTRATING CONTRADICTION RESOLUTION ===")
    
    # Find contradictions
    all_memories = justification_engine.memory_store.get_all_memories()
    contradictions = belief_system.contradiction_detector.find_contradictions(all_memories, all_memories)
    
    if not contradictions:
        logger.info("No contradictions found in test data.")
        return
        
    logger.info(f"Found {len(contradictions)} contradiction pairs")
    
    # Process each contradiction
    for i, (mem1, mem2) in enumerate(contradictions, 1):
        logger.info(f"\nContradiction {i}:")
        logger.info(f"  Memory 1: '{mem1.content}'")
        logger.info(f"  Memory 2: '{mem2.content}'")
        
        # Resolve the contradiction
        logger.info("Resolving contradiction...")
        decision = belief_system.arbitrate_contradiction([mem1], [mem2])
        
        # Log the decision
        if decision:
            logger.info(f"Decision: {decision.outcome}")
            logger.info(f"Reason: {decision.reason}")
            
            # Get explanation for the surviving memory
            surviving_id = decision.kept_memory_ids[0] if decision.kept_memory_ids else None
            if surviving_id:
                explanation = justification_engine.explain_memory(surviving_id)
                logger.info(f"\nExplanation for surviving belief:\n{explanation}")


def main():
    """Run the justification engine demo."""
    logger.info("Starting Justification Engine Demo")
    
    # Create necessary directories
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Create components
    embedding_engine = BasicEmbeddingEngine(dim=64)  # Smaller dim for faster demo
    memory_store = create_memory_store()
    coherence_scorer = BasicCoherenceScorer(memory_store=memory_store)
    contradiction_detector = ContradictionDetector(embedding_engine=embedding_engine)
    
    # Create belief revision system
    belief_system = BeliefRevisionSystem(
        memory_store=memory_store,
        contradiction_detector=contradiction_detector,
        coherence_scorer=coherence_scorer,
        decision_log_path="output/belief_decisions.log"
    )
    
    # Create the justification engine
    justification_engine = JustificationEngine(
        memory_store=memory_store,
        belief_system=belief_system,
        coherence_scorer=coherence_scorer,
        ancestry_tracker_enabled=True,
        source_tracker_enabled=True,
        explanation_log_path="output/explanations.log",
        visualization_dir="output"
    )
    
    # Choose a memory to explain
    all_memories = memory_store.get_all_memories()
    paris_capital_memory = None
    for memory in all_memories:
        if "Paris is the capital" in memory.content:
            paris_capital_memory = memory
            break
    
    if not paris_capital_memory:
        logger.error("Could not find Paris capital memory")
        return
        
    # Run the demonstrations
    demonstrate_belief_explanation(justification_engine, paris_capital_memory.id)
    derived_memory_id = demonstrate_ancestry_tracking(justification_engine, memory_store)
    demonstrate_source_tracing(justification_engine)
    demonstrate_contradiction_resolution(justification_engine, belief_system)
    
    logger.info("\nJustification Engine Demo completed successfully!")


if __name__ == "__main__":
    main() 