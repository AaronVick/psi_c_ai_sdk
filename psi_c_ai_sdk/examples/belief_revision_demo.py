#!/usr/bin/env python3
"""
Belief Revision System Demo

This script demonstrates the functionality of the Belief Revision System in the ΨC-AI SDK.
It shows how the system:
1. Detects contradictions between memories
2. Resolves contradictions using weighted arbitration
3. Updates trust levels for memory sources
4. Tracks decision history and provides statistics

The demo creates a set of contradictory memories and shows how the system arbitrates between them.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

from psi_c_ai_sdk.beliefs.revision import BeliefRevisionSystem
from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.memory.embedding import EmbeddingModel
from psi_c_ai_sdk.memory.coherence import CoherenceScorer
from psi_c_ai_sdk.beliefs.contradiction import ContradictionDetector
from psi_c_ai_sdk.util.logging import setup_logger

# Set up logging
logger = setup_logger("belief_revision_demo", level=logging.INFO)

def create_test_memories() -> List[Memory]:
    """
    Create a set of test memories with contradictory information.
    
    Returns:
        List of Memory objects with some contradictory content
    """
    # Base time for memory creation
    base_time = datetime.now() - timedelta(days=7)
    
    memories = []
    
    # Group 1: Weather contradictions
    memories.append(Memory(
        content="It was sunny all day yesterday.",
        source="weather_app",
        creation_time=base_time,
        importance=0.7,
        metadata={"confidence": 0.8, "category": "weather"}
    ))
    
    memories.append(Memory(
        content="It rained heavily yesterday afternoon.",
        source="personal_observation",
        creation_time=base_time + timedelta(hours=3),
        importance=0.6,
        metadata={"confidence": 0.9, "category": "weather"}
    ))
    
    memories.append(Memory(
        content="Yesterday had clear skies with no precipitation.",
        source="weather_service",
        creation_time=base_time + timedelta(hours=1),
        importance=0.75,
        metadata={"confidence": 0.85, "category": "weather"}
    ))
    
    # Group 2: Factual contradictions
    memories.append(Memory(
        content="Paris is the capital of Italy.",
        source="casual_conversation",
        creation_time=base_time - timedelta(days=2),
        importance=0.5,
        metadata={"confidence": 0.6, "category": "geography"}
    ))
    
    memories.append(Memory(
        content="Rome is the capital of Italy.",
        source="geography_book",
        creation_time=base_time - timedelta(days=3),
        importance=0.8,
        metadata={"confidence": 0.95, "category": "geography"}
    ))
    
    # Group 3: Preference contradictions
    memories.append(Memory(
        content="The user prefers dark mode interfaces.",
        source="user_settings",
        creation_time=base_time - timedelta(days=1),
        importance=0.85,
        metadata={"confidence": 0.9, "category": "preferences"}
    ))
    
    memories.append(Memory(
        content="The user mentioned they like bright, light-colored interfaces.",
        source="conversation",
        creation_time=base_time - timedelta(hours=12),
        importance=0.7,
        metadata={"confidence": 0.75, "category": "preferences"}
    ))
    
    # Group 4: Scheduling contradictions
    memories.append(Memory(
        content="The meeting is scheduled for 2 PM on Tuesday.",
        source="calendar_initial",
        creation_time=base_time - timedelta(days=5),
        importance=0.8,
        metadata={"confidence": 0.9, "category": "schedule"}
    ))
    
    memories.append(Memory(
        content="The Tuesday meeting was rescheduled to 3 PM.",
        source="calendar_update",
        creation_time=base_time - timedelta(days=2),
        importance=0.85,
        metadata={"confidence": 0.95, "category": "schedule"}
    ))
    
    # Add some non-contradictory memories
    memories.append(Memory(
        content="The user likes to go for a walk in the morning.",
        source="conversation",
        creation_time=base_time - timedelta(days=4),
        importance=0.6,
        metadata={"confidence": 0.8, "category": "habits"}
    ))
    
    memories.append(Memory(
        content="The user's favorite color is blue.",
        source="questionnaire",
        creation_time=base_time - timedelta(days=10),
        importance=0.5,
        metadata={"confidence": 0.9, "category": "preferences"}
    ))
    
    return memories

def initialize_system() -> Tuple[BeliefRevisionSystem, MemoryStore, ContradictionDetector]:
    """
    Initialize the belief revision system with necessary components.
    
    Returns:
        Tuple containing (BeliefRevisionSystem, MemoryStore, ContradictionDetector)
    """
    # Initialize components
    embedding_model = EmbeddingModel()
    memory_store = MemoryStore(embedding_model)
    coherence_scorer = CoherenceScorer()
    
    # Create contradiction detector with some simple patterns
    contradiction_detector = ContradictionDetector()
    contradiction_detector.add_contradiction_pattern(
        "It was sunny", "It rained", "Weather contradiction"
    )
    contradiction_detector.add_contradiction_pattern(
        "Paris is the capital", "Rome is the capital", "Geography contradiction"
    )
    contradiction_detector.add_contradiction_pattern(
        "prefers dark mode", "like bright, light-colored", "Preference contradiction"
    )
    contradiction_detector.add_contradiction_pattern(
        "scheduled for 2 PM", "rescheduled to 3 PM", "Schedule contradiction"
    )
    
    # Initialize belief revision system with custom weights
    belief_revision = BeliefRevisionSystem(
        memory_store=memory_store,
        contradiction_detector=contradiction_detector,
        coherence_scorer=coherence_scorer,
        weights={
            "trust": 0.3,
            "recency": 0.25,
            "coherence": 0.25,
            "importance": 0.2
        },
        trust_sources={
            "weather_app": 0.7,
            "personal_observation": 0.85,
            "weather_service": 0.8,
            "casual_conversation": 0.4,
            "geography_book": 0.9,
            "user_settings": 0.95,
            "conversation": 0.6,
            "calendar_initial": 0.7,
            "calendar_update": 0.85,
            "questionnaire": 0.8
        }
    )
    
    return belief_revision, memory_store, contradiction_detector

def print_trust_levels(belief_system: BeliefRevisionSystem) -> None:
    """Print current trust levels for all sources."""
    logger.info("\n--- Current Trust Levels ---")
    for source, trust in sorted(belief_system.trust_sources.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{source}: {trust:.2f}")

def print_contradiction_resolution(memory1: Memory, memory2: Memory, decision: Dict[str, Any]) -> None:
    """Print details about a contradiction resolution."""
    logger.info("\n--- Contradiction Resolution ---")
    logger.info(f"Memory 1: \"{memory1.content}\" (Source: {memory1.source})")
    logger.info(f"Memory 2: \"{memory2.content}\" (Source: {memory2.source})")
    logger.info(f"Decision: {decision['decision']}")
    
    if decision["scores"]:
        logger.info("\nScores:")
        for memory_id, score_details in decision["scores"].items():
            memory = memory1 if memory_id == memory1.id else memory2
            logger.info(f"\n{memory.source} ({memory_id}):")
            for factor, score in score_details.items():
                logger.info(f"  {factor}: {score:.2f}")

def run_demo() -> None:
    """Run the belief revision system demonstration."""
    logger.info("Starting Belief Revision System Demo")
    
    # Initialize the system
    belief_system, memory_store, contradiction_detector = initialize_system()
    
    # Create and add test memories
    memories = create_test_memories()
    for memory in memories:
        memory_store.add(memory)
    
    logger.info(f"Added {len(memories)} memories to the store")
    
    # Print initial trust levels
    print_trust_levels(belief_system)
    
    # Step 1: Find all contradictions
    logger.info("\nStep 1: Finding contradictions in memories...")
    contradictions = belief_system.find_contradictions_in_store()
    logger.info(f"Found {len(contradictions)} contradictions")
    
    # Step 2: Resolve contradictions one by one
    logger.info("\nStep 2: Resolving contradictions one by one...")
    
    for i, (memory1, memory2) in enumerate(contradictions):
        logger.info(f"\nContradiction {i+1}/{len(contradictions)}:")
        decision = belief_system.arbitrate_contradiction(memory1, memory2)
        print_contradiction_resolution(memory1, memory2, decision)
        
        # Update trust levels based on the decision
        belief_system.update_trust_based_on_decision(decision)
    
    # Step 3: Get updated trust levels
    logger.info("\nStep 3: Updated trust levels after contradiction resolution")
    print_trust_levels(belief_system)
    
    # Step 4: Use the batch method to process all contradictions
    logger.info("\nStep 4: Using batch method to find and resolve all contradictions")
    logger.info("Resetting memory store and trust levels for demonstration...")
    
    # Reset memory store and trust levels
    memory_store.clear()
    belief_system.reset_trust_levels()
    
    # Add memories again
    for memory in memories:
        memory_store.add(memory)
    
    # Run the batch method
    num_resolved = belief_system.find_and_resolve_contradictions(limit=5)
    logger.info(f"Resolved {num_resolved} contradictions in batch mode")
    
    # Step 5: Show decision history
    logger.info("\nStep 5: Decision history from batch processing")
    for i, decision in enumerate(belief_system.decision_history):
        logger.info(f"\nDecision {i+1}:")
        logger.info(f"Memory 1: \"{decision['memory1_content']}\" (Source: {decision['memory1_source']})")
        logger.info(f"Memory 2: \"{decision['memory2_content']}\" (Source: {decision['memory2_source']})")
        logger.info(f"Decision: {decision['decision']}")
        logger.info(f"Timestamp: {decision['timestamp']}")
        
    # Step 6: Get updated trust levels after batch processing
    logger.info("\nStep 6: Final trust levels after batch processing")
    print_trust_levels(belief_system)
    
    logger.info("\nBelief Revision System Demo completed successfully")

if __name__ == "__main__":
    run_demo() 