#!/usr/bin/env python3
"""
Contradiction Detection Demo for ΨC-AI SDK

This demonstration shows how the contradiction detection system identifies
different types of contradictions between memories, including direct negations,
property conflicts, temporal inconsistencies, numerical discrepancies, and
categorical contradictions.
"""

import logging
import sys
from uuid import uuid4
from datetime import datetime, timedelta

from psi_c_ai_sdk.memory.memory import Memory, MemoryType
from psi_c_ai_sdk.beliefs.contradiction import ContradictionDetector, ContradictionType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def create_memory(text, importance=0.5, memory_type=MemoryType.DECLARATIVE):
    """Create a memory with the given text."""
    return Memory(
        id=str(uuid4()),
        text=text,
        embedding=None,  # Not needed for this demo
        timestamp=datetime.now(),
        importance=importance,
        memory_type=memory_type,
        metadata={}
    )


def run_demo():
    """Run the contradiction detection demo."""
    logger.info("Starting Contradiction Detection Demo")
    
    # Create a contradiction detector
    detector = ContradictionDetector(
        semantic_threshold=0.6,
        entity_match_threshold=0.7
    )
    
    # Create test memories with different types of contradictions
    memories = [
        # Direct negation contradiction
        create_memory("The sky is blue."),
        create_memory("The sky is not blue."),
        
        # Property conflict
        create_memory("The car is red."),
        create_memory("The car is blue."),
        
        # Temporal contradiction
        create_memory("John was in New York yesterday."),
        create_memory("John was in London yesterday."),
        
        # Numerical contradiction
        create_memory("The building has 10 floors."),
        create_memory("The building has 20 floors."),
        
        # Categorical contradiction
        create_memory("A dolphin is a fish."),
        create_memory("A dolphin is a mammal."),
        
        # Non-contradictory memories (should not be detected)
        create_memory("The tree is tall."),
        create_memory("The tree has green leaves."),
        
        create_memory("Paris is the capital of France."),
        create_memory("Rome is the capital of Italy.")
    ]
    
    logger.info(f"Created {len(memories)} test memories")
    
    # Find contradictions
    contradictions = detector.find_contradictions(memories)
    
    logger.info(f"Found {len(contradictions)} contradictions")
    
    # Display results
    print("\n===== CONTRADICTION DETECTION RESULTS =====\n")
    
    if not contradictions:
        print("No contradictions detected.")
    else:
        for i, (mem1, mem2, confidence) in enumerate(contradictions, 1):
            print(f"Contradiction #{i} (Confidence: {confidence:.2f})")
            print(f"  Memory 1: \"{mem1.text}\"")
            print(f"  Memory 2: \"{mem2.text}\"")
            print()
    
    # Show statistics
    stats = detector.get_stats()
    print("\n===== DETECTION STATISTICS =====\n")
    print(f"Total comparisons: {stats['total_comparisons']}")
    print(f"Contradictions found: {stats['contradictions_found']}")
    print("\nBreakdown by contradiction type:")
    for ctype, count in stats['by_type'].items():
        if count > 0:
            print(f"  {ctype}: {count}")
    
    print("\n===== DEMO COMPLETED =====\n")


if __name__ == "__main__":
    run_demo() 