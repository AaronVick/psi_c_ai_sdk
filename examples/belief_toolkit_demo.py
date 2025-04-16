#!/usr/bin/env python3
"""
Belief Toolkit Demo

This script demonstrates how to use the Belief Toolkit to manage contradictions
in a memory system, including:
- Setting up a memory store and belief toolkit
- Adding memories with different sources and trust levels
- Detecting and resolving contradictions
- Analyzing decisions and resolution outcomes
"""

import logging
import time
from datetime import datetime, timedelta
import random

from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.embedding.engine import EmbeddingEngine
from psi_c_ai_sdk.coherence.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.beliefs.toolkit import BeliefToolkit
from psi_c_ai_sdk.beliefs.revision import TrustLevel, DecisionOutcome
from psi_c_ai_sdk.beliefs.contradiction import ContradictionType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("belief_toolkit_demo")

def create_test_memories():
    """Create a set of test memories, some with contradictions."""
    
    # Different categories of memories with contradictions
    memories = []
    
    # Facts about animals (with contradictions)
    memories.extend([
        {
            "content": "Lions are the largest big cats in Africa.",
            "source": "nature_guide",
            "trust_level": TrustLevel.HIGH,
            "created_at": datetime.now() - timedelta(days=30)
        },
        {
            "content": "Tigers are the largest big cats in the world.",
            "source": "wildlife_book",
            "trust_level": TrustLevel.HIGH,
            "created_at": datetime.now() - timedelta(days=20)
        },
        {
            "content": "Elephants are not afraid of mice.",
            "source": "research_paper",
            "trust_level": TrustLevel.VERIFIED,
            "created_at": datetime.now() - timedelta(days=10)
        },
        {
            "content": "Elephants are afraid of mice.",
            "source": "old_folklore",
            "trust_level": TrustLevel.LOW,
            "created_at": datetime.now() - timedelta(days=100)
        }
    ])
    
    # Facts about science (with contradictions)
    memories.extend([
        {
            "content": "The Earth orbits around the Sun.",
            "source": "astronomy_textbook",
            "trust_level": TrustLevel.VERIFIED,
            "created_at": datetime.now() - timedelta(days=5)
        },
        {
            "content": "The Sun orbits around the Earth.",
            "source": "ancient_text",
            "trust_level": TrustLevel.UNTRUSTED,
            "created_at": datetime.now() - timedelta(days=1000)
        },
        {
            "content": "Water boils at 100 degrees Celsius at sea level.",
            "source": "physics_textbook",
            "trust_level": TrustLevel.HIGH,
            "created_at": datetime.now() - timedelta(days=15)
        },
        {
            "content": "Water boils at 212 degrees Fahrenheit at sea level.",
            "source": "us_textbook",
            "trust_level": TrustLevel.HIGH,
            "created_at": datetime.now() - timedelta(days=25)
        }
    ])
    
    # Temporal contradictions
    memories.extend([
        {
            "content": "The meeting will be held on Monday at 10 AM.",
            "source": "initial_schedule",
            "trust_level": TrustLevel.MEDIUM,
            "created_at": datetime.now() - timedelta(days=7)
        },
        {
            "content": "The meeting has been rescheduled to Tuesday at 2 PM.",
            "source": "calendar_update",
            "trust_level": TrustLevel.MEDIUM,
            "created_at": datetime.now() - timedelta(days=2)
        }
    ])
    
    # Non-contradictory memories for context
    memories.extend([
        {
            "content": "The Pacific Ocean is the largest ocean on Earth.",
            "source": "geography_book",
            "trust_level": TrustLevel.HIGH,
            "created_at": datetime.now() - timedelta(days=60)
        },
        {
            "content": "Humans have 23 pairs of chromosomes.",
            "source": "biology_textbook",
            "trust_level": TrustLevel.HIGH,
            "created_at": datetime.now() - timedelta(days=45)
        },
        {
            "content": "Mount Everest is the tallest mountain above sea level.",
            "source": "geography_book",
            "trust_level": TrustLevel.HIGH,
            "created_at": datetime.now() - timedelta(days=50)
        }
    ])
    
    return memories

def main():
    """Main function to run the belief toolkit demo."""
    logger.info("Starting Belief Toolkit Demo")
    
    # Initialize components
    memory_store = MemoryStore()
    embedding_engine = EmbeddingEngine()
    coherence_scorer = BasicCoherenceScorer()
    
    # Create belief toolkit
    belief_toolkit = BeliefToolkit(
        memory_store=memory_store,
        embedding_engine=embedding_engine,
        coherence_scorer=coherence_scorer,
        coherence_weight=0.3,
        trust_weight=0.4,  # Emphasize trust
        recency_weight=0.2,
        entropy_weight=0.1,
        decision_log_path="belief_decisions.log"
    )
    
    # Add custom contradiction patterns
    belief_toolkit.add_pattern(
        name="time_contradiction",
        pattern=r"(will be|is scheduled for|is going to be) .* (at|on) (.+)",
        contradiction_type=ContradictionType.TEMPORAL,
        confidence=0.7
    )
    
    belief_toolkit.add_pattern(
        name="measurement_contradiction",
        pattern=r"(\d+) (degrees|meters|feet|pounds|kilograms)",
        contradiction_type=ContradictionType.FACTUAL,
        confidence=0.8
    )
    
    # Add common antonym pairs
    belief_toolkit.add_antonym_pair("afraid", "unafraid")
    belief_toolkit.add_antonym_pair("largest", "smallest")
    
    # Get test memories
    memories = create_test_memories()
    
    # First, add all non-contradictory memories with contradiction checking disabled
    logger.info("Adding base memories...")
    for memory in memories[:5]:
        memory_id, _ = belief_toolkit.add_memory(
            content=memory["content"],
            source=memory["source"],
            trust_level=memory["trust_level"],
            check_contradictions=False,
            metadata={"created_at": memory["created_at"].isoformat()}
        )
        logger.info(f"Added memory: {memory['content']}")
    
    # Then add potentially contradictory memories one by one with contradiction checking
    logger.info("\nAdding potentially contradictory memories...")
    for memory in memories[5:]:
        logger.info(f"Adding: {memory['content']}")
        memory_id, decisions = belief_toolkit.add_memory(
            content=memory["content"],
            source=memory["source"],
            trust_level=memory["trust_level"],
            check_contradictions=True,
            metadata={"created_at": memory["created_at"].isoformat()}
        )
        
        if decisions:
            logger.info(f"Found and resolved {len(decisions)} contradictions")
            for decision in decisions:
                kept_content = decision.get("memory1_content" 
                    if decision["outcome"] == DecisionOutcome.KEEP_FIRST.name 
                    else "memory2_content")
                
                logger.info(f"Resolution: {decision['outcome']}")
                logger.info(f"Kept: \"{kept_content}\"")
                logger.info(f"Scores: {decision['weighted_score1']} vs {decision['weighted_score2']}\n")
        else:
            logger.info("No contradictions found")
    
    # Get all remaining memories
    remaining_memories = memory_store.get_all_memories()
    logger.info(f"\nRemaining memories after contradiction resolution: {len(remaining_memories)}")
    for memory in remaining_memories:
        logger.info(f"- {memory.content}")
    
    # Show statistics
    stats = belief_toolkit.get_stats()
    logger.info("\nContradiction Statistics:")
    logger.info(f"Contradictions found: {stats['contradictions_found']}")
    logger.info(f"Contradictions resolved: {stats['contradictions_resolved']}")
    logger.info(f"Memories removed: {stats['memories_removed']}")
    logger.info(f"Memory merges: {stats['memory_merges']}")
    
    # Show trust levels
    logger.info("\nFinal Trust Levels:")
    sources = {m["source"] for m in memories}
    for source in sources:
        trust_level = belief_toolkit.get_source_trust(source)
        logger.info(f"{source}: {trust_level.name}")
    
    # Save state
    logger.info("\nSaving belief system state...")
    belief_toolkit.save_state("belief_system_state.json")
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    main() 