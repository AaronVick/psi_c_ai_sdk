#!/usr/bin/env python3
"""
Temporal Coherence Quickstart - ΨC-AI SDK

A minimal example demonstrating the core functionality of the temporal coherence 
components in the ΨC-AI SDK. This quickstart focuses on:

1. Detecting temporal patterns in memories (recurring themes, causal relationships)
2. Checking timeline consistency (contradictions, gaps, impossible sequences)

This is a simplified version of the full temporal_coherence_demo.py example.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import ΨC-AI SDK components
from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.temporal_coherence.temporal_pattern import TemporalPatternDetector
from psi_c_ai_sdk.temporal_coherence.timeline_consistency import TimelineConsistencyChecker
from psi_c_ai_sdk.safety.contradiction_detector import ContradictionDetector
from psi_c_ai_sdk.coherence.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.embedding.embedding import BasicEmbeddingEngine


def main():
    """Demonstrate core temporal coherence functionality."""
    # Initialize components
    embedding_engine = BasicEmbeddingEngine()
    memory_store = MemoryStore()
    coherence_scorer = BasicCoherenceScorer(embedding_engine)
    contradiction_detector = ContradictionDetector()
    
    # Create test memories with temporal relationships
    now = datetime.now()
    memories = [
        # Base event
        Memory(
            id="1",
            content="I started working on the AI project on Monday.",
            embedding=None,
            timestamp=now - timedelta(days=10),
            importance=0.8
        ),
        # Follow-up event (causal relationship)
        Memory(
            id="2",
            content="After starting the AI project, I identified key requirements.",
            embedding=None,
            timestamp=now - timedelta(days=9),
            importance=0.7
        ),
        # Recurring theme
        Memory(
            id="3",
            content="Safety considerations are a top priority for our AI system.",
            embedding=None,
            timestamp=now - timedelta(days=8),
            importance=0.6
        ),
        # Same recurring theme
        Memory(
            id="4",
            content="Our team discussed AI safety protocols during the weekly meeting.",
            embedding=None,
            timestamp=now - timedelta(days=5),
            importance=0.7
        ),
        # Contradictory memory (impossible sequence)
        Memory(
            id="5",
            content="After completing the entire AI project, I started the requirements phase.",
            embedding=None,
            timestamp=now - timedelta(days=7),
            importance=0.5
        ),
        # Timeline gap
        Memory(
            id="6",
            content="Initial planning for this AI project was done last year.",
            embedding=None,
            timestamp=now - timedelta(days=365),
            importance=0.3
        ),
    ]
    
    # Add access history for pattern detection
    for memory in memories:
        # Generate embedding
        memory.embedding = embedding_engine.generate_embedding(memory.content)
        # Simulate access history (attribute used by pattern detector)
        memory.access_history = [memory.timestamp, memory.timestamp + timedelta(days=1)]
        # Add to store
        memory_store.add_memory(memory)
    
    logger.info(f"Added {len(memories)} memories to the store")
    
    # 1. Temporal Pattern Detection
    logger.info("Detecting temporal patterns...")
    pattern_detector = TemporalPatternDetector(
        memory_store=memory_store,
        min_pattern_confidence=0.5,
        time_window=timedelta(days=30),
        min_pattern_occurrences=2
    )
    
    # Detect all patterns
    patterns = pattern_detector.detect_all_patterns()
    logger.info(f"Detected {len(patterns)} temporal patterns")
    
    # Show detected patterns
    if patterns:
        print("\n=== Detected Temporal Patterns ===")
        for i, pattern in enumerate(patterns, 1):
            print(f"{i}. Type: {pattern.pattern_type}")
            print(f"   Confidence: {pattern.confidence:.2f}")
            print(f"   Memory IDs: {pattern.memory_ids}")
            print(f"   Time range: {pattern.start_time.strftime('%Y-%m-%d')} to {pattern.end_time.strftime('%Y-%m-%d')}")
            if pattern.metadata:
                print(f"   Metadata: {pattern.metadata}")
            print()
    
    # 2. Timeline Consistency Checking
    logger.info("Checking timeline consistency...")
    consistency_checker = TimelineConsistencyChecker(
        memory_store=memory_store,
        contradiction_detector=contradiction_detector,
        timeline_gap_threshold=timedelta(days=30),
        severity_threshold=0.3
    )
    
    # Check for inconsistencies
    inconsistencies = consistency_checker.check_timeline_consistency()
    logger.info(f"Detected {len(inconsistencies)} timeline inconsistencies")
    
    # Show detected inconsistencies
    if inconsistencies:
        print("\n=== Detected Timeline Inconsistencies ===")
        for i, inconsistency in enumerate(inconsistencies, 1):
            print(f"{i}. Type: {inconsistency.inconsistency_type}")
            print(f"   Severity: {inconsistency.severity:.2f}")
            print(f"   Description: {inconsistency.description}")
            print(f"   Memory IDs: {inconsistency.memory_ids}")
            if inconsistency.repair_suggestions:
                print(f"   Repair suggestions: {inconsistency.repair_suggestions}")
            print()
    
    # 3. Demonstrate specific pattern detection methods
    print("\n=== Specific Pattern Detection ===")
    
    # Detect recurring themes
    recurring_themes = pattern_detector.detect_recurring_themes()
    print(f"Recurring themes: {len(recurring_themes)}")
    
    # Detect causal relationships
    causal_relationships = pattern_detector.detect_causal_relationships()
    print(f"Causal relationships: {len(causal_relationships)}")
    
    # Get patterns for a specific memory
    memory_patterns = pattern_detector.get_patterns_for_memory("1")
    print(f"Patterns involving memory #1: {len(memory_patterns)}")


if __name__ == "__main__":
    main() 