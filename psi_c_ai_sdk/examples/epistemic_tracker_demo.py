#!/usr/bin/env python3
"""
Epistemic Status Tracker Demo

This script demonstrates the functionality of the Epistemic Status Tracking system in the ΨC-AI SDK.
It shows how the system:
1. Classifies beliefs into different epistemic states (established, confident, uncertain, etc.)
2. Calculates confidence scores based on multiple factors
3. Tracks stability of beliefs over time
4. Propagates uncertainty through related memories
5. Identifies and tracks knowledge gaps

The demo simulates a growing knowledge base and shows how epistemic status evolves.
"""

import logging
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from pprint import pprint

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.memory.embedding import EmbeddingModel
from psi_c_ai_sdk.memory.coherence import CoherenceScorer
from psi_c_ai_sdk.beliefs.contradiction import ContradictionDetector
from psi_c_ai_sdk.beliefs.revision import BeliefRevisionSystem
from psi_c_ai_sdk.epistemic.epistemic_tracker import (
    EpistemicStatusTracker,
    EpistemicClassifier,
    UncertaintyPropagator,
    KnowledgeState
)
from psi_c_ai_sdk.util.logging import setup_logger

# Set up logging
logger = setup_logger("epistemic_tracker_demo", level=logging.INFO)


def create_test_memories() -> List[Memory]:
    """
    Create a set of test memories with varying properties for testing epistemic status.
    
    Returns:
        List of Memory objects
    """
    base_time = datetime.now() - timedelta(days=30)
    memories = []
    
    # Group 1: Well-established facts with high confidence
    established_facts = [
        {
            "content": "Water is composed of hydrogen and oxygen atoms.",
            "source": "scientific_knowledge",
            "creation_time": base_time - timedelta(days=25),
            "importance": 0.9,
            "metadata": {
                "confidence": 0.95,
                "category": "science",
                "verification_count": 5
            }
        },
        {
            "content": "The Earth orbits around the Sun.",
            "source": "scientific_knowledge",
            "creation_time": base_time - timedelta(days=28),
            "importance": 0.85,
            "metadata": {
                "confidence": 0.98,
                "category": "astronomy",
                "verification_count": 4
            }
        },
        {
            "content": "Paris is the capital of France.",
            "source": "geographic_data",
            "creation_time": base_time - timedelta(days=20),
            "importance": 0.8,
            "metadata": {
                "confidence": 0.99,
                "category": "geography",
                "verification_count": 3
            }
        }
    ]
    
    # Group 2: Confident beliefs with good but not perfect confidence
    confident_beliefs = [
        {
            "content": "The user prefers dark mode interfaces.",
            "source": "user_preferences",
            "creation_time": base_time - timedelta(days=10),
            "importance": 0.75,
            "metadata": {
                "confidence": 0.85,
                "category": "preferences",
                "observation_count": 3
            }
        },
        {
            "content": "The user typically works from 9 AM to 5 PM on weekdays.",
            "source": "usage_patterns",
            "creation_time": base_time - timedelta(days=15),
            "importance": 0.7,
            "metadata": {
                "confidence": 0.8,
                "category": "habits",
                "observation_count": 4
            }
        },
        {
            "content": "The user is interested in artificial intelligence topics.",
            "source": "content_interaction",
            "creation_time": base_time - timedelta(days=12),
            "importance": 0.8,
            "metadata": {
                "confidence": 0.85,
                "category": "interests",
                "observation_count": 5
            }
        }
    ]
    
    # Group 3: Tentative beliefs with moderate confidence
    tentative_beliefs = [
        {
            "content": "The user might be planning a trip to Japan next month.",
            "source": "conversation",
            "creation_time": base_time - timedelta(days=5),
            "importance": 0.6,
            "metadata": {
                "confidence": 0.6,
                "category": "plans",
                "observation_count": 1
            }
        },
        {
            "content": "The user seems to prefer fiction books over non-fiction.",
            "source": "content_interaction",
            "creation_time": base_time - timedelta(days=8),
            "importance": 0.5,
            "metadata": {
                "confidence": 0.65,
                "category": "preferences",
                "observation_count": 2
            }
        },
        {
            "content": "The user may be considering changing jobs in the near future.",
            "source": "conversation",
            "creation_time": base_time - timedelta(days=3),
            "importance": 0.7,
            "metadata": {
                "confidence": 0.55,
                "category": "career",
                "observation_count": 1
            }
        }
    ]
    
    # Group 4: Uncertain beliefs with low confidence
    uncertain_beliefs = [
        {
            "content": "The user might be allergic to peanuts.",
            "source": "casual_mention",
            "creation_time": base_time - timedelta(days=2),
            "importance": 0.8,  # High importance due to health implications
            "metadata": {
                "confidence": 0.4,
                "category": "health",
                "observation_count": 1
            }
        },
        {
            "content": "The user possibly owns a cat.",
            "source": "background_noise",
            "creation_time": base_time - timedelta(days=1),
            "importance": 0.3,
            "metadata": {
                "confidence": 0.3,
                "category": "personal",
                "observation_count": 1
            }
        },
        {
            "content": "The user may prefer tea over coffee in the morning.",
            "source": "single_observation",
            "creation_time": base_time - timedelta(hours=12),
            "importance": 0.4,
            "metadata": {
                "confidence": 0.35,
                "category": "preferences",
                "observation_count": 1
            }
        }
    ]
    
    # Group 5: Contradictory beliefs
    contradictory_beliefs = [
        {
            "content": "The user's favorite color is blue.",
            "source": "user_profile",
            "creation_time": base_time - timedelta(days=18),
            "importance": 0.5,
            "metadata": {
                "confidence": 0.7,
                "category": "preferences",
                "observation_count": 2
            }
        },
        {
            "content": "The user mentioned their favorite color is green.",
            "source": "conversation",
            "creation_time": base_time - timedelta(days=2),
            "importance": 0.5,
            "metadata": {
                "confidence": 0.75,
                "category": "preferences",
                "observation_count": 1
            }
        },
        {
            "content": "The meeting is scheduled for 2 PM tomorrow.",
            "source": "calendar_initial",
            "creation_time": base_time - timedelta(days=3),
            "importance": 0.7,
            "metadata": {
                "confidence": 0.8,
                "category": "schedule",
                "observation_count": 1
            }
        },
        {
            "content": "The meeting has been rescheduled to 3 PM tomorrow.",
            "source": "calendar_update",
            "creation_time": base_time - timedelta(days=1),
            "importance": 0.75,
            "metadata": {
                "confidence": 0.9,
                "category": "schedule",
                "observation_count": 1
            }
        }
    ]
    
    # Group 6: Knowledge gaps (placeholders)
    knowledge_gaps = [
        {
            "content": "The user's dietary restrictions are unknown.",
            "source": "system_analysis",
            "creation_time": base_time - timedelta(hours=6),
            "importance": 0.6,
            "metadata": {
                "confidence": 0.2,
                "category": "health",
                "is_placeholder": True,
                "needs_information": True
            }
        },
        {
            "content": "The user's preference for notification frequency is not established.",
            "source": "system_analysis",
            "creation_time": base_time - timedelta(hours=12),
            "importance": 0.55,
            "metadata": {
                "confidence": 0.1,
                "category": "preferences",
                "is_placeholder": True,
                "needs_information": True
            }
        }
    ]
    
    # Combine all groups
    all_beliefs = (
        established_facts + 
        confident_beliefs + 
        tentative_beliefs + 
        uncertain_beliefs + 
        contradictory_beliefs +
        knowledge_gaps
    )
    
    # Create memory objects
    for belief in all_beliefs:
        memory = Memory(
            content=belief["content"],
            source=belief["source"],
            creation_time=belief["creation_time"],
            importance=belief["importance"],
            metadata=belief["metadata"]
        )
        memories.append(memory)
    
    # Add a few semantically related memories for testing uncertainty propagation
    related_memories = [
        # Related to Japan trip
        Memory(
            content="The user has been researching Japanese phrases recently.",
            source="search_history",
            creation_time=base_time - timedelta(days=4),
            importance=0.5,
            metadata={
                "confidence": 0.7,
                "category": "interests",
                "related_to": "Japan trip"
            }
        ),
        Memory(
            content="The user has been viewing Tokyo hotel websites.",
            source="browsing_history",
            creation_time=base_time - timedelta(days=3),
            importance=0.55,
            metadata={
                "confidence": 0.65,
                "category": "plans",
                "related_to": "Japan trip"
            }
        ),
        
        # Related to job change
        Memory(
            content="The user updated their resume last week.",
            source="file_activity",
            creation_time=base_time - timedelta(days=7),
            importance=0.6,
            metadata={
                "confidence": 0.7,
                "category": "career",
                "related_to": "job change"
            }
        ),
        Memory(
            content="The user has been viewing job posting websites.",
            source="browsing_history",
            creation_time=base_time - timedelta(days=5),
            importance=0.65,
            metadata={
                "confidence": 0.6,
                "category": "career",
                "related_to": "job change"
            }
        )
    ]
    
    memories.extend(related_memories)
    
    return memories


def simulate_state_changes(
    tracker: EpistemicStatusTracker, 
    memory_store: MemoryStore,
    iterations: int = 5
) -> List[Dict[str, Any]]:
    """
    Simulate epistemological changes to memories over time.
    
    Args:
        tracker: Epistemic status tracker
        memory_store: Memory store
        iterations: Number of change iterations to simulate
        
    Returns:
        List of events that occurred during simulation
    """
    events = []
    
    for i in range(iterations):
        logger.info(f"\nSimulating epistemic changes - Iteration {i+1}/{iterations}")
        
        # Get all memories
        all_memories = memory_store.get_all()
        
        # 1. Randomly select a memory to verify (boosting confidence)
        if all_memories:
            memory = random.choice(all_memories)
            verification_success = random.random() > 0.3  # 70% chance of successful verification
            
            tracker.classifier.update_verification(memory.id, verification_success)
            
            event = {
                "type": "verification",
                "memory_id": memory.id,
                "memory_content": memory.content,
                "success": verification_success,
                "timestamp": datetime.now()
            }
            events.append(event)
            
            # Update memory status after verification
            old_status = tracker.get_memory_status(memory.id)
            new_status = tracker.update_memory_status(memory)
            
            logger.info(f"Verification {'succeeded' if verification_success else 'failed'} "
                       f"for: \"{memory.content[:50]}...\"")
            if old_status != new_status:
                logger.info(f"  Status changed: {old_status} → {new_status}")
                
                # Propagate status change
                affected = tracker.propagate_status_change(memory.id, new_status, old_status)
                if affected:
                    logger.info(f"  Status change propagated to {len(affected)} related memories")
                    for affected_id, status in affected.items():
                        affected_memory = next((m for m in all_memories if m.id == affected_id), None)
                        if affected_memory:
                            logger.info(f"    - \"{affected_memory.content[:40]}...\" → {status}")
        
        # 2. Randomly select a memory to add a contradiction to
        if len(all_memories) >= 2 and random.random() > 0.5:  # 50% chance
            memory = random.choice(all_memories)
            
            # Increment contradiction count
            tracker.classifier.update_contradiction_count(memory.id)
            
            event = {
                "type": "contradiction_added",
                "memory_id": memory.id,
                "memory_content": memory.content,
                "timestamp": datetime.now()
            }
            events.append(event)
            
            # Update memory status after adding contradiction
            old_status = tracker.get_memory_status(memory.id)
            new_status = tracker.update_memory_status(memory)
            
            logger.info(f"Contradiction discovered for: \"{memory.content[:50]}...\"")
            if old_status != new_status:
                logger.info(f"  Status changed: {old_status} → {new_status}")
        
        # 3. Register a knowledge gap
        if random.random() > 0.7:  # 30% chance
            topics = [
                "user's favorite music genre",
                "user's preferred working hours",
                "user's birthday",
                "user's travel frequency",
                "user's opinion on AI assistants",
                "user's educational background"
            ]
            
            topic = random.choice(topics)
            importance = random.uniform(0.4, 0.9)
            
            gap_id = tracker.register_knowledge_gap(topic, importance)
            
            event = {
                "type": "knowledge_gap",
                "gap_id": gap_id,
                "topic": topic,
                "importance": importance,
                "timestamp": datetime.now()
            }
            events.append(event)
            
            logger.info(f"Registered knowledge gap: '{topic}' (importance: {importance:.2f})")
        
        # 4. Randomly resolve a knowledge gap
        open_gaps = tracker.get_open_knowledge_gaps()
        if open_gaps and random.random() > 0.6:  # 40% chance
            gap = random.choice(open_gaps)
            resolution = f"Learned from user interaction on {datetime.now().strftime('%Y-%m-%d')}"
            
            tracker.mark_knowledge_gap_resolved(gap["id"], resolution)
            
            event = {
                "type": "gap_resolution",
                "gap_id": gap["id"],
                "topic": gap["topic"],
                "resolution": resolution,
                "timestamp": datetime.now()
            }
            events.append(event)
            
            logger.info(f"Resolved knowledge gap: '{gap['topic']}'")
            logger.info(f"  Resolution: {resolution}")
        
        # Run a full scan at the end of each iteration
        tracker.scan_all_memories(force=True)
        
        # Pause between iterations
        time.sleep(0.5)
    
    return events


def plot_epistemic_distribution(tracker: EpistemicStatusTracker, filename: str) -> None:
    """
    Plot distribution of epistemic states.
    
    Args:
        tracker: Epistemic status tracker
        filename: Output filename for plot
    """
    distribution = tracker.metrics["status_distribution"]
    
    # Prepare data for plotting
    states = []
    counts = []
    
    # Convert Enum keys to strings and sort by count
    for state_enum, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        if isinstance(state_enum, KnowledgeState):
            state_name = state_enum.name
        else:
            state_name = str(state_enum)
        
        states.append(state_name)
        counts.append(count)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Use different colors based on epistemic quality
    colors = ['#2ca02c', '#2ca02c', '#98df8a', '#dbdb8d', '#d62728', '#ff9896', '#7f7f7f', '#c7c7c7']
    if len(states) > len(colors):
        colors = colors + ['#7f7f7f'] * (len(states) - len(colors))
    
    bars = plt.bar(states, counts, color=colors[:len(states)])
    
    plt.xlabel('Epistemic State')
    plt.ylabel('Number of Memories')
    plt.title('Distribution of Epistemic States')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Epistemic state distribution plot saved to '{filename}'")
    plt.close()


def plot_confidence_distribution(tracker: EpistemicStatusTracker, memory_store: MemoryStore, filename: str) -> None:
    """
    Plot distribution of confidence scores.
    
    Args:
        tracker: Epistemic status tracker
        memory_store: Memory store
        filename: Output filename for plot
    """
    all_memories = memory_store.get_all()
    
    # Get confidence scores
    confidence_scores = [tracker.get_memory_confidence(memory.id) for memory in all_memories]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    # Calculate histogram
    hist, bins = np.histogram(confidence_scores, bins=10, range=(0, 1))
    
    # Plot histogram
    plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge', 
           color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add average line
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    plt.axvline(x=avg_confidence, color='red', linestyle='--', 
               label=f'Average: {avg_confidence:.2f}')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Memories')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Confidence distribution plot saved to '{filename}'")
    plt.close()


def plot_metrics_over_time(tracker: EpistemicStatusTracker, filename: str) -> None:
    """
    Plot epistemic metrics over time.
    
    Args:
        tracker: Epistemic status tracker
        filename: Output filename for plot
    """
    history = tracker.metrics["update_history"]
    
    if len(history) < 2:
        logger.warning("Not enough history data to plot metrics over time")
        return
    
    # Extract data
    timestamps = [entry["timestamp"] for entry in history]
    confidence = [entry["avg_confidence"] for entry in history]
    stability = [entry["stability_score"] for entry in history]
    
    # Convert timestamps to relative seconds for x-axis
    start_time = timestamps[0]
    seconds = [(t - start_time).total_seconds() for t in timestamps]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(seconds, confidence, 'b-o', label='Average Confidence')
    plt.plot(seconds, stability, 'g-o', label='Stability Score')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Score')
    plt.title('Epistemic Metrics Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Metrics over time plot saved to '{filename}'")
    plt.close()


def run_demo() -> None:
    """Run the epistemic status tracker demonstration."""
    logger.info("Starting Epistemic Status Tracker Demo")
    
    # Initialize core components
    embedding_model = EmbeddingModel()
    memory_store = MemoryStore(embedding_model)
    coherence_scorer = CoherenceScorer()
    contradiction_detector = ContradictionDetector()
    
    # Initialize belief revision system
    belief_revision = BeliefRevisionSystem(
        memory_store=memory_store,
        contradiction_detector=contradiction_detector,
        coherence_scorer=coherence_scorer
    )
    
    # Set up initial trust levels
    source_trust = {
        "scientific_knowledge": 0.95,
        "geographic_data": 0.9,
        "user_preferences": 0.85,
        "usage_patterns": 0.8,
        "content_interaction": 0.75,
        "calendar_update": 0.9,
        "calendar_initial": 0.7,
        "conversation": 0.65,
        "user_profile": 0.7,
        "casual_mention": 0.5,
        "background_noise": 0.4,
        "single_observation": 0.45,
        "system_analysis": 0.8,
        "search_history": 0.7,
        "browsing_history": 0.6,
        "file_activity": 0.65
    }
    
    for source, trust in source_trust.items():
        belief_revision.set_trust_level(source, trust)
    
    # Initialize epistemic status components
    classifier = EpistemicClassifier(
        stability_threshold=0.6,
        certainty_threshold=0.7,
        confidence_decay_rate=0.01,
        history_window=5
    )
    
    uncertainty_propagator = UncertaintyPropagator(
        coherence_scorer=coherence_scorer,
        propagation_threshold=0.65,
        uncertainty_decay=0.25
    )
    
    # Initialize tracker
    tracker = EpistemicStatusTracker(
        memory_store=memory_store,
        classifier=classifier,
        contradiction_detector=contradiction_detector,
        coherence_scorer=coherence_scorer,
        uncertainty_propagator=uncertainty_propagator,
        belief_revision_system=belief_revision,
        scan_interval=5
    )
    
    # Create and add test memories
    logger.info("Loading test memories...")
    memories = create_test_memories()
    for memory in memories:
        memory_store.add(memory)
    logger.info(f"Added {len(memories)} memories to store")
    
    # Run initial scan
    logger.info("Performing initial epistemic status scan...")
    scan_results = tracker.scan_all_memories(force=True)
    
    # Display initial state distribution
    status_distribution = scan_results["status_distribution"]
    
    distribution_table = [
        [state.name if isinstance(state, KnowledgeState) else str(state), count]
        for state, count in sorted(status_distribution.items(), 
                                  key=lambda x: x[1], reverse=True)
    ]
    
    logger.info("\nInitial Epistemic State Distribution:")
    logger.info(tabulate(distribution_table, headers=["State", "Count"], tablefmt="grid"))
    
    # Show confidence for different types of memories
    logger.info("\nSample Memory Confidence Scores:")
    sample_memories = [
        ("ESTABLISHED", "Water is composed of hydrogen and oxygen atoms."),
        ("CONFIDENT", "The user prefers dark mode interfaces."),
        ("TENTATIVE", "The user might be planning a trip to Japan next month."),
        ("UNCERTAIN", "The user might be allergic to peanuts."),
        ("UNSTABLE", "The user's favorite color is blue.")
    ]
    
    confidence_table = []
    for expected_state, content_start in sample_memories:
        # Find a memory starting with this content
        memory = next((m for m in memories if m.content.startswith(content_start)), None)
        if memory:
            state = tracker.get_memory_status(memory.id)
            confidence = tracker.get_memory_confidence(memory.id)
            confidence_table.append([
                content_start[:50] + "...",
                state.name if state else "UNKNOWN",
                f"{confidence:.2f}"
            ])
    
    logger.info(tabulate(confidence_table, 
                       headers=["Memory", "State", "Confidence"], 
                       tablefmt="grid"))
    
    # Show knowledge gaps
    tracker.register_knowledge_gap("User's preferred communication channels", 0.7)
    tracker.register_knowledge_gap("User's typical response time to messages", 0.6)
    
    logger.info("\nKnowledge Gaps:")
    gaps = tracker.get_open_knowledge_gaps()
    gap_table = [
        [gap["topic"], f"{gap['importance']:.2f}", gap["identified_at"].strftime("%Y-%m-%d %H:%M")]
        for gap in gaps
    ]
    logger.info(tabulate(gap_table, 
                       headers=["Topic", "Importance", "Identified At"], 
                       tablefmt="grid"))
    
    # Simulate epistemic evolution
    logger.info("\nSimulating epistemic state evolution...")
    events = simulate_state_changes(tracker, memory_store, iterations=10)
    
    # Final scan
    logger.info("\nPerforming final epistemic status scan...")
    final_scan = tracker.scan_all_memories(force=True)
    
    # Display final statistics
    logger.info("\nFinal Epistemic Metrics:")
    metrics = tracker.get_metrics()
    metrics_table = [
        ["Average Confidence", f"{metrics['average_confidence']:.2f}"],
        ["Stability Score", f"{metrics['stability_score']:.2f}"],
        ["Contradiction Rate", f"{metrics['contradiction_rate']:.2f}"],
        ["Knowledge Gap Count", metrics['knowledge_gap_count']]
    ]
    logger.info(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Plot state distribution
    plot_epistemic_distribution(tracker, "epistemic_state_distribution.png")
    
    # Plot confidence distribution
    plot_confidence_distribution(tracker, memory_store, "confidence_distribution.png")
    
    # Plot metrics over time
    plot_metrics_over_time(tracker, "epistemic_metrics_over_time.png")
    
    # Display event summary
    logger.info("\nEpistemic Event Summary:")
    event_types = {}
    for event in events:
        event_type = event["type"]
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    event_table = [[event_type, count] for event_type, count in event_types.items()]
    logger.info(tabulate(event_table, headers=["Event Type", "Count"], tablefmt="grid"))
    
    logger.info("\nEpistemic Status Tracker Demo completed successfully")
    logger.info("Plots have been saved to the current directory")


if __name__ == "__main__":
    run_demo() 