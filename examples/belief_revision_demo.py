#!/usr/bin/env python3
"""
Belief Revision System Demo

This script demonstrates the belief revision system, which handles:
1. Arbitrating contradictory memories
2. Calibrating trust in information sources
3. Tracking belief revision decisions

The demo creates a set of memories with contradictions from different sources,
then shows how the system resolves these contradictions based on coherence,
trust, recency, and other factors.
"""

import time
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.embedding.embedding import BasicEmbeddingEngine
from psi_c_ai_sdk.coherence.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.contradiction.contradiction import ContradictionDetector
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator
from psi_c_ai_sdk.beliefs.revision import (
    BeliefRevisionSystem, TrustSource, RevisionDecision, DecisionReason
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("belief_revision_demo")


def create_test_memories(count: int = 20) -> List[Memory]:
    """
    Create a set of test memories with some contradictions.
    
    Args:
        count: Number of memories to create
        
    Returns:
        List of Memory objects
    """
    memories = []
    
    # Create some sets of facts with contradictions
    fact_sets = [
        # Weather facts (true vs false)
        [
            {"content": "Today is sunny with clear skies.", "source": "weather_app", "truth": True},
            {"content": "Today is rainy with cloudy skies.", "source": "news_channel", "truth": False}
        ],
        # Physics facts (true vs false)
        [
            {"content": "Light travels at approximately 299,792,458 meters per second in vacuum.", "source": "science_textbook", "truth": True},
            {"content": "Light travels at exactly 300,000 kilometers per second in all mediums.", "source": "social_media", "truth": False}
        ],
        # History facts (true vs false)
        [
            {"content": "The United States declared independence in 1776.", "source": "history_book", "truth": True},
            {"content": "The United States declared independence in 1780.", "source": "random_website", "truth": False}
        ],
        # Biology facts (true vs false)
        [
            {"content": "DNA has a double helix structure.", "source": "biology_journal", "truth": True},
            {"content": "DNA has a single helix structure.", "source": "unreliable_blog", "truth": False}
        ],
        # Geography facts (true vs false)
        [
            {"content": "The Amazon River is in South America.", "source": "geography_atlas", "truth": True},
            {"content": "The Amazon River is in Africa.", "source": "social_media", "truth": False}
        ],
        # Mathematics facts (true vs false)
        [
            {"content": "The value of pi is approximately 3.14159.", "source": "math_reference", "truth": True},
            {"content": "The value of pi is exactly 3.2.", "source": "random_website", "truth": False}
        ],
        # Astronomy facts (true vs false)
        [
            {"content": "Jupiter is the largest planet in our solar system.", "source": "astronomy_textbook", "truth": True},
            {"content": "Saturn is the largest planet in our solar system.", "source": "unreliable_blog", "truth": False}
        ],
        # Health facts (true vs false)
        [
            {"content": "Regular exercise is beneficial for cardiovascular health.", "source": "medical_journal", "truth": True},
            {"content": "Exercise has no impact on cardiovascular health.", "source": "random_forum", "truth": False}
        ],
        # Technology facts (true vs false)
        [
            {"content": "Computers process information using binary code.", "source": "tech_textbook", "truth": True},
            {"content": "Computers do not use binary code for information processing.", "source": "social_media", "truth": False}
        ],
        # Nutrition facts (true vs false)
        [
            {"content": "Fruits are generally a good source of vitamins.", "source": "nutrition_guide", "truth": True},
            {"content": "Fruits contain no essential vitamins or nutrients.", "source": "unreliable_blog", "truth": False}
        ]
    ]
    
    # Create additional regular memories (non-contradictory)
    regular_facts = [
        {"content": "Water boils at 100 degrees Celsius at standard pressure.", "source": "science_textbook", "domain": "physics"},
        {"content": "The Earth rotates on its axis once every 24 hours.", "source": "astronomy_textbook", "domain": "astronomy"},
        {"content": "Paris is the capital of France.", "source": "geography_atlas", "domain": "geography"},
        {"content": "Photosynthesis is the process by which plants convert light energy into chemical energy.", "source": "biology_journal", "domain": "biology"},
        {"content": "The Great Wall of China is visible from space.", "source": "random_website", "domain": "geography", "truth": False},
        {"content": "The human body has 206 bones.", "source": "medical_journal", "domain": "biology"},
        {"content": "Python is a programming language.", "source": "tech_textbook", "domain": "technology"},
        {"content": "The Mona Lisa was painted by Leonardo da Vinci.", "source": "history_book", "domain": "art"},
        {"content": "Proteins are made up of amino acids.", "source": "biology_journal", "domain": "biology"},
        {"content": "Mount Everest is the tallest mountain in the world.", "source": "geography_atlas", "domain": "geography"},
        {"content": "Gold is a chemical element with the symbol Au.", "source": "science_textbook", "domain": "chemistry"},
        {"content": "Shakespeare wrote 'Romeo and Juliet'.", "source": "history_book", "domain": "literature"},
        {"content": "Sound travels faster in water than in air.", "source": "science_textbook", "domain": "physics"},
        {"content": "Bananas are a good source of potassium.", "source": "nutrition_guide", "domain": "nutrition"},
        {"content": "The Sahara is the largest hot desert in the world.", "source": "geography_atlas", "domain": "geography"},
        {"content": "The Nile is the longest river in the world.", "source": "geography_atlas", "domain": "geography"},
        {"content": "The heart pumps blood through the body.", "source": "medical_journal", "domain": "biology"},
        {"content": "Mozart was a famous composer.", "source": "history_book", "domain": "music"},
        {"content": "Oxygen is necessary for human survival.", "source": "medical_journal", "domain": "biology"},
        {"content": "The speed of sound in air is approximately 343 meters per second.", "source": "science_textbook", "domain": "physics"}
    ]
    
    # Create enough memories to meet the requested count
    all_facts = []
    
    # Add contradictory pairs
    for fact_set in fact_sets:
        all_facts.extend(fact_set)
    
    # Add regular facts
    all_facts.extend(regular_facts)
    
    # Ensure we have enough facts
    if len(all_facts) < count:
        all_facts = all_facts * (count // len(all_facts) + 1)
    
    # Take a random subset if we have too many
    if len(all_facts) > count:
        all_facts = random.sample(all_facts, count)
    
    # Convert to Memory objects with appropriate metadata
    for i, fact in enumerate(all_facts):
        # Create memory with appropriate metadata
        metadata = {
            "source": fact["source"],
            "domain": fact.get("domain", "general_knowledge"),
            "importance": random.uniform(0.5, 1.0),
            "confidence": random.uniform(0.7, 1.0)
        }
        
        # Add truth flag for evaluation (normally this wouldn't be in real data)
        if "truth" in fact:
            metadata["_truth"] = fact["truth"]
        
        # Vary creation times to test recency effects
        # Memories created between 1 and 30 days ago
        days_ago = random.uniform(1, 30)
        creation_time = time.time() - (days_ago * 24 * 60 * 60)
        
        memory = Memory(
            content=fact["content"],
            metadata=metadata,
            creation_time=creation_time
        )
        
        memories.append(memory)
    
    return memories


def create_trust_sources() -> Dict[str, TrustSource]:
    """
    Create a set of trust sources with different initial trust levels.
    
    Returns:
        Dictionary mapping source IDs to TrustSource objects
    """
    sources = {
        "science_textbook": TrustSource(
            source_id="science_textbook",
            name="Science Textbook",
            initial_trust=0.9,
            expertise={
                "physics": 0.95,
                "chemistry": 0.95,
                "biology": 0.9
            }
        ),
        "medical_journal": TrustSource(
            source_id="medical_journal",
            name="Medical Journal",
            initial_trust=0.85,
            expertise={
                "biology": 0.9,
                "health": 0.95,
                "nutrition": 0.8
            }
        ),
        "history_book": TrustSource(
            source_id="history_book",
            name="History Book",
            initial_trust=0.8,
            expertise={
                "history": 0.9,
                "geography": 0.7,
                "art": 0.7,
                "literature": 0.8
            }
        ),
        "geography_atlas": TrustSource(
            source_id="geography_atlas",
            name="Geography Atlas",
            initial_trust=0.85,
            expertise={
                "geography": 0.95
            }
        ),
        "astronomy_textbook": TrustSource(
            source_id="astronomy_textbook",
            name="Astronomy Textbook",
            initial_trust=0.85,
            expertise={
                "astronomy": 0.9,
                "physics": 0.8
            }
        ),
        "tech_textbook": TrustSource(
            source_id="tech_textbook",
            name="Technology Textbook",
            initial_trust=0.8,
            expertise={
                "technology": 0.9,
                "computer_science": 0.85
            }
        ),
        "biology_journal": TrustSource(
            source_id="biology_journal",
            name="Biology Journal",
            initial_trust=0.85,
            expertise={
                "biology": 0.95,
                "health": 0.8
            }
        ),
        "nutrition_guide": TrustSource(
            source_id="nutrition_guide",
            name="Nutrition Guide",
            initial_trust=0.75,
            expertise={
                "nutrition": 0.9,
                "health": 0.75
            }
        ),
        "math_reference": TrustSource(
            source_id="math_reference",
            name="Mathematical Reference",
            initial_trust=0.9,
            expertise={
                "mathematics": 0.95
            }
        ),
        "weather_app": TrustSource(
            source_id="weather_app",
            name="Weather App",
            initial_trust=0.7,
            expertise={
                "weather": 0.8
            }
        ),
        "news_channel": TrustSource(
            source_id="news_channel",
            name="News Channel",
            initial_trust=0.6,
            expertise={
                "current_events": 0.7,
                "weather": 0.6,
                "politics": 0.5
            }
        ),
        "random_website": TrustSource(
            source_id="random_website",
            name="Random Website",
            initial_trust=0.4,
            expertise={}
        ),
        "social_media": TrustSource(
            source_id="social_media",
            name="Social Media",
            initial_trust=0.3,
            expertise={}
        ),
        "unreliable_blog": TrustSource(
            source_id="unreliable_blog",
            name="Unreliable Blog",
            initial_trust=0.2,
            expertise={}
        ),
        "random_forum": TrustSource(
            source_id="random_forum",
            name="Random Internet Forum",
            initial_trust=0.25,
            expertise={}
        )
    }
    
    return sources


def evaluate_decisions(decisions: List[RevisionDecision], memory_store: MemoryStore) -> Dict[str, Any]:
    """
    Evaluate the quality of belief revision decisions.
    
    This function checks if the system kept the "true" memory and rejected the "false" one
    based on the hidden _truth metadata field.
    
    Args:
        decisions: List of revision decisions
        memory_store: Memory store containing memories
        
    Returns:
        Evaluation metrics
    """
    correct_decisions = 0
    incorrect_decisions = 0
    uncertain_decisions = 0
    
    reason_counts = {}
    
    for decision in decisions:
        # Get the kept memory
        kept_memory = memory_store.get_memory(decision.kept_memory_id)
        
        # The rejected memory might be gone if it was removed
        # We can get info from the decision details
        
        # Count the reason
        reason = decision.reason
        reason_name = reason.name
        reason_counts[reason_name] = reason_counts.get(reason_name, 0) + 1
        
        # Check if kept memory has a truth flag
        if kept_memory and kept_memory.metadata and "_truth" in kept_memory.metadata:
            if kept_memory.metadata["_truth"]:
                correct_decisions += 1
            else:
                incorrect_decisions += 1
        else:
            uncertain_decisions += 1
    
    # Calculate accuracy if we have ground truth
    accuracy = 0
    if (correct_decisions + incorrect_decisions) > 0:
        accuracy = correct_decisions / (correct_decisions + incorrect_decisions)
    
    return {
        "correct_decisions": correct_decisions,
        "incorrect_decisions": incorrect_decisions,
        "uncertain_decisions": uncertain_decisions,
        "accuracy": accuracy,
        "reason_counts": reason_counts
    }


def visualize_trust_evolution(trust_sources: Dict[str, TrustSource]) -> None:
    """
    Visualize how trust levels evolved during the demo.
    
    Args:
        trust_sources: Dictionary of trust sources
    """
    plt.figure(figsize=(12, 8))
    
    # Sort sources by final trust level
    sorted_sources = sorted(
        trust_sources.values(), 
        key=lambda x: x.trust_level, 
        reverse=True
    )
    
    # Plot each source's initial and final trust
    source_names = [s.name for s in sorted_sources]
    initial_trusts = [s.trust_history[0][1] for s in sorted_sources]
    final_trusts = [s.trust_level for s in sorted_sources]
    
    # Create indices for the bars
    x = np.arange(len(source_names))
    width = 0.35
    
    # Create the bars
    plt.bar(x - width/2, initial_trusts, width, label='Initial Trust')
    plt.bar(x + width/2, final_trusts, width, label='Final Trust')
    
    # Add labels and title
    plt.xlabel('Information Source')
    plt.ylabel('Trust Level')
    plt.title('Trust Level Evolution After Belief Revision')
    plt.xticks(x, source_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('trust_evolution.png')
    plt.close()
    
    # Another figure for detailed trust history of selected sources
    plt.figure(figsize=(12, 8))
    
    # Select some interesting sources to show detailed history
    selected_sources = [
        "science_textbook", "unreliable_blog", 
        "social_media", "medical_journal", "random_website"
    ]
    
    for source_id in selected_sources:
        if source_id in trust_sources:
            source = trust_sources[source_id]
            history = source.trust_history
            
            # Convert timestamps to relative times (0 = start)
            start_time = history[0][0]
            times = [(t - start_time) / 60 for t, _ in history]  # Convert to minutes
            values = [v for _, v in history]
            
            plt.plot(times, values, marker='o', linewidth=2, label=source.name)
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Trust Level')
    plt.title('Trust Level History for Selected Sources')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trust_history.png')
    plt.close()


def visualize_decision_metrics(decisions: List[RevisionDecision], eval_metrics: Dict[str, Any]) -> None:
    """
    Visualize statistics about belief revision decisions.
    
    Args:
        decisions: List of revision decisions
        eval_metrics: Evaluation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Decision reasons
    reason_counts = eval_metrics["reason_counts"]
    reasons = list(reason_counts.keys())
    counts = list(reason_counts.values())
    
    # Sort by count
    sorted_data = sorted(zip(reasons, counts), key=lambda x: x[1], reverse=True)
    reasons, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    ax1.bar(reasons, counts)
    ax1.set_title('Primary Reasons for Belief Revision Decisions')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Decision Reason')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Decision confidence distribution
    confidences = [d.confidence for d in decisions]
    
    ax2.hist(confidences, bins=10, alpha=0.7)
    ax2.set_title('Distribution of Decision Confidence')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Count')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('decision_metrics.png')
    plt.close()
    
    # Create a pie chart for correct/incorrect decisions
    plt.figure(figsize=(10, 8))
    labels = ['Correct', 'Incorrect', 'Uncertain']
    sizes = [
        eval_metrics["correct_decisions"], 
        eval_metrics["incorrect_decisions"], 
        eval_metrics["uncertain_decisions"]
    ]
    colors = ['#4CAF50', '#F44336', '#FFC107']
    explode = (0.1, 0, 0)  # explode the 1st slice (Correct)
    
    plt.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Belief Revision Decision Accuracy')
    
    plt.tight_layout()
    plt.savefig('decision_accuracy.png')
    plt.close()


def main():
    """Run the belief revision demo."""
    logger.info("Starting Belief Revision System Demo")
    
    # Create components
    embedding_engine = BasicEmbeddingEngine(dim=64)  # Smaller dim for faster demo
    memory_store = MemoryStore(embedding_engine=embedding_engine)
    coherence_scorer = BasicCoherenceScorer(memory_store=memory_store)
    contradiction_detector = ContradictionDetector(embedding_engine=embedding_engine)
    entropy_calculator = EntropyCalculator()
    
    # Create trust sources
    trust_sources = create_trust_sources()
    
    # Create the belief revision system
    belief_system = BeliefRevisionSystem(
        memory_store=memory_store,
        contradiction_detector=contradiction_detector,
        coherence_scorer=coherence_scorer,
        entropy_calculator=entropy_calculator,
        trust_sources=trust_sources,
        decision_log_path="belief_revision_decisions.log"
    )
    
    # Create test memories
    memories = create_test_memories(50)
    
    # Add memories to memory store
    for memory in memories:
        memory_store.add_memory(memory)
    
    logger.info(f"Added {len(memories)} memories to the memory store")
    
    # Find and resolve contradictions
    decisions = belief_system.find_and_resolve_contradictions(limit=20)
    
    logger.info(f"Resolved {len(decisions)} contradictions")
    
    # Evaluate the decisions
    eval_metrics = evaluate_decisions(decisions, memory_store)
    
    logger.info("Belief revision evaluation:")
    logger.info(f"Correct decisions: {eval_metrics['correct_decisions']}")
    logger.info(f"Incorrect decisions: {eval_metrics['incorrect_decisions']}")
    logger.info(f"Uncertain decisions: {eval_metrics['uncertain_decisions']}")
    logger.info(f"Accuracy: {eval_metrics['accuracy']:.2f}")
    
    # Show reason breakdown
    logger.info("Decision reasons:")
    for reason, count in eval_metrics["reason_counts"].items():
        logger.info(f"  {reason}: {count}")
    
    # Display trust levels
    logger.info("Final trust levels:")
    for source_id, source in sorted(
        trust_sources.items(), 
        key=lambda x: x[1].trust_level, 
        reverse=True
    ):
        logger.info(f"  {source.name}: {source.trust_level:.3f}")
    
    # Visualize results
    visualize_trust_evolution(trust_sources)
    visualize_decision_metrics(decisions, eval_metrics)
    
    logger.info("Visualizations saved to:")
    logger.info("  - trust_evolution.png - Trust level changes")
    logger.info("  - trust_history.png - Detailed trust history")
    logger.info("  - decision_metrics.png - Decision reason statistics")
    logger.info("  - decision_accuracy.png - Decision accuracy chart")
    
    logger.info("Belief Revision Demo completed")


if __name__ == "__main__":
    main() 