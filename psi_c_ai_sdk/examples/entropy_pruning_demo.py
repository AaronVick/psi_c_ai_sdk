#!/usr/bin/env python3
"""
Entropy-Based Memory Pruning Demo

This script demonstrates the functionality of the Entropy-Based Memory Pruning System in the ΨC-AI SDK.
It shows how the system:
1. Calculates entropy in different memory configurations
2. Implements various pruning strategies (temporal decay, outlier removal, redundancy removal)
3. Analyzes the effect of pruning on memory organization and system performance
4. Visualizes memory distribution before and after pruning

This demo helps understand how strategic memory pruning can maintain cognitive efficiency.
"""

import logging
import random
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.memory.embedding import EmbeddingModel
from psi_c_ai_sdk.entropy.entropy import (
    EntropyCalculator, 
    EmbeddingEntropyMeasure, 
    SemanticCoherenceEntropyMeasure,
    TemporalCoherenceEntropyMeasure
)
from psi_c_ai_sdk.entropy.pruning import EntropyPruner, PruningStrategy
from psi_c_ai_sdk.util.logging import setup_logger

# Set up logging
logger = setup_logger("entropy_pruning_demo", level=logging.INFO)

def generate_clustered_memories(
    count: int, 
    cluster_count: int = 5, 
    outlier_percentage: float = 0.1,
    time_span_days: int = 30
) -> List[Memory]:
    """
    Generate memories that form clusters with some outliers for testing pruning.
    
    Args:
        count: Number of memories to generate
        cluster_count: Number of clusters to form
        outlier_percentage: Percentage of memories that should be outliers
        time_span_days: Time span in days to distribute the memories
        
    Returns:
        List of generated Memory objects
    """
    memories = []
    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_span_days)
    time_range = (end_time - start_time).total_seconds()
    
    # Define clusters (topics with related content)
    clusters = [
        {
            "name": "Technology Preferences",
            "templates": [
                "User prefers {tech_item} over alternatives.",
                "User mentioned they find {tech_item} {evaluation}.",
                "According to conversation, user's experience with {tech_item} has been {evaluation}.",
                "User's preference for {tech_item} is {level}.",
                "User has {frequency} used {tech_item} in the past.",
            ],
            "items": ["smartphones", "laptops", "tablets", "smart watches", "wireless earbuds", 
                     "smart home devices", "voice assistants", "streaming services"],
            "evaluations": ["excellent", "very good", "satisfactory", "disappointing", 
                           "frustrating", "impressive", "innovative", "outdated"],
            "levels": ["strong", "moderate", "weak", "growing", "declining", "consistent"],
            "frequencies": ["frequently", "occasionally", "rarely", "never", "always"],
        },
        {
            "name": "Travel Experiences",
            "templates": [
                "User traveled to {location} and found it {evaluation}.",
                "User expressed {level} interest in visiting {location}.",
                "User {evaluation} their trip to {location} last {time_period}.",
                "User has been to {location} {frequency}.",
                "User mentioned that {location} is known for {feature}.",
            ],
            "locations": ["Paris", "Tokyo", "New York", "London", "Sydney", "Rome", "Dubai", 
                         "Barcelona", "San Francisco", "Berlin"],
            "evaluations": ["enjoyed", "loved", "disliked", "was disappointed by", 
                           "was impressed by", "found relaxing", "found stressful"],
            "levels": ["high", "moderate", "low", "increasing", "decreasing"],
            "time_periods": ["month", "year", "summer", "winter", "spring", "fall"],
            "frequencies": ["multiple times", "once", "twice", "never", "annually"],
            "features": ["cuisine", "architecture", "museums", "nightlife", "shopping", 
                        "beaches", "friendly locals", "historical sites"],
        },
        {
            "name": "Food Preferences",
            "templates": [
                "User {evaluation} {cuisine} food.",
                "User mentioned their favorite dish is {dish} from {cuisine} cuisine.",
                "User has a {level} preference for {cuisine} restaurants.",
                "User {frequency} cooks {cuisine} food at home.",
                "User is {evaluation} about trying new {cuisine} recipes.",
            ],
            "cuisines": ["Italian", "Japanese", "Mexican", "Indian", "Chinese", "French", 
                        "Thai", "Mediterranean", "Korean", "American"],
            "dishes": ["pasta", "sushi", "tacos", "curry", "dumplings", "croissants", 
                      "pad thai", "hummus", "bibimbap", "burgers"],
            "evaluations": ["enjoys", "loves", "prefers", "dislikes", "is neutral about", 
                           "is passionate about", "is curious about"],
            "levels": ["strong", "moderate", "mild", "growing", "consistent"],
            "frequencies": ["frequently", "occasionally", "rarely", "never", "always"],
        },
        {
            "name": "Work Habits",
            "templates": [
                "User typically works from {location} {frequency}.",
                "User's productivity is {level} when working {time_of_day}.",
                "User prefers to have meetings in the {time_of_day}.",
                "User finds {work_activity} {evaluation} for productivity.",
                "User uses {tool} for {work_activity} {frequency}.",
            ],
            "locations": ["home", "office", "café", "co-working space", "outdoors"],
            "work_activities": ["video meetings", "email", "focused work", "brainstorming", 
                               "planning", "reporting", "team collaboration"],
            "tools": ["Slack", "Zoom", "Google Docs", "Microsoft Office", "project management software", 
                     "note-taking apps", "time tracking tools"],
            "evaluations": ["essential", "helpful", "distracting", "ineffective", "valuable"],
            "levels": ["highest", "high", "moderate", "low", "inconsistent"],
            "frequencies": ["daily", "several times a week", "weekly", "rarely", "as needed"],
            "time_of_day": ["morning", "afternoon", "evening", "late night", "early morning"],
        },
        {
            "name": "Entertainment Preferences",
            "templates": [
                "User enjoys watching {genre} movies and TV shows.",
                "User has been {frequency} watching {show_type} on {platform}.",
                "User expressed {level} interest in {genre} content.",
                "User mentioned that {show} is their favorite {show_type}.",
                "User finds {genre} {evaluation} compared to other genres.",
            ],
            "genres": ["sci-fi", "comedy", "drama", "thriller", "documentary", "action", 
                      "romance", "horror", "fantasy", "animated"],
            "shows": ["Stranger Things", "The Office", "Game of Thrones", "Breaking Bad", 
                     "The Crown", "Friends", "The Mandalorian", "Black Mirror"],
            "platforms": ["Netflix", "HBO", "Disney+", "Amazon Prime", "Hulu", "Apple TV+"],
            "show_types": ["series", "movies", "documentaries", "reality shows", "animated shows"],
            "evaluations": ["more engaging", "more entertaining", "more thought-provoking", 
                           "less interesting", "more relaxing", "more exciting"],
            "levels": ["high", "moderate", "low", "growing", "declining", "passionate"],
            "frequencies": ["regularly", "occasionally", "binge", "selectively", "rarely"],
        },
    ]
    
    # Calculate counts
    outlier_count = int(count * outlier_percentage)
    cluster_memory_count = count - outlier_count
    memories_per_cluster = cluster_memory_count // cluster_count
    
    # Generate cluster-based memories
    for cluster_idx in range(min(cluster_count, len(clusters))):
        cluster = clusters[cluster_idx]
        
        for i in range(memories_per_cluster):
            # Select a random template from the cluster
            template = random.choice(cluster["templates"])
            content = template
            
            # Fill in template placeholders with cluster-specific values
            for key, values in cluster.items():
                if key != "name" and key != "templates":
                    placeholder = "{" + key.rstrip("s") + "}"
                    if placeholder in content:
                        content = content.replace(placeholder, random.choice(values))
            
            # Create random timestamp within the time range
            # Cluster memories are weighted toward more recent times
            recency_bias = random.random() * 0.7  # 0-0.7 range for recency bias
            time_position = recency_bias + (random.random() * 0.3)  # 0-1 range with bias toward recent
            random_time_offset = time_position * time_range
            timestamp = start_time + timedelta(seconds=random_time_offset)
            
            # Create memory with cluster-specific importance
            memory = Memory(
                content=content,
                source=random.choice(["user_input", "system_observation", "data_analysis"]),
                creation_time=timestamp,
                importance=random.uniform(0.5, 0.9),  # Cluster memories are more important
                metadata={
                    "confidence": random.uniform(0.7, 1.0),
                    "category": cluster["name"],
                    "cluster_id": cluster_idx,
                    "is_outlier": False
                }
            )
            memories.append(memory)
    
    # Generate outlier memories (unrelated to any cluster)
    outlier_templates = [
        "User mentioned something about {random_topic} once.",
        "There was a brief discussion about {random_topic}.",
        "User referred to {random_topic} in passing.",
        "A tangential mention of {random_topic} occurred.",
        "User briefly touched on the subject of {random_topic}."
    ]
    
    random_topics = [
        "underwater basket weaving", "the migration patterns of swallows",
        "the proper way to fold paper airplanes", "ancient Egyptian calendar systems",
        "the history of pocket calculators", "competitive stamp collecting",
        "artisanal pencil sharpening", "the precise definition of sandwich",
        "optimal methods for sorting socks", "regulations for miniature golf courses",
        "the evolutionary purpose of hiccups", "differences in doorknob designs"
    ]
    
    for i in range(outlier_count):
        template = random.choice(outlier_templates)
        content = template.replace("{random_topic}", random.choice(random_topics))
        
        # Outliers are distributed randomly throughout the time period
        random_time_offset = random.random() * time_range
        timestamp = start_time + timedelta(seconds=random_time_offset)
        
        # Outliers typically have lower importance
        memory = Memory(
            content=content,
            source=random.choice(["user_input", "system_observation", "data_analysis"]),
            creation_time=timestamp,
            importance=random.uniform(0.1, 0.4),  # Lower importance
            metadata={
                "confidence": random.uniform(0.3, 0.6),  # Lower confidence
                "category": "Miscellaneous",
                "is_outlier": True
            }
        )
        memories.append(memory)
    
    # Add some duplicate/near-duplicate memories for redundancy testing
    duplicate_count = min(20, int(count * 0.05))  # 5% duplicates, max 20
    for i in range(duplicate_count):
        if len(memories) > 0:
            # Pick a random memory to duplicate
            original = random.choice(memories)
            duplicate = Memory(
                content=original.content,  # Same content
                source=original.source,
                # Slightly different timestamp (a few minutes to a few hours later)
                creation_time=original.creation_time + timedelta(minutes=random.randint(5, 180)),
                importance=original.importance * random.uniform(0.8, 1.2),  # Similar importance
                metadata={
                    **original.metadata,
                    "is_duplicate": True
                }
            )
            memories.append(duplicate)
    
    # Sort by creation time
    memories.sort(key=lambda m: m.creation_time)
    return memories

def analyze_memory_distribution(memories: List[Memory]) -> Dict[str, Any]:
    """Analyze the distribution of memories for reporting."""
    result = {
        "total_count": len(memories),
        "categories": {},
        "sources": {},
        "time_distribution": {
            "newest": None,
            "oldest": None,
            "avg_age_days": 0
        },
        "importance": {
            "high": 0,  # >0.7
            "medium": 0,  # 0.4-0.7
            "low": 0,  # <0.4
            "avg": 0
        },
        "outliers": 0,
        "duplicates": 0
    }
    
    if not memories:
        return result
    
    # Initialize with first memory
    newest_time = oldest_time = memories[0].creation_time
    importance_sum = 0
    
    for memory in memories:
        # Categories
        category = memory.metadata.get("category", "Uncategorized")
        result["categories"][category] = result["categories"].get(category, 0) + 1
        
        # Sources
        result["sources"][memory.source] = result["sources"].get(memory.source, 0) + 1
        
        # Time distribution
        if memory.creation_time > newest_time:
            newest_time = memory.creation_time
        if memory.creation_time < oldest_time:
            oldest_time = memory.creation_time
            
        # Importance
        importance_sum += memory.importance
        if memory.importance > 0.7:
            result["importance"]["high"] += 1
        elif memory.importance >= 0.4:
            result["importance"]["medium"] += 1
        else:
            result["importance"]["low"] += 1
            
        # Outliers and duplicates
        if memory.metadata.get("is_outlier", False):
            result["outliers"] += 1
        if memory.metadata.get("is_duplicate", False):
            result["duplicates"] += 1
    
    # Calculate averages
    result["time_distribution"]["newest"] = newest_time
    result["time_distribution"]["oldest"] = oldest_time
    time_span = (newest_time - oldest_time).total_seconds()
    result["time_distribution"]["span_days"] = time_span / (60 * 60 * 24)
    result["importance"]["avg"] = importance_sum / len(memories)
    
    return result

def plot_memory_distribution(before_analysis: Dict[str, Any], after_analysis: Dict[str, Any], 
                            pruning_strategy: str, filename: str) -> None:
    """Plot the distribution of memories before and after pruning."""
    plt.figure(figsize=(15, 10))
    
    # 1. Create category distribution subplot
    plt.subplot(2, 2, 1)
    before_categories = before_analysis["categories"]
    after_categories = after_analysis["categories"]
    
    # Combine keys from both dictionaries
    all_categories = sorted(set(list(before_categories.keys()) + list(after_categories.keys())))
    
    # Prepare data for plotting
    before_counts = [before_categories.get(cat, 0) for cat in all_categories]
    after_counts = [after_categories.get(cat, 0) for cat in all_categories]
    
    x = np.arange(len(all_categories))
    width = 0.35
    
    plt.bar(x - width/2, before_counts, width, label='Before Pruning')
    plt.bar(x + width/2, after_counts, width, label='After Pruning')
    
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title('Memory Categories Before and After Pruning')
    plt.xticks(x, all_categories, rotation=45, ha='right')
    plt.legend()
    
    # 2. Create importance distribution subplot
    plt.subplot(2, 2, 2)
    importance_labels = ['Low', 'Medium', 'High']
    before_importance = [before_analysis["importance"]["low"], 
                        before_analysis["importance"]["medium"], 
                        before_analysis["importance"]["high"]]
    after_importance = [after_analysis["importance"]["low"], 
                       after_analysis["importance"]["medium"], 
                       after_analysis["importance"]["high"]]
    
    x = np.arange(len(importance_labels))
    plt.bar(x - width/2, before_importance, width, label='Before Pruning')
    plt.bar(x + width/2, after_importance, width, label='After Pruning')
    
    plt.xlabel('Importance Level')
    plt.ylabel('Count')
    plt.title('Memory Importance Before and After Pruning')
    plt.xticks(x, importance_labels)
    plt.legend()
    
    # 3. Create outliers and duplicates subplot
    plt.subplot(2, 2, 3)
    special_labels = ['Outliers', 'Duplicates']
    before_special = [before_analysis["outliers"], before_analysis["duplicates"]]
    after_special = [after_analysis["outliers"], after_analysis["duplicates"]]
    
    x = np.arange(len(special_labels))
    plt.bar(x - width/2, before_special, width, label='Before Pruning')
    plt.bar(x + width/2, after_special, width, label='After Pruning')
    
    plt.xlabel('Memory Type')
    plt.ylabel('Count')
    plt.title('Special Memory Types Before and After Pruning')
    plt.xticks(x, special_labels)
    plt.legend()
    
    # 4. Create sources subplot
    plt.subplot(2, 2, 4)
    before_sources = before_analysis["sources"]
    after_sources = after_analysis["sources"]
    
    all_sources = sorted(set(list(before_sources.keys()) + list(after_sources.keys())))
    before_source_counts = [before_sources.get(src, 0) for src in all_sources]
    after_source_counts = [after_sources.get(src, 0) for src in all_sources]
    
    x = np.arange(len(all_sources))
    plt.bar(x - width/2, before_source_counts, width, label='Before Pruning')
    plt.bar(x + width/2, after_source_counts, width, label='After Pruning')
    
    plt.xlabel('Sources')
    plt.ylabel('Count')
    plt.title('Memory Sources Before and After Pruning')
    plt.xticks(x, all_sources, rotation=45, ha='right')
    plt.legend()
    
    # Add super title
    plt.suptitle(f'Memory Distribution: {pruning_strategy} Strategy', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    plt.savefig(filename)
    logger.info(f"Memory distribution plot saved to '{filename}'")
    plt.close()

def run_pruning_experiment(
    pruning_strategy: PruningStrategy, 
    target_reduction: float,
    memory_count: int = 200,
    plot_filename: str = None
) -> Dict[str, Any]:
    """
    Run an experiment with a specific pruning strategy.
    
    Args:
        pruning_strategy: The pruning strategy to test
        target_reduction: Target percentage of memories to remove
        memory_count: Number of test memories to generate
        plot_filename: Filename to save the distribution plot
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(f"\n{'='*80}\nRunning pruning experiment with {pruning_strategy.name} strategy")
    logger.info(f"Target reduction: {target_reduction*100:.1f}%")
    
    # Initialize core components
    embedding_model = EmbeddingModel()
    memory_store = MemoryStore(embedding_model)
    
    # Set up entropy measures
    embedding_entropy = EmbeddingEntropyMeasure()
    semantic_entropy = SemanticCoherenceEntropyMeasure()
    temporal_entropy = TemporalCoherenceEntropyMeasure()
    
    # Create entropy calculator with weights based on strategy
    weights = {"embedding": 0.33, "semantic": 0.33, "temporal": 0.33}
    
    # Adjust weights based on strategy
    if pruning_strategy == PruningStrategy.OUTLIER_REMOVAL:
        weights = {"embedding": 0.6, "semantic": 0.3, "temporal": 0.1}
    elif pruning_strategy == PruningStrategy.REDUNDANCY_REMOVAL:
        weights = {"embedding": 0.3, "semantic": 0.6, "temporal": 0.1}
    elif pruning_strategy == PruningStrategy.TEMPORAL_DECAY:
        weights = {"embedding": 0.1, "semantic": 0.3, "temporal": 0.6}
    
    entropy_calculator = EntropyCalculator(
        measures={
            "embedding": embedding_entropy,
            "semantic": semantic_entropy,
            "temporal": temporal_entropy
        },
        weights=weights
    )
    
    # Create entropy pruner
    entropy_pruner = EntropyPruner(
        memory_store=memory_store,
        entropy_calculator=entropy_calculator
    )
    
    # Generate and add test memories
    logger.info(f"Generating {memory_count} test memories...")
    test_memories = generate_clustered_memories(
        count=memory_count,
        cluster_count=5,
        outlier_percentage=0.1,
        time_span_days=30
    )
    
    for memory in test_memories:
        memory_store.add(memory)
    
    logger.info(f"Added {len(test_memories)} memories to store")
    
    # Calculate initial entropy
    initial_entropy_values = entropy_calculator.calculate_memory_store_entropy(memory_store)
    initial_total_entropy = entropy_calculator.calculate_total_entropy(initial_entropy_values)
    
    logger.info(f"Initial entropy values: {json.dumps(initial_entropy_values, indent=2)}")
    logger.info(f"Initial total entropy: {initial_total_entropy:.4f}")
    
    # Analyze initial memory distribution
    initial_memories = memory_store.get_all()
    before_analysis = analyze_memory_distribution(initial_memories)
    
    # Apply pruning
    logger.info(f"Applying {pruning_strategy.name} pruning strategy...")
    start_time = time.time()
    removed_count = entropy_pruner.prune(
        strategy=pruning_strategy,
        target_reduction=target_reduction
    )
    pruning_time = time.time() - start_time
    
    # Calculate post-pruning entropy
    remaining_memories = memory_store.get_all()
    final_entropy_values = entropy_calculator.calculate_memory_store_entropy(memory_store)
    final_total_entropy = entropy_calculator.calculate_total_entropy(final_entropy_values)
    
    logger.info(f"Final entropy values: {json.dumps(final_entropy_values, indent=2)}")
    logger.info(f"Final total entropy: {final_total_entropy:.4f}")
    
    # Analyze final memory distribution
    after_analysis = analyze_memory_distribution(remaining_memories)
    
    # Calculate removal statistics
    removal_percentage = removed_count / len(initial_memories) * 100
    entropy_reduction = ((initial_total_entropy - final_total_entropy) / initial_total_entropy) * 100
    
    # Print results
    logger.info(f"Pruning completed in {pruning_time:.4f} seconds")
    logger.info(f"Removed {removed_count} memories ({removal_percentage:.1f}%) of {len(initial_memories)} total")
    logger.info(f"Entropy reduction: {entropy_reduction:.1f}%")
    
    # Create table of category changes
    categories_table = []
    for category in sorted(set(before_analysis["categories"].keys()) | set(after_analysis["categories"].keys())):
        before = before_analysis["categories"].get(category, 0)
        after = after_analysis["categories"].get(category, 0)
        change = after - before
        change_pct = (change / before * 100) if before > 0 else float('inf')
        categories_table.append([
            category, before, after, change, f"{change_pct:.1f}%"
        ])
    
    logger.info("\nCategory Changes:")
    logger.info(tabulate(categories_table, 
                       headers=["Category", "Before", "After", "Change", "Change %"],
                       tablefmt="grid"))
    
    # Create importance distribution table
    importance_table = [
        ["Low", before_analysis["importance"]["low"], after_analysis["importance"]["low"]],
        ["Medium", before_analysis["importance"]["medium"], after_analysis["importance"]["medium"]],
        ["High", before_analysis["importance"]["high"], after_analysis["importance"]["high"]],
        ["Average", f"{before_analysis['importance']['avg']:.3f}", f"{after_analysis['importance']['avg']:.3f}"]
    ]
    
    logger.info("\nImportance Distribution:")
    logger.info(tabulate(importance_table,
                       headers=["Level", "Before", "After"],
                       tablefmt="grid"))
    
    # Plot memory distribution
    if plot_filename:
        plot_memory_distribution(before_analysis, after_analysis, 
                               pruning_strategy.name, plot_filename)
    
    # Return experiment results
    return {
        "strategy": pruning_strategy.name,
        "target_reduction": target_reduction,
        "initial_memories": len(initial_memories),
        "removed_count": removed_count,
        "removal_percentage": removal_percentage,
        "initial_entropy": initial_total_entropy,
        "final_entropy": final_total_entropy,
        "entropy_reduction": entropy_reduction,
        "pruning_time": pruning_time,
        "before_analysis": before_analysis,
        "after_analysis": after_analysis
    }

def compare_strategies() -> None:
    """Run experiments with different pruning strategies and compare results."""
    memory_count = 200
    target_reduction = 0.25  # 25% reduction target
    
    strategies = [
        PruningStrategy.TEMPORAL_DECAY,
        PruningStrategy.OUTLIER_REMOVAL,
        PruningStrategy.REDUNDANCY_REMOVAL,
        PruningStrategy.AGGRESSIVE
    ]
    
    results = []
    
    for strategy in strategies:
        plot_filename = f"pruning_{strategy.name.lower()}.png"
        result = run_pruning_experiment(
            pruning_strategy=strategy,
            target_reduction=target_reduction,
            memory_count=memory_count,
            plot_filename=plot_filename
        )
        results.append(result)
    
    # Create comparison table
    comparison_table = []
    for result in results:
        comparison_table.append([
            result["strategy"],
            result["removed_count"],
            f"{result['removal_percentage']:.1f}%",
            f"{result['initial_entropy']:.4f}",
            f"{result['final_entropy']:.4f}",
            f"{result['entropy_reduction']:.1f}%",
            f"{result['pruning_time']:.4f}s"
        ])
    
    logger.info("\n\nStrategy Comparison:")
    logger.info(tabulate(comparison_table,
                       headers=["Strategy", "Removed", "Removal %", "Initial Entropy", 
                               "Final Entropy", "Entropy Reduction", "Time"],
                       tablefmt="grid"))
    
    # Create and save comparison chart
    plt.figure(figsize=(12, 8))
    
    # Prepare data for comparison chart
    strategy_names = [result["strategy"] for result in results]
    removal_percentages = [result["removal_percentage"] for result in results]
    entropy_reductions = [result["entropy_reduction"] for result in results]
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    plt.bar(x - width/2, removal_percentages, width, label='Removal %')
    plt.bar(x + width/2, entropy_reductions, width, label='Entropy Reduction %')
    
    plt.xlabel('Pruning Strategy')
    plt.ylabel('Percentage')
    plt.title('Comparison of Pruning Strategies')
    plt.xticks(x, strategy_names)
    plt.legend()
    
    plt.savefig("pruning_strategy_comparison.png")
    logger.info("Strategy comparison chart saved to 'pruning_strategy_comparison.png'")
    plt.close()
    
    # Additional insight: Average importance of removed memories by strategy
    logger.info("\nRemoved Memory Characteristics by Strategy:")
    
    for result in results:
        strategy = result["strategy"]
        before = result["before_analysis"]
        after = result["after_analysis"]
        
        removed_low = before["importance"]["low"] - after["importance"]["low"]
        removed_med = before["importance"]["medium"] - after["importance"]["medium"]
        removed_high = before["importance"]["high"] - after["importance"]["high"]
        total_removed = removed_low + removed_med + removed_high
        
        if total_removed > 0:
            low_pct = removed_low / total_removed * 100
            med_pct = removed_med / total_removed * 100
            high_pct = removed_high / total_removed * 100
            
            logger.info(f"\n{strategy}:")
            logger.info(f"  Low importance: {removed_low} ({low_pct:.1f}%)")
            logger.info(f"  Medium importance: {removed_med} ({med_pct:.1f}%)")
            logger.info(f"  High importance: {removed_high} ({high_pct:.1f}%)")
            
            # Outliers and duplicates
            removed_outliers = before["outliers"] - after["outliers"]
            removed_duplicates = before["duplicates"] - after["duplicates"]
            
            logger.info(f"  Removed outliers: {removed_outliers} of {before['outliers']} ({removed_outliers/before['outliers']*100 if before['outliers']>0 else 0:.1f}%)")
            logger.info(f"  Removed duplicates: {removed_duplicates} of {before['duplicates']} ({removed_duplicates/before['duplicates']*100 if before['duplicates']>0 else 0:.1f}%)")

def run_demo() -> None:
    """Run the entropy pruning system demonstration."""
    logger.info("Starting Entropy-Based Memory Pruning Demo")
    
    # Compare different pruning strategies
    compare_strategies()
    
    # Test incremental pruning with outlier removal
    logger.info("\n\nTesting Incremental Pruning with Outlier Removal")
    memory_count = 300
    
    # Initialize components
    embedding_model = EmbeddingModel()
    memory_store = MemoryStore(embedding_model)
    
    # Set up entropy calculator
    entropy_calculator = EntropyCalculator(
        measures={
            "embedding": EmbeddingEntropyMeasure(),
            "semantic": SemanticCoherenceEntropyMeasure(),
            "temporal": TemporalCoherenceEntropyMeasure()
        },
        weights={"embedding": 0.6, "semantic": 0.3, "temporal": 0.1}
    )
    
    # Create entropy pruner
    entropy_pruner = EntropyPruner(
        memory_store=memory_store,
        entropy_calculator=entropy_calculator
    )
    
    # Generate and add test memories
    logger.info(f"Generating {memory_count} test memories for incremental test...")
    test_memories = generate_clustered_memories(
        count=memory_count,
        cluster_count=5,
        outlier_percentage=0.15,  # Higher outlier percentage for this test
        time_span_days=30
    )
    
    for memory in test_memories:
        memory_store.add(memory)
    
    # Track entropy over incremental pruning steps
    initial_memories = len(memory_store.get_all())
    entropy_values = []
    memory_counts = []
    
    # Calculate initial entropy
    initial_entropy = entropy_calculator.calculate_total_entropy(
        entropy_calculator.calculate_memory_store_entropy(memory_store)
    )
    
    entropy_values.append(initial_entropy)
    memory_counts.append(initial_memories)
    
    logger.info(f"Initial memory count: {initial_memories}")
    logger.info(f"Initial entropy: {initial_entropy:.4f}")
    
    # Perform incremental pruning
    incremental_steps = 5
    step_reduction = 0.1  # 10% per step
    
    for step in range(incremental_steps):
        logger.info(f"\nIncremental pruning step {step+1}/{incremental_steps}...")
        
        # Apply pruning
        removed = entropy_pruner.prune(
            strategy=PruningStrategy.OUTLIER_REMOVAL,
            target_reduction=step_reduction
        )
        
        # Calculate new entropy
        current_count = len(memory_store.get_all())
        current_entropy = entropy_calculator.calculate_total_entropy(
            entropy_calculator.calculate_memory_store_entropy(memory_store)
        )
        
        entropy_values.append(current_entropy)
        memory_counts.append(current_count)
        
        logger.info(f"Removed {removed} memories, {current_count} remaining")
        logger.info(f"Current entropy: {current_entropy:.4f}")
    
    # Plot incremental pruning results
    plt.figure(figsize=(10, 6))
    
    # Plot entropy values
    plt.subplot(1, 2, 1)
    plt.plot(entropy_values, 'b-o')
    plt.xlabel('Pruning Step')
    plt.ylabel('Total Entropy')
    plt.title('Entropy Change with Incremental Pruning')
    plt.grid(True, alpha=0.3)
    
    # Plot memory counts
    plt.subplot(1, 2, 2)
    plt.plot(memory_counts, 'r-o')
    plt.xlabel('Pruning Step')
    plt.ylabel('Memory Count')
    plt.title('Memory Count with Incremental Pruning')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('incremental_pruning.png')
    logger.info("Incremental pruning chart saved to 'incremental_pruning.png'")
    plt.close()
    
    logger.info("\nEntropy-Based Memory Pruning Demo completed successfully")

if __name__ == "__main__":
    run_demo() 