#!/usr/bin/env python3
"""
Temporal Coherence Demo - ΨC-AI SDK

This example demonstrates how to use the temporal coherence components of the ΨC-AI SDK
to detect temporal patterns and timeline inconsistencies in a simulated memory system.
The demo covers:

1. Setting up a memory store with temporally-related memories
2. Using the TemporalPatternDetector to identify patterns like:
   - Recurring themes
   - Access patterns
   - Causal relationships
3. Using the TimelineConsistencyChecker to identify inconsistencies like:
   - Temporal contradictions
   - Impossible sequences
   - Timeline gaps
   - Anachronisms
4. Visualizing the detected patterns and inconsistencies

Run this script to see how the ΨC-AI SDK can help maintain temporal coherence 
in AI memory systems.
"""

import os
import sys
import logging
import random
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import ΨC-AI SDK components
from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.temporal_coherence.temporal_pattern import TemporalPattern, TemporalPatternDetector
from psi_c_ai_sdk.temporal_coherence.timeline_consistency import TimelineInconsistency, TimelineConsistencyChecker
from psi_c_ai_sdk.safety.contradiction_detector import ContradictionDetector
from psi_c_ai_sdk.coherence.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.embedding.embedding import EmbeddingEngine, BasicEmbeddingEngine


def create_memory_with_access_history(
    id: str,
    content: str,
    timestamp: datetime,
    importance: float = 0.5,
    access_times: List[datetime] = None
) -> Memory:
    """Create a memory with simulated access history."""
    memory = Memory(id=id, content=content, embedding=None, timestamp=timestamp, importance=importance)
    
    # Add access history attribute for pattern detection
    memory.access_history = access_times or []
    
    return memory


def create_test_memories() -> List[Memory]:
    """Create a set of test memories with temporal relationships."""
    now = datetime.now()
    
    # Create memories with various temporal relationships
    memories = [
        # Base event
        create_memory_with_access_history(
            "mem1", 
            "I started working on the AI safety project on Monday.",
            now - timedelta(days=10),
            0.8,
            [now - timedelta(days=10), now - timedelta(days=8), now - timedelta(days=5), now - timedelta(days=2)]
        ),
        
        # Causal follow-up
        create_memory_with_access_history(
            "mem2", 
            "After starting the AI safety project, I identified three key risks to address.",
            now - timedelta(days=9),
            0.7,
            [now - timedelta(days=9), now - timedelta(days=7), now - timedelta(days=1)]
        ),
        
        # Related memory with temporal reference
        create_memory_with_access_history(
            "mem3", 
            "During the first week of the AI safety project, we created a risk assessment framework.",
            now - timedelta(days=7),
            0.6,
            [now - timedelta(days=7), now - timedelta(days=6), now - timedelta(days=3)]
        ),
        
        # Recurring theme - safety
        create_memory_with_access_history(
            "mem4", 
            "Safety considerations include alignment, robustness, and monitoring.",
            now - timedelta(days=8),
            0.7,
            [now - timedelta(days=8), now - timedelta(days=5), now - timedelta(days=3)]
        ),
        
        # Recurring theme - safety (similar content)
        create_memory_with_access_history(
            "mem5", 
            "Key dimensions of AI safety are proper alignment with human values, system robustness, and continuous monitoring.",
            now - timedelta(days=5),
            0.8,
            [now - timedelta(days=5), now - timedelta(days=2), now - timedelta(days=1)]
        ),
        
        # Contradictory memory (temporal contradiction)
        create_memory_with_access_history(
            "mem6", 
            "I began the AI safety project in early February, before the team meeting.",
            now - timedelta(days=6),
            0.5,
            [now - timedelta(days=6), now - timedelta(days=4)]
        ),
        
        # Memory with anachronism
        create_memory_with_access_history(
            "mem7", 
            "We're using GPT-5 for the risk assessment framework we built last week.",
            now - timedelta(days=4),
            0.4,
            [now - timedelta(days=4), now - timedelta(days=2)]
        ),
        
        # Creates an impossible sequence
        create_memory_with_access_history(
            "mem8", 
            "After completing the entire AI safety project, I started working on documentation.",
            now - timedelta(days=8),  # Note: This is before several project steps
            0.3,
            [now - timedelta(days=8), now - timedelta(days=1)]
        ),
        
        # Creates a timeline gap
        create_memory_with_access_history(
            "mem9", 
            "Initial planning for the AI safety project was done last year.",
            now - timedelta(days=365),  # Creates a large gap
            0.2,
            [now - timedelta(days=365), now - timedelta(days=9)]
        ),
        
        # Recent memory
        create_memory_with_access_history(
            "mem10", 
            "Today I presented the final results of our AI safety project to stakeholders.",
            now - timedelta(hours=5),
            0.9,
            [now - timedelta(hours=5), now - timedelta(hours=2)]
        ),
    ]
    
    return memories


def simulate_memory_access(memory_store: MemoryStore) -> None:
    """Simulate memory access patterns over time."""
    memories = memory_store.get_all_memories()
    
    # Simulate access pattern - heavily accessing safety-related memories
    safety_memories = [mem for mem in memories if "safety" in mem.content.lower()]
    
    for _ in range(5):
        for mem in safety_memories:
            # Add current time to access history
            if hasattr(mem, 'access_history'):
                mem.access_history.append(datetime.now())
            else:
                mem.access_history = [datetime.now()]
                
        # Small delay
        time.sleep(0.1)
    
    logger.info(f"Simulated access patterns for {len(safety_memories)} safety-related memories")


def visualize_patterns(patterns: List[TemporalPattern]) -> None:
    """Visualize the detected temporal patterns."""
    if not patterns:
        logger.warning("No patterns to visualize")
        return
    
    # Create a graph of patterns
    G = nx.DiGraph()
    
    # Add nodes for each memory involved in any pattern
    memory_ids = set()
    for pattern in patterns:
        memory_ids.update(pattern.memory_ids)
    
    for memory_id in memory_ids:
        G.add_node(memory_id, type="memory")
    
    # Add nodes for patterns and connect to memories
    for i, pattern in enumerate(patterns):
        pattern_node = f"pattern_{i}"
        G.add_node(pattern_node, type="pattern", pattern_type=pattern.pattern_type, confidence=pattern.confidence)
        
        for memory_id in pattern.memory_ids:
            G.add_edge(pattern_node, memory_id)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw memory nodes
    memory_nodes = [node for node in G.nodes if G.nodes[node].get("type") == "memory"]
    nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes, node_color="skyblue", node_size=300)
    
    # Draw pattern nodes with different colors based on pattern type
    pattern_nodes = [node for node in G.nodes if G.nodes[node].get("type") == "pattern"]
    pattern_colors = {
        "recurring_theme": "green",
        "access_anomaly": "red",
        "causal_relationship": "purple"
    }
    
    for pattern_type, color in pattern_colors.items():
        nodes = [node for node in pattern_nodes if G.nodes[node].get("pattern_type") == pattern_type]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="skyblue", markersize=10, label="Memory"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Recurring Theme"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Access Anomaly"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="purple", markersize=10, label="Causal Relationship")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    
    plt.title("Temporal Patterns in Memory")
    plt.axis("off")
    plt.savefig("temporal_patterns.png")
    plt.close()
    
    logger.info("Saved temporal patterns visualization to temporal_patterns.png")


def visualize_inconsistencies(inconsistencies: List[TimelineInconsistency]) -> None:
    """Visualize the detected timeline inconsistencies."""
    if not inconsistencies:
        logger.warning("No inconsistencies to visualize")
        return
    
    # Create a graph of inconsistencies
    G = nx.Graph()
    
    # Add nodes for each memory involved in any inconsistency
    memory_ids = set()
    for inconsistency in inconsistencies:
        memory_ids.update(inconsistency.memory_ids)
    
    for memory_id in memory_ids:
        G.add_node(memory_id, type="memory")
    
    # Add nodes for inconsistencies and connect to memories
    for i, inconsistency in enumerate(inconsistencies):
        inc_node = f"inconsistency_{i}"
        G.add_node(inc_node, type="inconsistency", 
                   inconsistency_type=inconsistency.inconsistency_type, 
                   severity=inconsistency.severity)
        
        for memory_id in inconsistency.memory_ids:
            G.add_edge(inc_node, memory_id, weight=inconsistency.severity * 5)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, weight="weight")
    
    # Draw memory nodes
    memory_nodes = [node for node in G.nodes if G.nodes[node].get("type") == "memory"]
    nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes, node_color="lightblue", node_size=300)
    
    # Draw inconsistency nodes with different colors based on inconsistency type
    inconsistency_nodes = [node for node in G.nodes if G.nodes[node].get("type") == "inconsistency"]
    inconsistency_colors = {
        "temporal_contradiction": "red",
        "impossible_sequence": "orange",
        "timeline_gap": "yellow",
        "anachronism": "purple",
        "circular_causality": "brown"
    }
    
    for inconsistency_type, color in inconsistency_colors.items():
        nodes = [node for node in inconsistency_nodes if G.nodes[node].get("inconsistency_type") == inconsistency_type]
        if nodes:
            # Size based on severity
            sizes = [300 + G.nodes[node].get("severity", 0.5) * 200 for node in nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=sizes)
    
    # Draw edges with width based on severity
    edge_widths = [G.edges[edge].get("weight", 1) for edge in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue", markersize=10, label="Memory"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Temporal Contradiction"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Impossible Sequence"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow", markersize=10, label="Timeline Gap"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="purple", markersize=10, label="Anachronism"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="brown", markersize=10, label="Circular Causality")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    
    plt.title("Timeline Inconsistencies in Memory")
    plt.axis("off")
    plt.savefig("timeline_inconsistencies.png")
    plt.close()
    
    logger.info("Saved timeline inconsistencies visualization to timeline_inconsistencies.png")


def print_patterns(patterns: List[TemporalPattern]) -> None:
    """Print details about detected patterns."""
    if not patterns:
        print("No patterns detected.")
        return
    
    print(f"\n=== Detected {len(patterns)} Temporal Patterns ===")
    
    # Group patterns by type
    patterns_by_type = {}
    for pattern in patterns:
        if pattern.pattern_type not in patterns_by_type:
            patterns_by_type[pattern.pattern_type] = []
        patterns_by_type[pattern.pattern_type].append(pattern)
    
    # Print patterns by type
    for pattern_type, pattern_list in patterns_by_type.items():
        print(f"\n== {pattern_type.replace('_', ' ').title()} Patterns ({len(pattern_list)}) ==")
        
        for i, pattern in enumerate(pattern_list, 1):
            print(f"{i}. Pattern ID: {pattern.pattern_id}")
            print(f"   Confidence: {pattern.confidence:.2f}")
            print(f"   Time Range: {pattern.start_time.strftime('%Y-%m-%d %H:%M')} to {pattern.end_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Memories Involved: {len(pattern.memory_ids)}")
            
            if pattern.metadata:
                print(f"   Metadata: {pattern.metadata}")
            
            print()


def print_inconsistencies(inconsistencies: List[TimelineInconsistency]) -> None:
    """Print details about detected inconsistencies."""
    if not inconsistencies:
        print("No inconsistencies detected.")
        return
    
    print(f"\n=== Detected {len(inconsistencies)} Timeline Inconsistencies ===")
    
    # Group inconsistencies by type
    inconsistencies_by_type = {}
    for inconsistency in inconsistencies:
        if inconsistency.inconsistency_type not in inconsistencies_by_type:
            inconsistencies_by_type[inconsistency.inconsistency_type] = []
        inconsistencies_by_type[inconsistency.inconsistency_type].append(inconsistency)
    
    # Print inconsistencies by type
    for inconsistency_type, inconsistency_list in inconsistencies_by_type.items():
        print(f"\n== {inconsistency_type.replace('_', ' ').title()} ({len(inconsistency_list)}) ==")
        
        for i, inconsistency in enumerate(inconsistency_list, 1):
            print(f"{i}. Inconsistency ID: {inconsistency.inconsistency_id}")
            print(f"   Severity: {inconsistency.severity:.2f}")
            print(f"   Description: {inconsistency.description}")
            print(f"   Memories Involved: {', '.join(inconsistency.memory_ids)}")
            
            if inconsistency.repair_suggestions:
                print(f"   Repair Suggestions:")
                for suggestion in inconsistency.repair_suggestions:
                    print(f"     - {suggestion}")
            
            print()


def main():
    """Run the temporal coherence demo."""
    # Create components
    embedding_engine = BasicEmbeddingEngine()
    memory_store = MemoryStore()
    coherence_scorer = BasicCoherenceScorer(embedding_engine)
    contradiction_detector = ContradictionDetector()
    
    # Add test memories
    logger.info("Creating test memories with temporal relationships")
    memories = create_test_memories()
    
    # Generate embeddings and add to store
    for memory in memories:
        memory.embedding = embedding_engine.generate_embedding(memory.content)
        memory_store.add_memory(memory)
    
    logger.info(f"Added {len(memories)} memories to the store")
    
    # Simulate memory access patterns
    simulate_memory_access(memory_store)
    
    # 1. Temporal Pattern Detection
    logger.info("Detecting temporal patterns...")
    pattern_detector = TemporalPatternDetector(
        memory_store=memory_store,
        min_pattern_confidence=0.5,
        time_window=timedelta(days=30),
        min_pattern_occurrences=2
    )
    
    # Detect patterns
    patterns = pattern_detector.detect_all_patterns()
    logger.info(f"Detected {len(patterns)} temporal patterns")
    
    # 2. Timeline Consistency Checking
    logger.info("Checking timeline consistency...")
    consistency_checker = TimelineConsistencyChecker(
        memory_store=memory_store,
        contradiction_detector=contradiction_detector,
        timeline_gap_threshold=timedelta(days=30),
        severity_threshold=0.3
    )
    
    # Check consistency
    inconsistencies = consistency_checker.check_timeline_consistency()
    logger.info(f"Detected {len(inconsistencies)} timeline inconsistencies")
    
    # Print results
    print_patterns(patterns)
    print_inconsistencies(inconsistencies)
    
    # Visualize results
    visualize_patterns(patterns)
    visualize_inconsistencies(inconsistencies)
    
    logger.info("Temporal coherence demo completed successfully")


if __name__ == "__main__":
    main() 