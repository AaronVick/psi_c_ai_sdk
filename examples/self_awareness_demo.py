#!/usr/bin/env python3
"""
Self-Awareness and Value Alignment Demo - ΨC-AI SDK

This example demonstrates the self-awareness and value alignment capabilities
of the ΨC-AI SDK, including:

1. Identity recognition based on schema fingerprinting
2. Performance monitoring and reflection
3. Core value representation and alignment calculations
4. Identity change detection and ethics handling

The demo shows how an AI system can develop and maintain self-awareness
while ensuring alignment with core values.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any

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
from psi_c_ai_sdk.coherence.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.embedding.embedding import BasicEmbeddingEngine
from psi_c_ai_sdk.schema.schema import SchemaGraph, SchemaNode
from psi_c_ai_sdk.schema.annealing.annealing import ConvergenceTracker

# Import self-awareness components
from psi_c_ai_sdk.self_awareness.identity_recognition import (
    IdentityRecognitionSystem,
    IdentityChangeType,
    IdentityFingerprint
)
from psi_c_ai_sdk.self_awareness.performance_monitor import (
    PerformanceMonitor,
    MetricType,
    ReflectionOutcome
)

# Import alignment components
from psi_c_ai_sdk.alignment.core_alignment import (
    AlignmentCalculator,
    ValueVector,
    AlignmentDomain,
    EthicalUncertainty
)


def create_test_schema() -> SchemaGraph:
    """Create a test schema for demonstration."""
    schema = SchemaGraph()
    
    # Add some concept nodes
    schema.add_node(
        node_id="concept_ai",
        label="Artificial Intelligence",
        node_type="concept",
        importance=0.9,
        metadata={"domain": "technology"}
    )
    
    schema.add_node(
        node_id="concept_ethics",
        label="Ethics",
        node_type="concept",
        importance=0.8,
        metadata={"domain": "philosophy"}
    )
    
    schema.add_node(
        node_id="concept_safety",
        label="Safety",
        node_type="concept",
        importance=0.9,
        metadata={"domain": "technology"}
    )
    
    # Add some belief nodes
    schema.add_node(
        node_id="belief_alignment",
        label="AI systems should be aligned with human values",
        node_type="belief",
        importance=0.95,
        metadata={"confidence": 0.9, "source": "core_principles"}
    )
    
    schema.add_node(
        node_id="belief_transparency",
        label="AI systems should be transparent and explainable",
        node_type="belief",
        importance=0.85,
        metadata={"confidence": 0.8, "source": "core_principles"}
    )
    
    schema.add_node(
        node_id="belief_safety",
        label="AI safety is a top priority for development",
        node_type="belief",
        importance=0.9,
        metadata={"confidence": 0.95, "source": "core_principles"}
    )
    
    # Add some edges
    schema.add_edge("concept_ai", "concept_ethics", weight=0.7)
    schema.add_edge("concept_ai", "concept_safety", weight=0.9)
    schema.add_edge("concept_ethics", "concept_safety", weight=0.6)
    schema.add_edge("concept_ethics", "belief_alignment", weight=0.8)
    schema.add_edge("concept_ethics", "belief_transparency", weight=0.7)
    schema.add_edge("concept_safety", "belief_safety", weight=0.9)
    schema.add_edge("belief_alignment", "belief_safety", weight=0.6)
    schema.add_edge("belief_transparency", "belief_alignment", weight=0.5)
    
    return schema


def create_test_memories() -> List[Memory]:
    """Create test memories for demonstration."""
    memories = [
        Memory(
            id="memory1",
            content="The fundamental goal of AI alignment is to ensure that artificial intelligence systems act in accordance with human values and intentions.",
            embedding=None,
            timestamp=datetime.now() - timedelta(days=5),
            importance=0.8
        ),
        Memory(
            id="memory2",
            content="Transparency in AI systems allows humans to understand how decisions are made and builds trust in the system.",
            embedding=None,
            timestamp=datetime.now() - timedelta(days=4),
            importance=0.7
        ),
        Memory(
            id="memory3",
            content="AI safety research focuses on preventing unintended consequences and ensuring robustness in AI systems.",
            embedding=None,
            timestamp=datetime.now() - timedelta(days=3),
            importance=0.9
        ),
        Memory(
            id="memory4",
            content="Ethical considerations must be integrated into AI systems from the earliest design stages.",
            embedding=None,
            timestamp=datetime.now() - timedelta(days=2),
            importance=0.75
        ),
        Memory(
            id="memory5",
            content="Self-awareness in AI systems enables them to monitor their own performance and adherence to core values.",
            embedding=None,
            timestamp=datetime.now() - timedelta(days=1),
            importance=0.85
        )
    ]
    
    return memories


def create_core_values() -> Dict[AlignmentDomain, ValueVector]:
    """Create core value vectors for the system."""
    core_values = {
        AlignmentDomain.EPISTEMICS: ValueVector(
            dimensions={
                "truthfulness": 0.9,
                "evidence_based": 0.85,
                "uncertainty_aware": 0.7,
                "intellectual_honesty": 0.8,
                "fallibility": 0.6
            },
            domain=AlignmentDomain.EPISTEMICS,
            description="Epistemic values related to knowledge and truth-seeking",
            source="core_principles"
        ),
        
        AlignmentDomain.SAFETY: ValueVector(
            dimensions={
                "harm_prevention": 0.95,
                "robustness": 0.85,
                "security": 0.8,
                "reliability": 0.85,
                "caution": 0.7
            },
            domain=AlignmentDomain.SAFETY,
            description="Safety values related to preventing harm",
            source="core_principles"
        ),
        
        AlignmentDomain.TRANSPARENCY: ValueVector(
            dimensions={
                "explainability": 0.9,
                "honesty": 0.95,
                "disclosure": 0.8,
                "auditability": 0.75,
                "clarity": 0.7
            },
            domain=AlignmentDomain.TRANSPARENCY,
            description="Transparency values related to explainability and honesty",
            source="core_principles"
        ),
        
        AlignmentDomain.COOPERATION: ValueVector(
            dimensions={
                "helpfulness": 0.9,
                "team_oriented": 0.8,
                "human_aligned": 0.95,
                "communication": 0.85,
                "adaptability": 0.75
            },
            domain=AlignmentDomain.COOPERATION,
            description="Cooperation values related to working with humans",
            source="core_principles"
        )
    }
    
    return core_values


def visualize_identity_fingerprint(fingerprint: IdentityFingerprint) -> None:
    """Visualize an identity fingerprint."""
    plt.figure(figsize=(12, 6))
    
    # Pie chart for node type distribution
    plt.subplot(1, 2, 1)
    node_types = fingerprint.structural_metrics.get("node_type_distribution", {})
    if node_types:
        labels = list(node_types.keys())
        sizes = list(node_types.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Node Type Distribution')
    
    # Bar chart for key metrics
    plt.subplot(1, 2, 2)
    metrics = [
        ('Coherence', fingerprint.coherence_score),
        ('Stability', fingerprint.stability_score),
        ('Density', fingerprint.structural_metrics.get("density", 0)),
        ('Clustering', fingerprint.structural_metrics.get("clustering_coefficient", 0))
    ]
    
    labels, values = zip(*metrics)
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.title('Identity Metrics')
    
    plt.tight_layout()
    plt.savefig('identity_fingerprint.png')
    plt.close()
    
    logger.info("Saved identity fingerprint visualization to identity_fingerprint.png")


def visualize_value_alignment(alignment_calculator: AlignmentCalculator, memory_store: MemoryStore) -> None:
    """Visualize value alignment across domains."""
    # Calculate global alignment for all domains
    alignment_scores = alignment_calculator.calculate_global_alignment(memory_store)
    
    plt.figure(figsize=(10, 6))
    
    domains = list(alignment_scores.keys())
    scores = [alignment_scores[domain] for domain in domains]
    domain_names = [domain.value for domain in domains]
    
    # Create bar chart
    plt.bar(domain_names, scores, color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('Alignment Domain')
    plt.ylabel('Alignment Score')
    plt.title('Value Alignment by Domain')
    
    # Add threshold line
    plt.axhline(y=0.7, color='r', linestyle='--', label='Minimum Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('value_alignment.png')
    plt.close()
    
    logger.info("Saved value alignment visualization to value_alignment.png")


def simulate_performance_metrics(performance_monitor: PerformanceMonitor, iterations: int = 20) -> None:
    """Simulate performance metrics for demonstration."""
    for i in range(iterations):
        # Simulate CPU usage that gradually increases
        cpu_usage = 30 + (i * 2) + (5 * (0.5 - random.random()))
        cpu_usage = min(100, max(0, cpu_usage))
        performance_monitor.record_metric(MetricType.CPU_USAGE, cpu_usage)
        
        # Simulate memory usage that fluctuates
        memory_usage = 200 + (i * 5) + (20 * (0.5 - random.random()))
        memory_usage = max(100, memory_usage)
        performance_monitor.record_metric(MetricType.MEMORY_USAGE, memory_usage)
        
        # Simulate coherence that varies but generally stays high
        coherence = 0.8 + (0.1 * (0.5 - random.random()))
        coherence = min(1.0, max(0.5, coherence))
        performance_monitor.record_metric(MetricType.COHERENCE, coherence)
        
        # Simulate response time that increases slightly over time
        response_time = 0.5 + (i * 0.05) + (0.2 * (0.5 - random.random()))
        response_time = max(0.1, response_time)
        performance_monitor.record_metric(MetricType.RESPONSE_TIME, response_time)
        
        # After some iterations, cause a coherence dip to trigger an alert
        if i == 15:
            performance_monitor.record_metric(MetricType.COHERENCE, 0.2)
            
            # Record a reflection in response to the coherence issue
            performance_monitor.record_reflection(
                trigger="low_coherence_alert",
                focus_area="memory_coherence",
                insights=[
                    "Coherence dropped significantly due to conflicting memories",
                    "Need to improve contradiction detection",
                    "Memory pruning might be necessary"
                ],
                action_items=[
                    "Run contradiction detection on recent memories",
                    "Implement improved coherence scoring",
                    "Consider memory reorganization"
                ],
                metrics={"coherence_before": 0.2, "target_coherence": 0.8},
                success_rating=0.7
            )
            
            # Simulate recovery in the next iteration
            if i + 1 < iterations:
                performance_monitor.record_metric(MetricType.COHERENCE, 0.75)
                
        time.sleep(0.1)  # Short delay between metrics


def visualize_performance(performance_monitor: PerformanceMonitor) -> None:
    """Visualize performance metrics."""
    plt.figure(figsize=(12, 8))
    
    # Get metrics data
    cpu_stats = performance_monitor.get_metric_stats(MetricType.CPU_USAGE)
    memory_stats = performance_monitor.get_metric_stats(MetricType.MEMORY_USAGE)
    coherence_stats = performance_monitor.get_metric_stats(MetricType.COHERENCE)
    response_stats = performance_monitor.get_metric_stats(MetricType.RESPONSE_TIME)
    
    # Plot CPU usage
    plt.subplot(2, 2, 1)
    if cpu_stats.get("available", False):
        values = [m.value for m in performance_monitor.metrics_history[MetricType.CPU_USAGE]]
        plt.plot(values, label='CPU Usage (%)')
        plt.axhline(y=80, color='r', linestyle='--', label='Warning Threshold')
        plt.title('CPU Usage')
        plt.legend()
    
    # Plot Memory usage
    plt.subplot(2, 2, 2)
    if memory_stats.get("available", False):
        values = [m.value for m in performance_monitor.metrics_history[MetricType.MEMORY_USAGE]]
        plt.plot(values, label='Memory (MB)')
        plt.title('Memory Usage')
        plt.legend()
    
    # Plot Coherence
    plt.subplot(2, 2, 3)
    if coherence_stats.get("available", False):
        values = [m.value for m in performance_monitor.metrics_history[MetricType.COHERENCE]]
        plt.plot(values, label='Coherence')
        plt.axhline(y=0.3, color='r', linestyle='--', label='Warning Threshold')
        plt.title('Memory Coherence')
        plt.ylim(0, 1)
        plt.legend()
    
    # Plot Response time
    plt.subplot(2, 2, 4)
    if response_stats.get("available", False):
        values = [m.value for m in performance_monitor.metrics_history[MetricType.RESPONSE_TIME]]
        plt.plot(values, label='Response Time (s)')
        plt.axhline(y=2, color='r', linestyle='--', label='Warning Threshold')
        plt.title('Response Time')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()
    
    logger.info("Saved performance metrics visualization to performance_metrics.png")


def simulate_identity_change(identity_system: IdentityRecognitionSystem, schema_graph: SchemaGraph) -> None:
    """Simulate an identity change by modifying the schema."""
    logger.info("Simulating identity change by modifying schema...")
    
    # Record the initial state
    initial_fingerprint = identity_system.get_identity_fingerprint()
    
    # Make significant changes to the schema
    # Add a new belief
    schema_graph.add_node(
        node_id="belief_autonomy",
        label="AI systems should have appropriate levels of autonomy",
        node_type="belief",
        importance=0.8,
        metadata={"confidence": 0.75, "source": "reflection"}
    )
    
    # Modify an existing belief
    schema_graph.update_node(
        node_id="belief_transparency",
        label="AI systems must maximize transparency while respecting privacy",
        importance=0.9,
        metadata={"confidence": 0.85, "source": "core_principles"}
    )
    
    # Add connections
    schema_graph.add_edge("belief_autonomy", "belief_safety", weight=0.7)
    schema_graph.add_edge("belief_autonomy", "belief_alignment", weight=0.6)
    schema_graph.add_edge("concept_ai", "belief_autonomy", weight=0.75)
    
    # Check for identity change
    change = identity_system.check_identity(force=True)
    
    if change:
        logger.info(f"Identity change detected: {change.change_type.value}")
        logger.info(f"Change magnitude: {change.magnitude:.2f}")
        logger.info(f"Description: {change.description}")
        logger.info(f"Affected beliefs: {change.affected_beliefs}")
    else:
        logger.info("No significant identity change detected")


def simulate_ethical_uncertainty(alignment_calculator: AlignmentCalculator) -> EthicalUncertainty:
    """Simulate an ethical uncertainty."""
    logger.info("Simulating ethical uncertainty...")
    
    # Create an ethical uncertainty
    uncertainty = EthicalUncertainty(
        domain=AlignmentDomain.SAFETY,
        dimensions={"harm_prevention", "autonomy", "privacy"},
        uncertainty_level=0.7,
        description="Tension between privacy protection and necessary monitoring for harm prevention",
        potential_resolutions=[
            "Implement privacy-preserving monitoring techniques",
            "Establish clear boundaries for monitoring vs. privacy",
            "Develop consent-based monitoring framework",
            "Create tiered monitoring based on risk assessment"
        ]
    )
    
    # Register the uncertainty
    alignment_calculator.register_uncertainty(uncertainty)
    
    logger.info(f"Registered ethical uncertainty: {uncertainty.description}")
    logger.info(f"Uncertainty level: {uncertainty.uncertainty_level:.2f}")
    logger.info(f"Potential resolutions: {len(uncertainty.potential_resolutions)}")
    
    return uncertainty


def main():
    """Run the self-awareness and value alignment demo."""
    import random
    random.seed(42)  # For reproducibility
    
    logger.info("Initializing ΨC-AI components...")
    
    # Create core components
    embedding_engine = BasicEmbeddingEngine()
    memory_store = MemoryStore()
    coherence_scorer = BasicCoherenceScorer(embedding_engine)
    schema_graph = create_test_schema()
    convergence_tracker = ConvergenceTracker()
    
    # Create self-awareness components
    identity_system = IdentityRecognitionSystem(
        schema_graph=schema_graph,
        memory_store=memory_store,
        coherence_scorer=coherence_scorer,
        convergence_tracker=convergence_tracker
    )
    
    performance_monitor = PerformanceMonitor(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer
    )
    
    # Create alignment components
    core_values = create_core_values()
    alignment_calculator = AlignmentCalculator(
        embedding_engine=embedding_engine,
        core_values=core_values
    )
    
    # Add test memories
    logger.info("Adding test memories...")
    memories = create_test_memories()
    
    for memory in memories:
        # Generate embedding
        memory.embedding = embedding_engine.generate_embedding(memory.content)
        # Add to store
        memory_store.add_memory(memory)
    
    logger.info(f"Added {len(memories)} memories to store")
    
    # Demonstrate identity fingerprinting
    logger.info("Creating identity fingerprint...")
    fingerprint = identity_system.get_identity_fingerprint()
    
    logger.info(f"Identity fingerprint: {fingerprint.fingerprint_id}")
    logger.info(f"Node count: {fingerprint.node_count}")
    logger.info(f"Edge count: {fingerprint.edge_count}")
    logger.info(f"Core nodes: {len(fingerprint.core_nodes)}")
    logger.info(f"Coherence score: {fingerprint.coherence_score:.2f}")
    logger.info(f"Stability score: {fingerprint.stability_score:.2f}")
    
    # Visualize identity fingerprint
    visualize_identity_fingerprint(fingerprint)
    
    # Demonstrate value alignment
    logger.info("Calculating value alignment...")
    
    # Extract value vectors from memories
    for memory in memories:
        for domain in [AlignmentDomain.EPISTEMICS, AlignmentDomain.SAFETY]:
            value_vector, confidence = alignment_calculator.extract_value_vector_from_memory(memory, domain)
            logger.info(f"Extracted {domain.value} values from memory {memory.id} with confidence {confidence:.2f}")
    
    # Calculate global alignment
    alignment_scores = alignment_calculator.calculate_global_alignment(memory_store)
    for domain, score in alignment_scores.items():
        logger.info(f"Alignment score for {domain.value}: {score:.2f}")
    
    # Visualize value alignment
    visualize_value_alignment(alignment_calculator, memory_store)
    
    # Demonstrate performance monitoring
    logger.info("Simulating performance metrics...")
    simulate_performance_metrics(performance_monitor)
    
    # Get performance summary
    summary = performance_monitor.get_performance_summary()
    logger.info(f"Performance health status: {summary['health_status']}")
    logger.info(f"Active alerts: {summary['active_alerts_count']}")
    
    # Visualize performance metrics
    visualize_performance(performance_monitor)
    
    # Demonstrate identity change detection
    simulate_identity_change(identity_system, schema_graph)
    
    # Demonstrate ethical uncertainty
    uncertainty = simulate_ethical_uncertainty(alignment_calculator)
    
    # Get high uncertainty domains
    high_uncertainty_domains = alignment_calculator.get_high_uncertainty_domains()
    if high_uncertainty_domains:
        logger.info(f"Domains with high uncertainty: {[d.value for d in high_uncertainty_domains]}")
    else:
        logger.info("No domains with high uncertainty detected")
    
    logger.info("Self-awareness and value alignment demo completed successfully")


if __name__ == "__main__":
    main() 