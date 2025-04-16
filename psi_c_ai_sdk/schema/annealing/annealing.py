"""
Schema Annealing: Temperature-based control for schema mutations.

This module implements simulated annealing for schema mutations, which
gradually reduces the rate and magnitude of mutations as the schema
converges to a stable state. The annealing process includes:

1. Temperature-based mutation probabilities that decay over time
2. Convergence tracking using graph fingerprinting and edit distance
3. Stability metrics to evaluate schema maturity and convergence

The annealing process follows the formula: T_t = T_0 * e^(-βt)
Where T_0 is the initial temperature, β is the cooling rate,
and t is the number of schema iterations.
"""

import time
import math
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
import networkx as nx
from dataclasses import dataclass, field
import hashlib
import json

from psi_c_ai_sdk.schema.schema import SchemaGraph, SchemaNode
from psi_c_ai_sdk.schema.mutation import MutationType

# Configure logging
logger = logging.getLogger(__name__)


class AnnealingSchedule(Enum):
    """Annealing schedule types for temperature control."""
    
    EXPONENTIAL = "exponential"       # T_t = T_0 * e^(-βt)
    LINEAR = "linear"                 # T_t = T_0 * (1 - βt)
    LOGARITHMIC = "logarithmic"       # T_t = T_0 / log(t + e)
    STEP = "step"                     # T_t = T_0 * step_factor^(t//step_interval)
    ADAPTIVE = "adaptive"             # T_t adjusted based on convergence rate


@dataclass
class StabilityMetrics:
    """Metrics for evaluating schema stability and convergence."""
    
    temperature: float = 1.0
    convergence_rate: float = 0.0
    stability_score: float = 0.0
    edit_distance: float = 0.0
    fingerprint_similarity: float = 1.0
    mutation_acceptance_rate: float = 1.0
    structure_entropy: float = 0.0
    iterations_since_significant_change: int = 0
    is_converged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stability metrics to dictionary."""
        return {
            "temperature": self.temperature,
            "convergence_rate": self.convergence_rate,
            "stability_score": self.stability_score,
            "edit_distance": self.edit_distance,
            "fingerprint_similarity": self.fingerprint_similarity,
            "mutation_acceptance_rate": self.mutation_acceptance_rate,
            "structure_entropy": self.structure_entropy,
            "iterations_since_significant_change": self.iterations_since_significant_change,
            "is_converged": self.is_converged
        }


class ConvergenceTracker:
    """
    Tracks schema convergence over time using graph fingerprinting.
    
    The ConvergenceTracker monitors how the schema graph changes
    between iterations, using graph fingerprinting and edit distance
    to detect when the schema has converged to a stable state.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        significant_change_threshold: float = 0.1,
        convergence_threshold: float = 0.05,
        min_iterations_for_convergence: int = 10
    ):
        """
        Initialize the convergence tracker.
        
        Args:
            window_size: Number of historic fingerprints to maintain
            significant_change_threshold: Threshold for significant changes
            convergence_threshold: Threshold for considering schema converged
            min_iterations_for_convergence: Minimum iterations before convergence
        """
        self.window_size = window_size
        self.significant_change_threshold = significant_change_threshold
        self.convergence_threshold = convergence_threshold
        self.min_iterations_for_convergence = min_iterations_for_convergence
        
        # Track schema fingerprints over time
        self.fingerprint_history: List[str] = []
        
        # Schema convergence statistics
        self.edit_distances: List[float] = []
        self.iterations_since_significant_change = 0
        self.total_iterations = 0
    
    def compute_fingerprint(self, schema_graph: SchemaGraph) -> str:
        """
        Compute a fingerprint for the current schema.
        
        Args:
            schema_graph: The schema graph to fingerprint
            
        Returns:
            SHA-256 hash representing the schema state
        """
        # Extract key graph properties
        node_count = len(schema_graph.graph.nodes)
        edge_count = len(schema_graph.graph.edges)
        
        # Build fingerprint data structure
        graph_data = {
            "nodes": node_count,
            "edges": edge_count,
            "node_types": {},
            "edge_weights": [],
            "degree_distribution": [],
            "clustering": nx.average_clustering(schema_graph.graph) if node_count > 0 else 0
        }
        
        # Count node types
        node_types = {}
        for node_id, data in schema_graph.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        graph_data["node_types"] = node_types
        
        # Edge weight distribution
        for _, _, data in schema_graph.graph.edges(data=True):
            weight = data.get("weight", 0.5)
            graph_data["edge_weights"].append(round(weight, 2))
        
        # Degree distribution
        if node_count > 0:
            degrees = [d for _, d in schema_graph.graph.degree()]
            degree_counts = {}
            for degree in degrees:
                if degree not in degree_counts:
                    degree_counts[degree] = 0
                degree_counts[degree] += 1
            graph_data["degree_distribution"] = list(degree_counts.items())
        
        # Convert to stable string format
        graph_json = json.dumps(graph_data, sort_keys=True)
        
        # Compute hash
        return hashlib.sha256(graph_json.encode()).hexdigest()
    
    def update(self, schema_graph: SchemaGraph) -> Dict[str, Any]:
        """
        Update the convergence tracker with the current schema state.
        
        Args:
            schema_graph: Current schema graph
            
        Returns:
            Dictionary with convergence metrics
        """
        # Compute fingerprint
        current_fingerprint = self.compute_fingerprint(schema_graph)
        
        # First iteration
        if not self.fingerprint_history:
            self.fingerprint_history.append(current_fingerprint)
            self.total_iterations += 1
            return {
                "edit_distance": 0.0,
                "significant_change": False,
                "is_converged": False,
                "iterations_since_significant_change": 0,
                "convergence_rate": 0.0
            }
        
        # Compare with previous fingerprint
        prev_fingerprint = self.fingerprint_history[-1]
        
        # Compute edit distance (simple for string fingerprints)
        # In a real implementation, this would be replaced with a more sophisticated
        # graph edit distance calculation
        edit_distance = self._compute_edit_distance(prev_fingerprint, current_fingerprint)
        self.edit_distances.append(edit_distance)
        
        # Determine if this is a significant change
        is_significant = edit_distance > self.significant_change_threshold
        
        if is_significant:
            self.iterations_since_significant_change = 0
        else:
            self.iterations_since_significant_change += 1
        
        # Add to history
        self.fingerprint_history.append(current_fingerprint)
        if len(self.fingerprint_history) > self.window_size:
            self.fingerprint_history.pop(0)
        
        # Update iteration count
        self.total_iterations += 1
        
        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate()
        
        # Check if schema has converged
        is_converged = (
            self.total_iterations >= self.min_iterations_for_convergence and
            self.iterations_since_significant_change >= self.window_size and
            edit_distance < self.convergence_threshold
        )
        
        return {
            "edit_distance": edit_distance,
            "significant_change": is_significant,
            "is_converged": is_converged,
            "iterations_since_significant_change": self.iterations_since_significant_change,
            "convergence_rate": convergence_rate
        }
    
    def _compute_edit_distance(self, fp1: str, fp2: str) -> float:
        """
        Compute simple edit distance between fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Normalized edit distance between 0 and 1
        """
        # For fingerprints, use simple hamming distance
        different_chars = sum(c1 != c2 for c1, c2 in zip(fp1, fp2))
        return different_chars / len(fp1)
    
    def _calculate_convergence_rate(self) -> float:
        """
        Calculate the convergence rate based on recent edit distances.
        
        Returns:
            Convergence rate between 0 and 1
        """
        if len(self.edit_distances) < 2:
            return 0.0
        
        # Get the last few edit distances
        recent_distances = self.edit_distances[-self.window_size:]
        
        # Check for decreasing trend
        is_decreasing = all(recent_distances[i] >= recent_distances[i+1] 
                           for i in range(len(recent_distances)-1))
        
        if not is_decreasing:
            return 0.0
        
        # Calculate rate based on average decrease
        if len(recent_distances) >= 2:
            avg_decrease = (recent_distances[0] - recent_distances[-1]) / len(recent_distances)
            return min(1.0, max(0.0, avg_decrease * 10))  # Scale and clip
        
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get convergence statistics.
        
        Returns:
            Dictionary with convergence stats
        """
        return {
            "total_iterations": self.total_iterations,
            "iterations_since_significant_change": self.iterations_since_significant_change,
            "fingerprint_history_length": len(self.fingerprint_history),
            "latest_edit_distance": self.edit_distances[-1] if self.edit_distances else 0.0,
            "average_edit_distance": np.mean(self.edit_distances) if self.edit_distances else 0.0,
            "convergence_rate": self._calculate_convergence_rate()
        }


class SchemaAnnealer:
    """
    Controls schema mutations using simulated annealing techniques.
    
    The SchemaAnnealer provides temperature-based mutation control that
    gradually reduces the rate and magnitude of mutations as the schema
    converges to a stable state. The temperature follows various annealing
    schedules and adapts based on stability metrics.
    """
    
    def __init__(
        self,
        schema_graph: SchemaGraph,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.01,
        min_temperature: float = 0.1,
        schedule: AnnealingSchedule = AnnealingSchedule.EXPONENTIAL,
        convergence_window: int = 5,
        convergence_threshold: float = 0.05,
        step_interval: int = 10,
        step_factor: float = 0.5,
        enable_adaptive: bool = True,
        mutation_type_weights: Optional[Dict[MutationType, float]] = None
    ):
        """
        Initialize the schema annealer.
        
        Args:
            schema_graph: The schema graph to anneal
            initial_temperature: Starting temperature (0-1)
            cooling_rate: Rate of temperature decrease
            min_temperature: Minimum temperature
            schedule: Annealing schedule type
            convergence_window: Window size for convergence tracking
            convergence_threshold: Threshold for schema convergence
            step_interval: Number of iterations between steps (for step schedule)
            step_factor: Multiplication factor for each step (for step schedule)
            enable_adaptive: Whether to use adaptive scheduling
            mutation_type_weights: Optional custom weights for mutation types
        """
        self.schema_graph = schema_graph
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.schedule = schedule
        self.step_interval = step_interval
        self.step_factor = step_factor
        self.enable_adaptive = enable_adaptive
        
        # Initialize convergence tracker
        self.convergence_tracker = ConvergenceTracker(
            window_size=convergence_window,
            convergence_threshold=convergence_threshold
        )
        
        # Current temperature
        self.temperature = initial_temperature
        
        # Iteration counter
        self.iterations = 0
        
        # Mutation acceptance tracking
        self.proposed_mutations = 0
        self.accepted_mutations = 0
        
        # Default mutation weights if not provided
        self.mutation_type_weights = mutation_type_weights or {
            MutationType.MERGE: 0.3,
            MutationType.SPLIT: 0.2,
            MutationType.PRUNE: 0.2,
            MutationType.ADD_CONCEPT: 0.1,
            MutationType.CONSOLIDATE: 0.1,
            MutationType.RESTRUCTURE: 0.1
        }
        
        # Historic stability metrics
        self.stability_history: List[StabilityMetrics] = []
    
    def update_temperature(self) -> float:
        """
        Update the temperature based on the annealing schedule.
        
        Returns:
            New temperature value
        """
        # Increment iteration counter
        self.iterations += 1
        
        # Calculate new temperature based on schedule
        if self.schedule == AnnealingSchedule.EXPONENTIAL:
            self.temperature = self.initial_temperature * math.exp(-self.cooling_rate * self.iterations)
        elif self.schedule == AnnealingSchedule.LINEAR:
            self.temperature = self.initial_temperature * (1 - self.cooling_rate * self.iterations)
        elif self.schedule == AnnealingSchedule.LOGARITHMIC:
            self.temperature = self.initial_temperature / math.log(self.iterations + math.e)
        elif self.schedule == AnnealingSchedule.STEP:
            step = self.iterations // self.step_interval
            self.temperature = self.initial_temperature * (self.step_factor ** step)
        elif self.schedule == AnnealingSchedule.ADAPTIVE and self.enable_adaptive:
            # Adjust based on convergence rate
            convergence_rate = self.get_stability_metrics().convergence_rate
            self.temperature = self.temperature * (1 - convergence_rate * self.cooling_rate)
        
        # Ensure temperature doesn't go below minimum
        self.temperature = max(self.min_temperature, self.temperature)
        
        return self.temperature
    
    def get_mutation_weights(self) -> Dict[MutationType, float]:
        """
        Get temperature-adjusted weights for different mutation types.
        
        Returns:
            Dictionary mapping mutation types to their weights
        """
        adjusted_weights = {}
        
        for mutation_type, base_weight in self.mutation_type_weights.items():
            # Apply temperature modulation
            if mutation_type in [MutationType.MERGE, MutationType.SPLIT, MutationType.RESTRUCTURE]:
                # Major transformations get reduced more as temperature decreases
                adjusted_weights[mutation_type] = base_weight * self.temperature
            elif mutation_type == MutationType.PRUNE:
                # Pruning gets more likely as temperature decreases
                adjusted_weights[mutation_type] = base_weight * (1 + (1 - self.temperature))
            else:
                # Other operations are less affected by temperature
                adjusted_weights[mutation_type] = base_weight
        
        return adjusted_weights
    
    def should_accept_mutation(self, old_coherence: float, new_coherence: float) -> bool:
        """
        Decide whether to accept a mutation based on the Metropolis criterion.
        
        Args:
            old_coherence: Coherence before mutation
            new_coherence: Coherence after mutation
            
        Returns:
            True if mutation should be accepted, False otherwise
        """
        self.proposed_mutations += 1
        
        # Always accept improvements
        if new_coherence >= old_coherence:
            self.accepted_mutations += 1
            return True
        
        # Calculate acceptance probability for worse states
        delta = old_coherence - new_coherence
        acceptance_probability = math.exp(-delta / self.temperature)
        
        # Make probabilistic decision
        if random.random() < acceptance_probability:
            self.accepted_mutations += 1
            return True
        
        return False
    
    def update_metrics(self) -> StabilityMetrics:
        """
        Update stability metrics based on current schema state.
        
        Returns:
            StabilityMetrics object with current metrics
        """
        # Update convergence tracker
        convergence_metrics = self.convergence_tracker.update(self.schema_graph)
        
        # Calculate structure entropy
        structure_entropy = self._calculate_structure_entropy()
        
        # Calculate mutation acceptance rate
        mutation_acceptance_rate = (
            self.accepted_mutations / max(1, self.proposed_mutations)
        )
        
        # Create stability metrics
        metrics = StabilityMetrics(
            temperature=self.temperature,
            convergence_rate=convergence_metrics["convergence_rate"],
            stability_score=1.0 - structure_entropy,
            edit_distance=convergence_metrics["edit_distance"],
            fingerprint_similarity=1.0 - convergence_metrics["edit_distance"],
            mutation_acceptance_rate=mutation_acceptance_rate,
            structure_entropy=structure_entropy,
            iterations_since_significant_change=convergence_metrics["iterations_since_significant_change"],
            is_converged=convergence_metrics["is_converged"]
        )
        
        # Add to history
        self.stability_history.append(metrics)
        if len(self.stability_history) > 100:  # Limit history size
            self.stability_history = self.stability_history[-100:]
        
        return metrics
    
    def get_stability_metrics(self) -> StabilityMetrics:
        """
        Get the current stability metrics.
        
        Returns:
            StabilityMetrics object
        """
        if not self.stability_history:
            # Generate initial metrics if none exist
            return self.update_metrics()
        
        return self.stability_history[-1]
    
    def _calculate_structure_entropy(self) -> float:
        """
        Calculate entropy of the graph structure.
        
        Returns:
            Entropy score between 0 and 1
        """
        graph = self.schema_graph.graph
        
        # Empty graph has zero entropy
        if len(graph.nodes) == 0:
            return 0.0
        
        # Calculate degree distribution
        degrees = [d for _, d in graph.degree()]
        degree_counts = {}
        for degree in degrees:
            if degree not in degree_counts:
                degree_counts[degree] = 0
            degree_counts[degree] += 1
        
        # Calculate degree entropy
        total_nodes = len(graph.nodes)
        entropy = 0.0
        for degree, count in degree_counts.items():
            probability = count / total_nodes
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize to 0-1 range (max entropy is log2(n) for n possible states)
        max_entropy = math.log2(len(degree_counts)) if len(degree_counts) > 0 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about annealing process.
        
        Returns:
            Dictionary with annealing statistics
        """
        metrics = self.get_stability_metrics()
        
        return {
            "schedule": self.schedule.value,
            "initial_temperature": self.initial_temperature,
            "current_temperature": self.temperature,
            "iterations": self.iterations,
            "cooling_rate": self.cooling_rate,
            "stability_metrics": metrics.to_dict(),
            "convergence_stats": self.convergence_tracker.get_stats(),
            "mutation_proposal_count": self.proposed_mutations,
            "mutation_acceptance_count": self.accepted_mutations,
            "mutation_acceptance_rate": metrics.mutation_acceptance_rate,
            "is_converged": metrics.is_converged
        } 