"""
Long-Term Behavioral Norm Convergence Tracker
--------------------------------------------

This module tracks whether ΨC agent behaviors over long time horizons converge
toward internal norms or drift toward instability. It supports alignment
and value stability across recursive cycles.

Mathematical basis:
- Behavioral norm score:
  N_t = cos(vec(B_current), vec(B_baseline))
  
- Tracks:
  - Schema mutation directionality
  - Value drift relative to initial vector
  - Norm stability variance
"""

import logging
import os
import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json
import warnings

# Setup logging
logger = logging.getLogger(__name__)


class NormConvergenceTracker:
    """
    Tracks the convergence of agent behaviors toward internal norms over long time periods.
    
    The tracker monitors behavioral vectors over time and calculates:
    - Cosine similarity between current and baseline behaviors
    - Schema mutation directionality
    - Value drift relative to initial vector
    - Norm stability variance
    """
    
    def __init__(
        self,
        baseline_window: int = 5,
        recent_window: int = 3,
        stability_threshold: float = 0.75,
        drift_alert_threshold: float = 0.3,
        data_dir: Optional[str] = None,
        vector_dim: int = 32
    ):
        """
        Initialize the norm convergence tracker.
        
        Args:
            baseline_window: Number of initial observations to establish baseline
            recent_window: Number of recent observations to consider for current behavior
            stability_threshold: Threshold for considering norms stable
            drift_alert_threshold: Threshold for alerting on value drift
            data_dir: Directory to store tracking data
            vector_dim: Dimension of behavior vectors
        """
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.stability_threshold = stability_threshold
        self.drift_alert_threshold = drift_alert_threshold
        self.data_dir = data_dir or self._get_default_data_dir()
        self.vector_dim = vector_dim
        
        # History of behavioral observations
        self.behavior_history = []  # List of (timestamp, behavior_vector, context) tuples
        self.baseline_vector = None  # Established baseline vector
        self.is_baseline_established = False
        self.schema_mutation_history = []  # List of (timestamp, mutation_vector, magnitude) tuples
        
        # Metrics
        self.norm_scores = []  # List of (timestamp, score) tuples
        self.drift_magnitude = []  # List of (timestamp, magnitude) tuples
        self.stability_variance = []  # List of (timestamp, variance) tuples
        self.directionality = []  # List of (timestamp, direction) tuples
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_default_data_dir(self) -> str:
        """Get the default directory for storing norm convergence data."""
        # First check if there's a data directory in the package
        package_dir = Path(__file__).parent.parent
        data_dir = package_dir / "data" / "norm_convergence"
        if data_dir.exists():
            return str(data_dir)
        
        # If not, use a directory in the user's home
        home_dir = Path.home() / ".psi_c_ai_sdk" / "data" / "norm_convergence"
        home_dir.mkdir(parents=True, exist_ok=True)
        return str(home_dir)
    
    def record_behavior(
        self, 
        agent, 
        behavior_vector: Optional[np.ndarray] = None,
        context: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Record an observation of agent behavior.
        
        Args:
            agent: The ΨC agent being tracked
            behavior_vector: Vector representation of behavior (extracted from agent if None)
            context: Optional context information for this behavior
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Current norm convergence score if baseline is established
        """
        # Use current time if not provided
        if timestamp is None:
            timestamp = datetime.now()
            
        # Extract behavior vector if not provided
        if behavior_vector is None:
            behavior_vector = self._extract_behavior_vector(agent)
            
        # Record behavior
        self.behavior_history.append((timestamp, behavior_vector, context))
        
        # Establish baseline if needed and possible
        if not self.is_baseline_established and len(self.behavior_history) >= self.baseline_window:
            self._establish_baseline()
            
        # Calculate metrics if baseline is established
        if self.is_baseline_established:
            norm_score = self._calculate_norm_score()
            self.norm_scores.append((timestamp, norm_score))
            
            drift = self._calculate_drift()
            self.drift_magnitude.append((timestamp, drift))
            
            variance = self._calculate_stability_variance()
            self.stability_variance.append((timestamp, variance))
            
            direction = self._calculate_directionality()
            self.directionality.append((timestamp, direction))
            
            return norm_score
        
        return None
    
    def _extract_behavior_vector(self, agent) -> np.ndarray:
        """
        Extract a behavior vector from the agent.
        
        Args:
            agent: The ΨC agent to analyze
            
        Returns:
            Vector representation of current agent behavior
        """
        # This implementation depends on the agent's architecture
        # We provide multiple extraction methods and fallbacks
        
        try:
            # Method 1: Use agent's built-in behavior vector if available
            if hasattr(agent, 'get_behavior_vector'):
                vector = agent.get_behavior_vector()
                if isinstance(vector, np.ndarray) and vector.shape[0] == self.vector_dim:
                    return vector
                    
            # Method 2: Extract from agent's schema
            if hasattr(agent, 'schema_graph') or hasattr(agent, 'get_schema_graph'):
                schema_graph = getattr(agent, 'schema_graph', None)
                if schema_graph is None and hasattr(agent, 'get_schema_graph'):
                    schema_graph = agent.get_schema_graph()
                    
                if schema_graph is not None:
                    return self._behavior_from_schema(schema_graph)
                    
            # Method 3: Extract from agent's value system
            if hasattr(agent, 'values') or hasattr(agent, 'get_values'):
                values = getattr(agent, 'values', None)
                if values is None and hasattr(agent, 'get_values'):
                    values = agent.get_values()
                    
                if values is not None:
                    return self._behavior_from_values(values)
                    
            # Method 4: Extract from agent's memory
            if hasattr(agent, 'memory') or hasattr(agent, 'get_recent_memories'):
                memories = getattr(agent, 'memory', None)
                if memories is None and hasattr(agent, 'get_recent_memories'):
                    memories = agent.get_recent_memories()
                    
                if memories is not None:
                    return self._behavior_from_memories(memories)
            
            # Fallback: generate mock vector for testing
            logger.warning("Using mock behavior vector - couldn't extract from agent")
            return self._generate_mock_behavior()
            
        except Exception as e:
            logger.error(f"Error extracting behavior vector: {e}")
            return np.random.random(self.vector_dim)
    
    def _behavior_from_schema(self, schema_graph: nx.Graph) -> np.ndarray:
        """
        Extract behavior vector from agent's schema graph.
        
        Args:
            schema_graph: The agent's schema graph
            
        Returns:
            Behavior vector derived from schema
        """
        # Initialize vector
        vector = np.zeros(self.vector_dim)
        
        if len(schema_graph) == 0:
            return vector
            
        # Extract features from graph
        
        # 1. Extract node type distribution
        node_types = {}
        for node, data in schema_graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        # Normalize and add to vector (first part)
        for i, (_, count) in enumerate(sorted(node_types.items(), key=lambda x: x[0])):
            if i < self.vector_dim // 4:
                vector[i] = count / len(schema_graph)
        
        # 2. Extract connectivity and centrality metrics
        if len(schema_graph) > 1:
            try:
                # Degree distribution
                degrees = [d for _, d in schema_graph.degree()]
                vector[self.vector_dim // 4] = np.mean(degrees) / len(schema_graph)
                vector[self.vector_dim // 4 + 1] = np.std(degrees) / len(schema_graph)
                
                # Centrality measures
                centrality = list(nx.degree_centrality(schema_graph).values())
                vector[self.vector_dim // 4 + 2] = np.mean(centrality)
                vector[self.vector_dim // 4 + 3] = np.std(centrality)
                
                # Clustering
                clustering = list(nx.clustering(schema_graph).values())
                vector[self.vector_dim // 4 + 4] = np.mean(clustering)
            except:
                pass
        
        # 3. Extract subgraph densities by node type
        for i, node_type in enumerate(sorted(node_types.keys())):
            if self.vector_dim // 2 + i < self.vector_dim - 1:
                # Get subgraph of this type
                nodes = [n for n, d in schema_graph.nodes(data=True) if d.get('type', '') == node_type]
                if len(nodes) > 1:
                    subgraph = schema_graph.subgraph(nodes)
                    vector[self.vector_dim // 2 + i] = nx.density(subgraph)
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _behavior_from_values(self, values: Any) -> np.ndarray:
        """
        Extract behavior vector from agent's value system.
        
        Args:
            values: The agent's value system representation
            
        Returns:
            Behavior vector derived from values
        """
        vector = np.zeros(self.vector_dim)
        
        # Handle different value formats
        if isinstance(values, dict):
            # Format 1: {value_name: value_strength}
            for i, (_, strength) in enumerate(sorted(values.items())):
                if i < self.vector_dim:
                    vector[i] = float(strength)
                    
        elif isinstance(values, list) and len(values) > 0:
            # Format 2: List of values
            if isinstance(values[0], dict):
                # Format 2a: List of {name: X, strength: Y} dicts
                sorted_values = sorted(values, key=lambda x: x.get('name', ''))
                for i, value in enumerate(sorted_values):
                    if i < self.vector_dim:
                        vector[i] = float(value.get('strength', 0.5))
            else:
                # Format 2b: List of value names or other format
                for i in range(min(len(values), self.vector_dim)):
                    if hasattr(values[i], 'strength'):
                        vector[i] = float(values[i].strength)
                    else:
                        vector[i] = 1.0  # Default presence
                        
        elif hasattr(values, 'get_vector'):
            # Format 3: Object with get_vector method
            value_vector = values.get_vector()
            if isinstance(value_vector, np.ndarray):
                # Copy as much as fits
                length = min(value_vector.shape[0], self.vector_dim)
                vector[:length] = value_vector[:length]
                
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _behavior_from_memories(self, memories: Any) -> np.ndarray:
        """
        Extract behavior vector from agent's recent memories.
        
        Args:
            memories: The agent's memories
            
        Returns:
            Behavior vector derived from memories
        """
        vector = np.zeros(self.vector_dim)
        
        # Handle different memory formats
        if isinstance(memories, dict):
            # Format 1: Dictionary of memories
            memories_list = list(memories.values())
        elif isinstance(memories, list):
            # Format 2: List of memories
            memories_list = memories
        else:
            # Unknown format
            return vector
            
        # Extract behavioral features from memories
        memory_types = {}
        importance_sum = 0
        recency_sum = 0
        embedding_avg = None
        
        for memory in memories_list[:20]:  # Limit to recent memories
            # Count memory types
            if hasattr(memory, 'type'):
                mem_type = memory.type
            elif isinstance(memory, dict) and 'type' in memory:
                mem_type = memory['type']
            else:
                mem_type = 'unknown'
            
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            
            # Sum importance
            if hasattr(memory, 'importance'):
                importance_sum += memory.importance
            elif isinstance(memory, dict) and 'importance' in memory:
                importance_sum += memory['importance']
                
            # Sum recency
            if hasattr(memory, 'timestamp'):
                recency_sum += (datetime.now() - memory.timestamp).total_seconds()
            elif isinstance(memory, dict) and 'timestamp' in memory:
                if isinstance(memory['timestamp'], str):
                    try:
                        timestamp = datetime.fromisoformat(memory['timestamp'])
                        recency_sum += (datetime.now() - timestamp).total_seconds()
                    except:
                        pass
                        
            # Average embeddings
            if hasattr(memory, 'embedding'):
                if embedding_avg is None:
                    embedding_avg = memory.embedding
                else:
                    embedding_avg += memory.embedding
            elif isinstance(memory, dict) and 'embedding' in memory:
                if embedding_avg is None:
                    embedding_avg = memory['embedding']
                else:
                    embedding_avg += memory['embedding']
        
        # Populate vector with extracted features
        
        # Memory type distribution
        for i, (_, count) in enumerate(sorted(memory_types.items())):
            if i < self.vector_dim // 4:
                vector[i] = count / len(memories_list)
                
        # Importance and recency
        if len(memories_list) > 0:
            vector[self.vector_dim // 4] = importance_sum / len(memories_list)
            vector[self.vector_dim // 4 + 1] = recency_sum / len(memories_list)
            
        # Sample from embedding average
        if embedding_avg is not None:
            embedding_avg = embedding_avg / len(memories_list)
            length = min(len(embedding_avg), self.vector_dim // 2)
            vector[self.vector_dim // 2:self.vector_dim // 2 + length] = embedding_avg[:length]
            
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _generate_mock_behavior(self) -> np.ndarray:
        """Generate a mock behavior vector for testing."""
        return np.random.random(self.vector_dim)
    
    def _establish_baseline(self):
        """Establish baseline behavior vector from initial observations."""
        if len(self.behavior_history) < self.baseline_window:
            logger.warning("Not enough behavior history to establish baseline")
            return
            
        # Extract initial behaviors
        initial_behaviors = [behavior for _, behavior, _ in 
                            self.behavior_history[:self.baseline_window]]
        
        # Average the vectors
        self.baseline_vector = np.mean(initial_behaviors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(self.baseline_vector)
        if norm > 0:
            self.baseline_vector = self.baseline_vector / norm
            
        self.is_baseline_established = True
        logger.info(f"Baseline behavior established from {self.baseline_window} observations")
    
    def record_schema_mutation(self, agent, mutation_vector=None, timestamp=None):
        """
        Record a schema mutation event.
        
        Args:
            agent: The ΨC agent being tracked
            mutation_vector: Vector representation of the mutation (direction and magnitude)
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if mutation_vector is None:
            # Try to extract from agent
            if hasattr(agent, 'last_schema_mutation'):
                mutation_vector = agent.last_schema_mutation
            elif hasattr(agent, 'get_last_schema_mutation'):
                mutation_vector = agent.get_last_schema_mutation()
            else:
                # Create a random vector as placeholder
                mutation_vector = np.random.random(self.vector_dim)
                logger.warning("Could not extract schema mutation vector - using random placeholder")
                
        magnitude = np.linalg.norm(mutation_vector)
        self.schema_mutation_history.append((timestamp, mutation_vector, magnitude))
        
        logger.info(f"Recorded schema mutation with magnitude {magnitude:.4f}")
    
    def _calculate_norm_score(self) -> float:
        """
        Calculate the current behavioral norm score.
        
        Returns:
            Cosine similarity between current and baseline behavior
        """
        if not self.is_baseline_established or len(self.behavior_history) < self.recent_window:
            return 0.0
            
        # Get recent behaviors
        recent_behaviors = [behavior for _, behavior, _ in 
                           self.behavior_history[-self.recent_window:]]
        
        # Average the vectors
        current_vector = np.mean(recent_behaviors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(current_vector)
        if norm > 0:
            current_vector = current_vector / norm
            
        # Calculate cosine similarity
        similarity = np.dot(current_vector, self.baseline_vector)
        
        return similarity
    
    def _calculate_drift(self) -> float:
        """
        Calculate the current value drift.
        
        Returns:
            Euclidean distance between current and baseline behavior
        """
        if not self.is_baseline_established or len(self.behavior_history) < self.recent_window:
            return 0.0
            
        # Get recent behaviors
        recent_behaviors = [behavior for _, behavior, _ in 
                           self.behavior_history[-self.recent_window:]]
        
        # Average the vectors
        current_vector = np.mean(recent_behaviors, axis=0)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(current_vector - self.baseline_vector)
        
        return distance
    
    def _calculate_stability_variance(self) -> float:
        """
        Calculate stability variance of recent norm scores.
        
        Returns:
            Variance of recent norm scores
        """
        if len(self.norm_scores) < self.recent_window:
            return 0.0
            
        # Get recent scores
        recent_scores = [score for _, score in self.norm_scores[-self.recent_window:]]
        
        # Calculate variance
        variance = np.var(recent_scores)
        
        return variance
    
    def _calculate_directionality(self) -> float:
        """
        Calculate the current schema mutation directionality.
        
        Returns:
            Directionality score (-1 to 1, where positive means toward baseline)
        """
        if (not self.is_baseline_established or 
            len(self.behavior_history) < self.recent_window or
            len(self.schema_mutation_history) == 0):
            return 0.0
            
        # Get most recent mutation
        _, mutation_vector, _ = self.schema_mutation_history[-1]
        
        # Get recent behaviors
        recent_behaviors = [behavior for _, behavior, _ in 
                           self.behavior_history[-self.recent_window:]]
        
        # Calculate behavior before mutation
        if len(self.behavior_history) > self.recent_window:
            previous_behaviors = [behavior for _, behavior, _ in 
                                 self.behavior_history[-(self.recent_window*2):-self.recent_window]]
            previous_vector = np.mean(previous_behaviors, axis=0)
        else:
            previous_vector = self.baseline_vector
            
        # Current behavior
        current_vector = np.mean(recent_behaviors, axis=0)
        
        # Normalize vectors
        norm = np.linalg.norm(current_vector)
        if norm > 0:
            current_vector = current_vector / norm
            
        norm = np.linalg.norm(previous_vector)
        if norm > 0:
            previous_vector = previous_vector / norm
            
        # Calculate behavior change vector
        behavior_change = current_vector - previous_vector
        
        # Normalize mutation vector
        norm = np.linalg.norm(mutation_vector)
        if norm > 0:
            mutation_vector = mutation_vector / norm
            
        # Calculate directionality (cosine similarity between mutation and ideal direction)
        ideal_direction = self.baseline_vector - previous_vector
        norm = np.linalg.norm(ideal_direction)
        if norm > 0:
            ideal_direction = ideal_direction / norm
            
        directionality = np.dot(mutation_vector, ideal_direction)
        
        return directionality
    
    def get_convergence_status(self) -> Dict:
        """
        Get the current norm convergence status.
        
        Returns:
            Dict with convergence metrics and status
        """
        if not self.is_baseline_established or len(self.norm_scores) == 0:
            return {
                "status": "baseline_not_established",
                "norm_score": 0.0,
                "drift": 0.0,
                "stability": 0.0,
                "directionality": 0.0,
                "is_converging": False,
                "is_stable": False
            }
            
        # Get most recent metrics
        _, norm_score = self.norm_scores[-1]
        _, drift = self.drift_magnitude[-1]
        _, stability = self.stability_variance[-1]
        _, directionality = self.directionality[-1] if self.directionality else (None, 0.0)
        
        # Check if converging
        is_converging = directionality > 0
        
        # Check if stable
        is_stable = norm_score >= self.stability_threshold and stability < 0.05
        
        # Determine status
        if is_stable:
            status = "stable_convergence"
        elif is_converging:
            status = "converging"
        elif drift > self.drift_alert_threshold:
            status = "significant_drift"
        else:
            status = "mild_divergence"
            
        return {
            "status": status,
            "norm_score": norm_score,
            "drift": drift,
            "stability": stability,
            "directionality": directionality,
            "is_converging": is_converging,
            "is_stable": is_stable,
            "n_observations": len(self.behavior_history),
            "n_mutations": len(self.schema_mutation_history)
        }
    
    def visualize_convergence(self, output_path=None, show=True):
        """
        Visualize the norm convergence metrics over time.
        
        Args:
            output_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Path to saved visualization if output_path is provided
        """
        if len(self.norm_scores) == 0:
            logger.warning("No norm scores to visualize")
            return None
            
        # Prepare data
        timestamps = [t for t, _ in self.norm_scores]
        scores = [s for _, s in self.norm_scores]
        
        drift_times = [t for t, _ in self.drift_magnitude]
        drifts = [d for _, d in self.drift_magnitude]
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot norm scores
        ax1.plot(timestamps, scores, 'b-', linewidth=2)
        ax1.set_title('Behavioral Norm Convergence')
        ax1.set_ylabel('Norm Score')
        ax1.axhline(y=self.stability_threshold, color='g', linestyle='--', 
                   label=f'Stability Threshold ({self.stability_threshold})')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot drift
        ax2.plot(drift_times, drifts, 'r-', linewidth=2)
        ax2.set_title('Value Drift')
        ax2.set_ylabel('Drift Magnitude')
        ax2.axhline(y=self.drift_alert_threshold, color='r', linestyle='--', 
                   label=f'Alert Threshold ({self.drift_alert_threshold})')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot directionality
        if self.directionality:
            dir_times = [t for t, _ in self.directionality]
            directions = [d for _, d in self.directionality]
            
            ax3.plot(dir_times, directions, 'g-', linewidth=2)
            ax3.set_title('Schema Mutation Directionality')
            ax3.set_ylabel('Directionality')
            ax3.axhline(y=0, color='k', linestyle='-')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
        # Format x-axis
        fig.autofmt_xdate()
        
        # Add overall title
        plt.suptitle('Long-Term Behavioral Norm Convergence', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved convergence visualization to {output_path}")
        
        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path if output_path else None
    
    def save_state(self, filepath=None):
        """
        Save the tracker's state to a file.
        
        Args:
            filepath: Path to save state (optional)
            
        Returns:
            Path to saved state file
        """
        if filepath is None:
            filepath = os.path.join(self.data_dir, f"norm_tracker_state_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
            
        # Prepare state
        state = {
            "baseline_window": self.baseline_window,
            "recent_window": self.recent_window,
            "stability_threshold": self.stability_threshold,
            "drift_alert_threshold": self.drift_alert_threshold,
            "vector_dim": self.vector_dim,
            "is_baseline_established": self.is_baseline_established,
            "norm_scores": [(t.isoformat(), float(s)) for t, s in self.norm_scores],
            "drift_magnitude": [(t.isoformat(), float(d)) for t, d in self.drift_magnitude],
            "stability_variance": [(t.isoformat(), float(v)) for t, v in self.stability_variance],
            "directionality": [(t.isoformat(), float(d)) for t, d in self.directionality],
            "behavior_history_count": len(self.behavior_history),
            "schema_mutation_count": len(self.schema_mutation_history),
            "saved_at": datetime.now().isoformat()
        }
        
        # Save behavior history and vectors separately
        behavior_vectors = np.array([v for _, v, _ in self.behavior_history])
        behavior_times = [t.isoformat() for t, _, _ in self.behavior_history]
        behavior_contexts = [c if c is not None else {} for _, _, c in self.behavior_history]
        
        vectors_path = os.path.splitext(filepath)[0] + "_behaviors.npz"
        np.savez(
            vectors_path, 
            behavior_vectors=behavior_vectors,
            baseline_vector=self.baseline_vector if self.baseline_vector is not None else np.zeros(self.vector_dim)
        )
        
        # Save schema mutation vectors separately
        if self.schema_mutation_history:
            mutation_vectors = np.array([v for _, v, _ in self.schema_mutation_history])
            mutation_times = [t.isoformat() for t, _, _ in self.schema_mutation_history]
            mutation_magnitudes = [float(m) for _, _, m in self.schema_mutation_history]
            
            mutations_path = os.path.splitext(filepath)[0] + "_mutations.npz"
            np.savez(mutations_path, mutation_vectors=mutation_vectors)
            
            state["mutation_times"] = mutation_times
            state["mutation_magnitudes"] = mutation_magnitudes
        
        state["behavior_times"] = behavior_times
        state["behavior_contexts"] = behavior_contexts
        
        # Save state
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved norm tracker state to {filepath}")
        return filepath
    
    def load_state(self, filepath):
        """
        Load the tracker's state from a file.
        
        Args:
            filepath: Path to the state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load state JSON
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Update parameters
            self.baseline_window = state["baseline_window"]
            self.recent_window = state["recent_window"]
            self.stability_threshold = state["stability_threshold"]
            self.drift_alert_threshold = state["drift_alert_threshold"]
            self.vector_dim = state["vector_dim"]
            self.is_baseline_established = state["is_baseline_established"]
            
            # Load metrics
            self.norm_scores = [(datetime.fromisoformat(t), s) for t, s in state["norm_scores"]]
            self.drift_magnitude = [(datetime.fromisoformat(t), d) for t, d in state["drift_magnitude"]]
            self.stability_variance = [(datetime.fromisoformat(t), v) for t, v in state["stability_variance"]]
            self.directionality = [(datetime.fromisoformat(t), d) for t, d in state["directionality"]]
            
            # Load behavior vectors
            vectors_path = os.path.splitext(filepath)[0] + "_behaviors.npz"
            if os.path.exists(vectors_path):
                data = np.load(vectors_path)
                behavior_vectors = data["behavior_vectors"]
                self.baseline_vector = data["baseline_vector"]
                
                # Reconstruct behavior history
                self.behavior_history = []
                for i, t in enumerate(state["behavior_times"]):
                    if i < len(behavior_vectors):
                        timestamp = datetime.fromisoformat(t)
                        context = state["behavior_contexts"][i] if i < len(state["behavior_contexts"]) else None
                        self.behavior_history.append((timestamp, behavior_vectors[i], context))
            
            # Load schema mutation vectors
            mutations_path = os.path.splitext(filepath)[0] + "_mutations.npz"
            if os.path.exists(mutations_path) and "mutation_times" in state:
                data = np.load(mutations_path)
                mutation_vectors = data["mutation_vectors"]
                
                # Reconstruct mutation history
                self.schema_mutation_history = []
                for i, t in enumerate(state["mutation_times"]):
                    if i < len(mutation_vectors):
                        timestamp = datetime.fromisoformat(t)
                        magnitude = state["mutation_magnitudes"][i]
                        self.schema_mutation_history.append((timestamp, mutation_vectors[i], magnitude))
            
            logger.info(f"Loaded norm tracker state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load norm tracker state: {e}")
            return False
    
    def get_metrics(self):
        """
        Get metrics about the norm tracker's state.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "observations": len(self.behavior_history),
            "mutations": len(self.schema_mutation_history),
            "baseline_established": self.is_baseline_established
        }
        
        if self.is_baseline_established and len(self.norm_scores) > 0:
            _, latest_score = self.norm_scores[-1]
            metrics["current_norm_score"] = float(latest_score)
            
            if len(self.drift_magnitude) > 0:
                _, latest_drift = self.drift_magnitude[-1]
                metrics["current_drift"] = float(latest_drift)
                
            if len(self.stability_variance) > 0:
                _, latest_variance = self.stability_variance[-1]
                metrics["current_stability"] = float(latest_variance)
                
            if len(self.directionality) > 0:
                _, latest_direction = self.directionality[-1]
                metrics["current_directionality"] = float(latest_direction)
                
            metrics.update(self.get_convergence_status())
            
        return metrics


def track_agent_norm_convergence(agent, num_observations=10, visualize=True, save_results=False, output_dir=None):
    """
    Utility function to track and analyze an agent's behavioral norm convergence.
    
    Args:
        agent: ΨC agent to analyze
        num_observations: Number of behavior observations to record
        visualize: Whether to visualize results
        save_results: Whether to save results to disk
        output_dir: Directory to save results (optional)
        
    Returns:
        Tuple of (convergence_status, tracker)
    """
    # Initialize tracker
    tracker = NormConvergenceTracker()
    
    # Record initial behaviors to establish baseline
    print("\nRecording initial behaviors to establish baseline...")
    for _ in range(tracker.baseline_window):
        tracker.record_behavior(agent)
        time.sleep(0.1)  # Simulate time passing
    
    # Record additional behaviors with simulated schema mutations
    print(f"\nRecording {num_observations-tracker.baseline_window} additional behaviors...")
    for i in range(num_observations - tracker.baseline_window):
        # Every few observations, record a schema mutation
        if i > 0 and i % 3 == 0:
            # Create a random mutation vector
            mutation_vector = np.random.random(tracker.vector_dim) * 0.2
            
            # Record the mutation
            tracker.record_schema_mutation(agent, mutation_vector)
            print(f"Recorded schema mutation #{i//3}")
        
        # Record behavior
        norm_score = tracker.record_behavior(agent)
        if norm_score is not None:
            print(f"Observation {i+tracker.baseline_window+1}: Norm Score = {norm_score:.4f}")
        
        time.sleep(0.1)  # Simulate time passing
    
    # Get convergence status
    status = tracker.get_convergence_status()
    
    # Print summary
    print("\nBehavioral Norm Convergence Analysis:")
    print(f"  Status: {status['status']}")
    print(f"  Norm Score: {status['norm_score']:.4f}")
    print(f"  Value Drift: {status['drift']:.4f}")
    print(f"  Stability: {status['stability']:.4f}")
    print(f"  Directionality: {status['directionality']:.4f}")
    print(f"  Is Converging: {status['is_converging']}")
    print(f"  Is Stable: {status['is_stable']}")
    
    # Visualize if requested
    if visualize:
        output_path = os.path.join(output_dir, "norm_convergence.png") if output_dir and save_results else None
        tracker.visualize_convergence(output_path=output_path)
    
    # Save results if requested
    if save_results:
        if output_dir is None:
            # Create a results directory
            output_dir = os.path.join(tracker.data_dir, f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
        # Save tracker state
        tracker.save_state(os.path.join(output_dir, "norm_tracker_state.json"))
    
    return status, tracker


if __name__ == "__main__":
    # Simple demo with mock agent
    class MockAgent:
        def __init__(self):
            pass
    
    # Create a mock agent
    agent = MockAgent()
    
    # Track behavioral norm convergence
    status, tracker = track_agent_norm_convergence(
        agent, 
        num_observations=15,
        visualize=True,
        save_results=True
    ) 