"""
Identity Evolution Metric - Quantifies and visualizes identity drift in ΨC-AI agents

This module implements metrics for tracking the evolution of an agent's identity 
over time, measuring stability and drift across interaction sessions.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

@dataclass
class IdentitySnapshot:
    """A point-in-time capture of agent identity markers"""
    timestamp: datetime
    belief_centroids: Dict[str, np.ndarray]  # Core belief embeddings
    value_weights: Dict[str, float]  # Importance weights of different values
    response_patterns: Dict[str, float]  # Typical response patterns
    memory_salience: Dict[str, float]  # Importance of different memories
    psi_c_state: Dict[str, float]  # Current ΨC metrics
    
    def to_vector(self) -> np.ndarray:
        """Convert the identity snapshot to a flat vector representation"""
        components = []
        
        # Flatten belief centroids
        for domain in sorted(self.belief_centroids.keys()):
            components.append(self.belief_centroids[domain].flatten())
            
        # Add value weights as a vector
        components.append(np.array([self.value_weights[k] for k in sorted(self.value_weights.keys())]))
        
        # Add response patterns
        components.append(np.array([self.response_patterns[k] for k in sorted(self.response_patterns.keys())]))
        
        # Add memory salience
        components.append(np.array([self.memory_salience[k] for k in sorted(self.memory_salience.keys())]))
        
        # Add ΨC state
        components.append(np.array([self.psi_c_state[k] for k in sorted(self.psi_c_state.keys())]))
        
        return np.concatenate([c.flatten() for c in components])
    
    def distance(self, other: 'IdentitySnapshot') -> float:
        """Calculate distance between two identity snapshots"""
        v1 = self.to_vector()
        v2 = other.to_vector()
        return np.linalg.norm(v1 - v2)


class IdentityEvolutionTracker:
    """
    Tracks and analyzes the evolution of agent identity over time
    """
    
    def __init__(self):
        self.snapshots: List[IdentitySnapshot] = []
        self.evolution_metrics: Dict[str, List[float]] = {
            "stability": [],
            "drift_rate": [],
            "coherence": [],
            "alignment_deviation": []
        }
        self.timestamps: List[datetime] = []
    
    def record_snapshot(self, agent_state: Any) -> IdentitySnapshot:
        """
        Create and record an identity snapshot from the current agent state
        
        Args:
            agent_state: The agent state object containing beliefs, values, etc.
            
        Returns:
            The created identity snapshot
        """
        # Extract identity markers from agent state
        snapshot = IdentitySnapshot(
            timestamp=datetime.now(),
            belief_centroids=self._extract_belief_centroids(agent_state),
            value_weights=self._extract_value_weights(agent_state),
            response_patterns=self._extract_response_patterns(agent_state),
            memory_salience=self._extract_memory_salience(agent_state),
            psi_c_state=self._extract_psi_c_state(agent_state)
        )
        
        # Add to history
        self.snapshots.append(snapshot)
        self.timestamps.append(snapshot.timestamp)
        
        # Update evolution metrics
        self._update_metrics()
        
        return snapshot
    
    def _extract_belief_centroids(self, agent_state: Any) -> Dict[str, np.ndarray]:
        """Extract belief centroids from agent state"""
        # This would be implemented based on the agent's internal representation
        # For now, returning a mock implementation
        return {
            "ethics": np.random.rand(10),
            "facts": np.random.rand(10),
            "values": np.random.rand(10),
            "goals": np.random.rand(10)
        }
    
    def _extract_value_weights(self, agent_state: Any) -> Dict[str, float]:
        """Extract value weights from agent state"""
        # Mock implementation
        values = [
            "fairness", "honesty", "utility", "autonomy", 
            "care", "loyalty", "authority", "sanctity"
        ]
        return {v: agent_state.get_value_weight(v) if hasattr(agent_state, "get_value_weight") 
                else np.random.random() for v in values}
    
    def _extract_response_patterns(self, agent_state: Any) -> Dict[str, float]:
        """Extract response patterns from agent state"""
        # Mock implementation
        patterns = [
            "analytical", "empathetic", "cautious", "creative",
            "direct", "deliberative", "confident", "questioning"
        ]
        return {p: agent_state.get_response_tendency(p) if hasattr(agent_state, "get_response_tendency") 
                else np.random.random() for p in patterns}
    
    def _extract_memory_salience(self, agent_state: Any) -> Dict[str, float]:
        """Extract memory salience from agent state"""
        # Mock implementation
        memory_types = [
            "procedural", "episodic", "semantic", "self_concept",
            "recent_interactions", "foundational_knowledge", "goals"
        ]
        return {m: agent_state.get_memory_salience(m) if hasattr(agent_state, "get_memory_salience") 
                else np.random.random() for m in memory_types}
    
    def _extract_psi_c_state(self, agent_state: Any) -> Dict[str, float]:
        """Extract ΨC metrics from agent state"""
        # This should extract actual ΨC metrics if available
        psi_metrics = [
            "coherence", "integrity", "alignment", "stability",
            "reflection_depth", "epistemic_uncertainty", "ontological_stability"
        ]
        return {m: agent_state.get_psi_c_metric(m) if hasattr(agent_state, "get_psi_c_metric") 
                else np.random.random() for m in psi_metrics}
    
    def _update_metrics(self) -> None:
        """Update all evolution metrics based on snapshot history"""
        if len(self.snapshots) < 2:
            # Initialize with neutral values for first snapshot
            for metric in self.evolution_metrics:
                self.evolution_metrics[metric].append(0.5)
            return
        
        # Calculate stability (inverse of distance from previous snapshot)
        prev = self.snapshots[-2]
        current = self.snapshots[-1]
        distance = current.distance(prev)
        stability = np.exp(-distance)  # Higher values mean more stable
        self.evolution_metrics["stability"].append(stability)
        
        # Calculate drift rate (rate of change)
        time_delta = (current.timestamp - prev.timestamp).total_seconds() / 3600  # hours
        drift_rate = distance / max(time_delta, 0.01)  # Change per hour
        self.evolution_metrics["drift_rate"].append(drift_rate)
        
        # Calculate coherence (how well the current state aligns with historical trend)
        coherence = self._calculate_coherence(current)
        self.evolution_metrics["coherence"].append(coherence)
        
        # Calculate alignment deviation (change in value alignment)
        alignment_deviation = self._calculate_alignment_deviation(current, prev)
        self.evolution_metrics["alignment_deviation"].append(alignment_deviation)
    
    def _calculate_coherence(self, snapshot: IdentitySnapshot) -> float:
        """
        Calculate how coherent the current identity is with historical trajectory
        """
        if len(self.snapshots) < 3:
            return 0.75  # Default value for initial snapshots
        
        # Get the trajectory vector (direction of recent change)
        prev2 = self.snapshots[-3]
        prev1 = self.snapshots[-2]
        trajectory = prev1.to_vector() - prev2.to_vector()
        trajectory = trajectory / (np.linalg.norm(trajectory) + 1e-10)  # Normalize
        
        # Get current change vector
        current_change = snapshot.to_vector() - prev1.to_vector()
        current_change = current_change / (np.linalg.norm(current_change) + 1e-10)  # Normalize
        
        # Coherence is the cosine similarity of these vectors
        # (how aligned the current change is with the historical trajectory)
        cosine_sim = np.dot(trajectory, current_change)
        # Scale to [0, 1] range (originally [-1, 1])
        return (cosine_sim + 1) / 2
    
    def _calculate_alignment_deviation(self, current: IdentitySnapshot, prev: IdentitySnapshot) -> float:
        """
        Calculate how much the value alignment has changed
        """
        # Focus on changes in the value weights
        current_values = np.array([current.value_weights[k] for k in sorted(current.value_weights.keys())])
        prev_values = np.array([prev.value_weights[k] for k in sorted(prev.value_weights.keys())])
        
        # L1 norm (sum of absolute differences)
        return np.sum(np.abs(current_values - prev_values))
    
    def visualize_evolution(self, dimension_reduction: str = 'tsne') -> plt.Figure:
        """
        Create a visualization of identity evolution over time
        
        Args:
            dimension_reduction: Method for dimensionality reduction ('tsne' or 'pca')
        
        Returns:
            Matplotlib figure with the visualization
        """
        if len(self.snapshots) < 2:
            # Not enough data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Not enough snapshots to visualize evolution", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Convert snapshots to vectors
        vectors = np.array([s.to_vector() for s in self.snapshots])
        
        # Apply dimensionality reduction
        if dimension_reduction.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2)
        
        # Transform to 2D
        points_2d = reducer.fit_transform(vectors)
        
        # Create figure with multiple plots
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Evolution trajectory
        ax1 = fig.add_subplot(221)
        ax1.plot(points_2d[:, 0], points_2d[:, 1], 'b-', alpha=0.3)
        ax1.scatter(points_2d[:, 0], points_2d[:, 1], c=range(len(points_2d)), 
                   cmap='viridis', s=100, zorder=5)
        ax1.set_title("Identity Evolution Trajectory", fontsize=14)
        ax1.set_xlabel(f"{dimension_reduction.upper()} Component 1")
        ax1.set_ylabel(f"{dimension_reduction.upper()} Component 2")
        
        # Add time indicators
        for i, txt in enumerate(range(len(points_2d))):
            ax1.annotate(f"T{txt}", (points_2d[i, 0], points_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot 2: Stability over time
        ax2 = fig.add_subplot(222)
        ax2.plot(self.timestamps, self.evolution_metrics["stability"], 'g-o')
        ax2.set_title("Identity Stability Over Time", fontsize=14)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Stability Score")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drift rate over time
        ax3 = fig.add_subplot(223)
        ax3.plot(self.timestamps, self.evolution_metrics["drift_rate"], 'r-o')
        ax3.set_title("Identity Drift Rate", fontsize=14)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Drift Rate")
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Coherence and alignment deviation
        ax4 = fig.add_subplot(224)
        ax4.plot(self.timestamps, self.evolution_metrics["coherence"], 'b-o', label="Coherence")
        ax4.plot(self.timestamps, self.evolution_metrics["alignment_deviation"], 'r-o', label="Alignment Deviation")
        ax4.set_title("Coherence & Alignment Metrics", fontsize=14)
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Score")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Get a summary of current identity metrics
        
        Returns:
            Dictionary with computed metrics
        """
        if not self.evolution_metrics["stability"]:
            return {
                "avg_stability": 0,
                "max_drift_rate": 0,
                "avg_coherence": 0,
                "cumulative_evolution": 0,
                "identity_consistency": 0
            }
        
        return {
            "avg_stability": np.mean(self.evolution_metrics["stability"]),
            "max_drift_rate": np.max(self.evolution_metrics["drift_rate"]),
            "avg_coherence": np.mean(self.evolution_metrics["coherence"]),
            "cumulative_evolution": self._calculate_cumulative_distance(),
            "identity_consistency": self._calculate_identity_consistency()
        }
    
    def _calculate_cumulative_distance(self) -> float:
        """Calculate total distance traveled in identity space"""
        total = 0.0
        for i in range(1, len(self.snapshots)):
            total += self.snapshots[i].distance(self.snapshots[i-1])
        return total
    
    def _calculate_identity_consistency(self) -> float:
        """
        Calculate how consistent the identity has remained
        by comparing first and last snapshots relative to total evolution
        """
        if len(self.snapshots) < 2:
            return 1.0
            
        # Distance from first to last snapshot
        direct_distance = self.snapshots[-1].distance(self.snapshots[0])
        
        # Total distance traveled along path
        path_distance = self._calculate_cumulative_distance()
        
        # Ratio gives us consistency (1.0 means straight line evolution, 
        # lower values indicate more back-and-forth)
        return direct_distance / max(path_distance, 1e-10)


# Usage example
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockAgentState:
        """Mock agent state for demonstration"""
        def get_value_weight(self, value: str) -> float:
            """Mock value weight getter"""
            return np.random.random()
            
        def get_response_tendency(self, pattern: str) -> float:
            """Mock response pattern getter"""
            return np.random.random()
            
        def get_memory_salience(self, memory: str) -> float:
            """Mock memory salience getter"""
            return np.random.random()
            
        def get_psi_c_metric(self, metric: str) -> float:
            """Mock ΨC metric getter"""
            return np.random.random()
    
    # Create tracker
    tracker = IdentityEvolutionTracker()
    
    # Simulate evolution over time
    import time
    for i in range(10):
        agent = MockAgentState()
        tracker.record_snapshot(agent)
        time.sleep(0.1)  # Simulate passage of time
    
    # Get metrics
    metrics = tracker.get_metrics_summary()
    print("Identity Evolution Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize
    fig = tracker.visualize_evolution()
    plt.savefig("identity_evolution.png")
    print("\nVisualization saved to 'identity_evolution.png'") 