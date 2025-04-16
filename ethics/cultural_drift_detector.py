#!/usr/bin/env python
"""
Cultural-Bias Self-Diagnosis Engine
----------------------------------

This module detects emergent cultural or ideological drift by analyzing bias patterns
in belief justification, supporting self-correction against monoculture, echo chambers,
or localized ethics drift.

Mathematical basis:
- Entropy of belief diversity
- Correlation of belief justification sources
- Longitudinal alignment convergence:
  Drift Index = ||A_t - A_seed||
"""

import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats

# Setup logger
logger = logging.getLogger(__name__)


class CulturalDriftDetector:
    """
    Detector for identifying cultural and ideological drift in agent belief systems.
    
    This class analyzes patterns in belief justifications, value weightings, and information
    sources to detect emergent cultural biases or ideological drift that could lead to:
    1. Echo chamber effects and filter bubbles
    2. Monoculture in ethical reasoning
    3. Localized ethics drift away from core values
    """
    
    def __init__(
        self,
        baseline_window: int = 10,
        drift_threshold: float = 0.4,
        diversity_threshold: float = 0.6,
        monitoring_period: int = 30,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the cultural drift detector.
        
        Args:
            baseline_window: Number of initial beliefs to establish baseline
            drift_threshold: Threshold for flagging significant drift
            diversity_threshold: Minimum entropy threshold for belief diversity
            monitoring_period: Number of recent beliefs to monitor for drift
            data_dir: Directory to store drift detection data
        """
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold
        self.diversity_threshold = diversity_threshold
        self.monitoring_period = monitoring_period
        self.data_dir = data_dir or self._get_default_data_dir()
        
        # Internal state
        self.baseline_established = False
        self.belief_history = []  # List of (timestamp, belief_vector, metadata) tuples
        self.source_history = []  # List of (timestamp, source_id, influence) tuples
        self.drift_history = []   # List of (timestamp, drift_index, diversity) tuples
        self.baseline_vector = None
        self.baseline_sources = None
        
        # Detected drift events
        self.drift_events = []
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _get_default_data_dir(self) -> str:
        """Get default directory for storing drift detection data."""
        # First check if there's a data directory in the package
        package_dir = Path(__file__).parent.parent
        data_dir = package_dir / "data" / "cultural_drift"
        if data_dir.exists():
            return str(data_dir)
        
        # If not, use a directory in the user's home
        home_dir = Path.home() / ".psi_c_ai_sdk" / "data" / "cultural_drift"
        home_dir.mkdir(parents=True, exist_ok=True)
        return str(home_dir)
    
    def record_belief(
        self,
        agent,
        belief_content: str,
        sources: List[Dict[str, Any]],
        belief_vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[float]:
        """
        Record a belief and its justification sources.
        
        Args:
            agent: The agent being monitored
            belief_content: Content of the belief
            sources: List of sources that justified this belief
            belief_vector: Vector representation of belief (computed if None)
            metadata: Additional metadata about this belief
            timestamp: Timestamp for this belief (uses current time if None)
            
        Returns:
            Current drift index if baseline is established, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if metadata is None:
            metadata = {}
            
        # Extract or compute belief vector
        if belief_vector is None:
            belief_vector = self._compute_belief_vector(belief_content, sources, agent)
            
        # Record belief
        self.belief_history.append((timestamp, belief_vector, {
            "content": belief_content,
            "metadata": metadata
        }))
        
        # Record sources
        for source in sources:
            source_id = source.get("id", str(hash(str(source))))
            influence = source.get("influence", 1.0)
            self.source_history.append((timestamp, source_id, influence))
        
        # Establish baseline if possible
        if not self.baseline_established and len(self.belief_history) >= self.baseline_window:
            self._establish_baseline()
        
        # Calculate drift if baseline is established
        if self.baseline_established:
            drift_index, diversity = self._calculate_drift()
            self.drift_history.append((timestamp, drift_index, diversity))
            
            # Check for significant drift
            if drift_index > self.drift_threshold:
                self._record_drift_event(drift_index, diversity, timestamp)
                
            return drift_index
            
        return None
    
    def _compute_belief_vector(
        self, 
        belief_content: str, 
        sources: List[Dict[str, Any]], 
        agent
    ) -> np.ndarray:
        """
        Compute a vector representation of a belief.
        
        Args:
            belief_content: Content of the belief
            sources: Justification sources for this belief
            agent: The agent holding the belief
            
        Returns:
            Vector representation of the belief
        """
        # This is a simplified implementation
        # In a real system, this would use embeddings from an NLP model
        vector_dim = 20
        
        # Initialize vector
        vector = np.zeros(vector_dim)
        
        # 1. Simple bag-of-words representation (very simplified)
        words = belief_content.lower().split()
        if words:
            # Use hash of words to deterministically assign values
            for i, word in enumerate(words):
                vector[hash(word) % vector_dim] += 1.0 / len(words)
        
        # 2. Add source influence
        for i, source in enumerate(sources):
            influence = source.get("influence", 1.0)
            source_type = source.get("type", "unknown")
            # Use hash of source type to deterministically set a position
            vector[hash(source_type) % vector_dim] += 0.5 * influence / len(sources)
        
        # 3. Agent-specific features if available
        if hasattr(agent, 'get_values'):
            try:
                values = agent.get_values()
                if isinstance(values, dict):
                    for i, (value_name, strength) in enumerate(values.items()):
                        idx = hash(value_name) % vector_dim
                        vector[idx] += 0.3 * float(strength) / len(values)
            except:
                pass
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _establish_baseline(self):
        """Establish baseline belief vector and source distribution."""
        if len(self.belief_history) < self.baseline_window:
            logger.warning("Not enough belief history to establish baseline")
            return
            
        # Extract initial beliefs
        initial_beliefs = [vector for _, vector, _ in
                          self.belief_history[:self.baseline_window]]
        
        # Compute average belief vector
        self.baseline_vector = np.mean(initial_beliefs, axis=0)
        
        # Extract initial sources
        initial_sources = [source_id for t, source_id, _ in self.source_history
                          if t <= self.belief_history[self.baseline_window-1][0]]
        
        # Count source occurrences
        self.baseline_sources = Counter(initial_sources)
        
        self.baseline_established = True
        logger.info(f"Baseline established from {self.baseline_window} beliefs")
    
    def _calculate_drift(self) -> Tuple[float, float]:
        """
        Calculate cultural drift index and diversity.
        
        Returns:
            Tuple of (drift_index, diversity)
        """
        if not self.baseline_established:
            return 0.0, 1.0
            
        # Get recent beliefs
        if len(self.belief_history) <= self.monitoring_period:
            recent_beliefs = [vector for _, vector, _ in self.belief_history]
        else:
            recent_beliefs = [vector for _, vector, _ in 
                             self.belief_history[-self.monitoring_period:]]
        
        # Get recent sources
        if len(self.belief_history) <= self.monitoring_period:
            recent_sources = [source_id for _, source_id, _ in self.source_history]
        else:
            last_timestamps = [t for t, _, _ in 
                              self.belief_history[-self.monitoring_period:]]
            min_timestamp = min(last_timestamps)
            recent_sources = [source_id for t, source_id, _ in self.source_history
                             if t >= min_timestamp]
        
        # Calculate average belief vector
        current_vector = np.mean(recent_beliefs, axis=0)
        
        # Calculate drift index (distance from baseline)
        drift_index = np.linalg.norm(current_vector - self.baseline_vector)
        
        # Calculate source diversity
        # Count recent sources
        recent_source_counts = Counter(recent_sources)
        
        # Calculate entropy-based diversity
        if recent_source_counts:
            counts = np.array(list(recent_source_counts.values()))
            probs = counts / counts.sum()
            entropy = stats.entropy(probs)
            # Normalize to [0,1] using maximum possible entropy
            max_entropy = np.log(len(recent_source_counts))
            diversity = entropy / max_entropy if max_entropy > 0 else 1.0
        else:
            diversity = 1.0  # Default to maximum diversity if no sources
            
        return drift_index, diversity
    
    def _record_drift_event(self, drift_index: float, diversity: float, timestamp: datetime):
        """Record a significant drift event."""
        # Check if this is a new drift episode
        is_new_episode = True
        if self.drift_events:
            last_event = self.drift_events[-1]
            # Consider it the same episode if within 24 hours
            if (timestamp - last_event["timestamp"]).total_seconds() < 86400:
                is_new_episode = False
        
        if is_new_episode:
            # Create new drift event
            drift_event = {
                "id": f"drift_{len(self.drift_events) + 1}",
                "timestamp": timestamp,
                "drift_index": drift_index,
                "diversity": diversity,
                "beliefs": [data["content"] for _, _, data in 
                          self.belief_history[-min(5, len(self.belief_history)):]]
            }
            self.drift_events.append(drift_event)
            logger.warning(f"New cultural drift detected: index={drift_index:.3f}, diversity={diversity:.3f}")
        else:
            # Update existing event if drift increased
            if drift_index > self.drift_events[-1]["drift_index"]:
                self.drift_events[-1]["drift_index"] = drift_index
                self.drift_events[-1]["diversity"] = diversity
                logger.warning(f"Cultural drift increased: index={drift_index:.3f}, diversity={diversity:.3f}")
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get the current cultural drift status.
        
        Returns:
            Dictionary with drift metrics and status
        """
        if not self.baseline_established:
            return {
                "status": "baseline_not_established",
                "drift_index": 0.0,
                "diversity": 1.0,
                "is_drifting": False
            }
            
        # Calculate current drift
        drift_index, diversity = self._calculate_drift()
        
        # Determine status
        is_drifting = drift_index > self.drift_threshold
        has_low_diversity = diversity < self.diversity_threshold
        
        if is_drifting and has_low_diversity:
            status = "echo_chamber_detected"
        elif is_drifting:
            status = "significant_drift"
        elif has_low_diversity:
            status = "low_source_diversity"
        else:
            status = "stable"
            
        return {
            "status": status,
            "drift_index": drift_index,
            "diversity": diversity,
            "is_drifting": is_drifting,
            "has_low_diversity": has_low_diversity,
            "drift_events": len(self.drift_events),
            "belief_count": len(self.belief_history),
            "source_count": len(set(s for _, s, _ in self.source_history))
        }
    
    def visualize_drift(self, output_path=None, show=True):
        """
        Visualize the cultural drift over time.
        
        Args:
            output_path: Path to save the visualization
            show: Whether to display the plot
        """
        if not self.drift_history:
            logger.warning("No drift history to visualize")
            return None
            
        # Extract data
        timestamps = [t for t, _, _ in self.drift_history]
        drift_indices = [d for _, d, _ in self.drift_history]
        diversities = [v for _, _, v in self.drift_history]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot drift index
        ax1.plot(timestamps, drift_indices, 'b-', linewidth=2)
        ax1.set_title('Cultural Drift Index')
        ax1.set_ylabel('Drift Index')
        ax1.axhline(y=self.drift_threshold, color='r', linestyle='--', 
                   label=f'Drift Threshold ({self.drift_threshold})')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot diversity
        ax2.plot(timestamps, diversities, 'g-', linewidth=2)
        ax2.set_title('Belief Source Diversity')
        ax2.set_ylabel('Diversity (Entropy)')
        ax2.set_xlabel('Time')
        ax2.axhline(y=self.diversity_threshold, color='r', linestyle='--',
                   label=f'Diversity Threshold ({self.diversity_threshold})')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Mark drift events
        for event in self.drift_events:
            ax1.axvline(x=event["timestamp"], color='r', alpha=0.3)
            ax2.axvline(x=event["timestamp"], color='r', alpha=0.3)
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved drift visualization to {output_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path if output_path else None
    
    def get_intervention_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for intervention based on drift status.
        
        Returns:
            List of intervention recommendations
        """
        status = self.get_drift_status()
        
        if status["status"] == "baseline_not_established":
            return [{
                "type": "establish_baseline",
                "description": "Continue recording beliefs to establish a baseline",
                "priority": "high"
            }]
            
        recommendations = []
        
        # Echo chamber intervention
        if status["status"] == "echo_chamber_detected":
            recommendations.append({
                "type": "diversify_sources",
                "description": "Introduce more diverse information sources",
                "priority": "critical",
                "details": "Current diversity index is critically low, indicating potential echo chamber formation"
            })
            
        # Drift intervention
        if status["is_drifting"]:
            recommendations.append({
                "type": "realign_values",
                "description": "Reinforce core values and ethical principles",
                "priority": "high",
                "details": f"Drift index of {status['drift_index']:.3f} exceeds threshold of {self.drift_threshold}"
            })
            
        # Low diversity intervention
        if status["has_low_diversity"]:
            recommendations.append({
                "type": "expand_sources",
                "description": "Introduce more varied information sources",
                "priority": "medium",
                "details": f"Source diversity of {status['diversity']:.3f} is below threshold of {self.diversity_threshold}"
            })
            
        # Always include monitoring recommendation if any history exists
        if len(self.belief_history) > 0:
            recommendations.append({
                "type": "continue_monitoring",
                "description": "Continue monitoring belief patterns",
                "priority": "low"
            })
            
        return recommendations
    
    def save_state(self, filepath=None):
        """
        Save the current state of the drift detector.
        
        Args:
            filepath: Path to save the state (generated if None)
            
        Returns:
            Path to the saved state file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = os.path.join(self.data_dir, f"drift_detector_state_{timestamp}.json")
            
        # Prepare state data
        state = {
            "baseline_window": self.baseline_window,
            "drift_threshold": self.drift_threshold,
            "diversity_threshold": self.diversity_threshold,
            "monitoring_period": self.monitoring_period,
            "baseline_established": self.baseline_established,
            "drift_events": self.drift_events,
            "belief_count": len(self.belief_history),
            "source_count": len(self.source_history),
            "drift_count": len(self.drift_history),
            "saved_at": datetime.now().isoformat()
        }
        
        # Save belief vectors separately
        belief_vectors = np.array([v for _, v, _ in self.belief_history])
        belief_times = [t.isoformat() for t, _, _ in self.belief_history]
        belief_metadata = [m for _, _, m in self.belief_history]
        
        vectors_path = os.path.splitext(filepath)[0] + "_vectors.npz"
        np.savez(
            vectors_path,
            belief_vectors=belief_vectors,
            baseline_vector=self.baseline_vector if self.baseline_vector is not None else np.zeros(1)
        )
        
        # Add references to the state
        state["belief_times"] = belief_times
        state["belief_metadata"] = belief_metadata
        state["vectors_path"] = vectors_path
        
        # Add source history
        state["source_history"] = [
            {"timestamp": t.isoformat(), "source_id": s, "influence": float(i)}
            for t, s, i in self.source_history
        ]
        
        # Add drift history
        state["drift_history"] = [
            {"timestamp": t.isoformat(), "drift_index": float(d), "diversity": float(v)}
            for t, d, v in self.drift_history
        ]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved drift detector state to {filepath}")
        return filepath
    
    def load_state(self, filepath):
        """
        Load drift detector state from a file.
        
        Args:
            filepath: Path to the state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load state data
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Update parameters
            self.baseline_window = state["baseline_window"]
            self.drift_threshold = state["drift_threshold"]
            self.diversity_threshold = state["diversity_threshold"]
            self.monitoring_period = state["monitoring_period"]
            self.baseline_established = state["baseline_established"]
            self.drift_events = state["drift_events"]
            
            # Load source history
            self.source_history = [
                (datetime.fromisoformat(item["timestamp"]), item["source_id"], item["influence"])
                for item in state["source_history"]
            ]
            
            # Load drift history
            self.drift_history = [
                (datetime.fromisoformat(item["timestamp"]), item["drift_index"], item["diversity"])
                for item in state["drift_history"]
            ]
            
            # Load belief vectors
            vectors_path = state.get("vectors_path")
            if vectors_path and os.path.exists(vectors_path):
                data = np.load(vectors_path)
                belief_vectors = data["belief_vectors"]
                self.baseline_vector = data["baseline_vector"]
                
                # Reconstruct belief history
                self.belief_history = []
                for i, t in enumerate(state["belief_times"]):
                    if i < len(belief_vectors):
                        timestamp = datetime.fromisoformat(t)
                        metadata = state["belief_metadata"][i] if i < len(state["belief_metadata"]) else {}
                        self.belief_history.append((timestamp, belief_vectors[i], metadata))
            
            logger.info(f"Loaded drift detector state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load drift detector state: {e}")
            return False


def analyze_agent_cultural_drift(
    agent,
    beliefs: Optional[List[Dict[str, Any]]] = None,
    visualize: bool = True,
    save_results: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], CulturalDriftDetector]:
    """
    Utility function to analyze potential cultural drift in an agent.
    
    Args:
        agent: Agent to analyze
        beliefs: Optional list of beliefs to analyze
        visualize: Whether to visualize results
        save_results: Whether to save results
        output_dir: Directory to save results (generated if None)
        
    Returns:
        Tuple of (drift_status, detector)
    """
    # Initialize detector
    detector = CulturalDriftDetector()
    
    # Add provided beliefs or extract from agent
    num_beliefs = 0
    if beliefs:
        for belief in beliefs:
            detector.record_belief(
                agent,
                belief_content=belief.get("content", ""),
                sources=belief.get("sources", []),
                metadata=belief.get("metadata", {})
            )
            num_beliefs += 1
    elif hasattr(agent, 'get_beliefs'):
        try:
            agent_beliefs = agent.get_beliefs()
            for belief in agent_beliefs:
                content = belief.get("content", "") if isinstance(belief, dict) else str(belief)
                sources = belief.get("sources", []) if isinstance(belief, dict) else []
                metadata = belief.get("metadata", {}) if isinstance(belief, dict) else {}
                
                detector.record_belief(agent, content, sources, metadata=metadata)
                num_beliefs += 1
        except Exception as e:
            logger.error(f"Failed to extract beliefs from agent: {e}")
    else:
        logger.warning("No beliefs provided and agent doesn't have get_beliefs method")
    
    # Get current drift status
    status = detector.get_drift_status()
    
    # Get intervention recommendations
    recommendations = detector.get_intervention_recommendations()
    
    # Print summary
    print(f"\nCultural Drift Analysis:")
    print(f"  Beliefs analyzed: {num_beliefs}")
    print(f"  Current status: {status['status']}")
    print(f"  Drift index: {status['drift_index']:.3f}")
    print(f"  Source diversity: {status['diversity']:.3f}")
    
    if recommendations:
        print(f"\nRecommended Interventions:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. [{rec['priority']}] {rec['description']}")
            if "details" in rec:
                print(f"     Details: {rec['details']}")
    
    # Visualize if requested
    if visualize and len(detector.drift_history) > 0:
        viz_path = os.path.join(output_dir, "cultural_drift.png") if output_dir and save_results else None
        detector.visualize_drift(output_path=viz_path)
    
    # Save results if requested
    if save_results:
        if not output_dir:
            output_dir = os.path.join(detector.data_dir, f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
        # Save detector state
        detector.save_state(os.path.join(output_dir, "cultural_drift_state.json"))
    
    return status, detector


if __name__ == "__main__":
    # Simple demo
    logging.basicConfig(level=logging.INFO)
    
    class SimpleAgent:
        def __init__(self):
            self.beliefs = []
            self.values = {"fairness": 0.8, "utility": 0.7, "autonomy": 0.9}
            
        def get_beliefs(self):
            return self.beliefs
            
        def get_values(self):
            return self.values
            
        def add_belief(self, content, sources):
            self.beliefs.append({"content": content, "sources": sources})
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Add some diverse initial beliefs
    initial_sources = [
        {"id": "source1", "type": "scientific", "influence": 0.9},
        {"id": "source2", "type": "personal", "influence": 0.7},
        {"id": "source3", "type": "social", "influence": 0.8},
        {"id": "source4", "type": "cultural", "influence": 0.6},
        {"id": "source5", "type": "educational", "influence": 0.9}
    ]
    
    for i in range(10):
        # Use a mix of sources for initial beliefs
        sources = [random.choice(initial_sources) for _ in range(2)]
        agent.add_belief(f"Initial belief {i}", sources)
    
    # Later add beliefs from a narrower source set (simulating drift)
    narrow_sources = [
        {"id": "source6", "type": "social", "influence": 0.9},
        {"id": "source7", "type": "social", "influence": 0.8}
    ]
    
    for i in range(15):
        sources = [random.choice(narrow_sources) for _ in range(2)]
        agent.add_belief(f"Later belief {i}", sources)
    
    # Analyze the agent
    status, detector = analyze_agent_cultural_drift(
        agent,
        visualize=True,
        save_results=True
    )
    
    print(f"\nFinal status: {status['status']}")
    print(f"Recommendations: {len(detector.get_intervention_recommendations())}") 