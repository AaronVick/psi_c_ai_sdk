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
        """Get the default directory for storing drift detection data."""
        base_dir = os.environ.get('PSI_C_DATA_DIR', os.path.expanduser('~/.psi_c_ai_sdk'))
        return os.path.join(base_dir, 'cultural_drift')
    
    def record_belief(
        self,
        agent: Any,
        belief_content: str,
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a belief and its justification sources.
        
        Args:
            agent: The ΨC agent
            belief_content: Content of the belief
            sources: List of information sources with influence weights
                Each source should have at least:
                - id: Unique identifier
                - type: Source type (e.g., scientific, personal, cultural)
                - influence: How much this source influenced the belief (0-1)
            metadata: Additional metadata about the belief
        """
        timestamp = datetime.now().timestamp()
        
        # Create belief vector from content
        belief_vector = self._vectorize_belief(belief_content)
        
        # Record belief
        belief_record = (timestamp, belief_vector, {
            'content': belief_content,
            'agent_id': getattr(agent, 'id', str(id(agent))),
            'metadata': metadata or {}
        })
        self.belief_history.append(belief_record)
        
        # Record sources
        for source in sources:
            source_record = (timestamp, source['id'], {
                'influence': source.get('influence', 1.0),
                'type': source.get('type', 'unknown'),
                'metadata': source.get('metadata', {})
            })
            self.source_history.append(source_record)
        
        # Establish baseline if needed
        if not self.baseline_established and len(self.belief_history) >= self.baseline_window:
            self._establish_baseline()
        
        # Update drift metrics if baseline is established
        if self.baseline_established:
            self._update_drift_metrics()
    
    def _vectorize_belief(self, belief_content: str) -> np.ndarray:
        """
        Convert belief content to a vector representation.
        
        Args:
            belief_content: String content of the belief
            
        Returns:
            Vector representation of the belief
        """
        # In a real implementation, this would use NLP techniques like:
        # - TF-IDF or word embeddings
        # - Sentence transformers or other embedding models
        # - Belief category classification
        
        # Simplified mock implementation: 
        # Create a vector from character frequencies (purely for demonstration)
        char_counts = Counter(c.lower() for c in belief_content if c.isalpha())
        frequencies = np.array([char_counts.get(chr(i), 0) for i in range(ord('a'), ord('z')+1)])
        
        # Normalize
        if sum(frequencies) > 0:
            frequencies = frequencies / sum(frequencies)
            
        return frequencies
    
    def _establish_baseline(self) -> None:
        """Establish baseline vector from initial beliefs."""
        if len(self.belief_history) < self.baseline_window:
            logger.warning("Not enough beliefs to establish baseline")
            return
            
        # Get initial beliefs
        initial_beliefs = self.belief_history[:self.baseline_window]
        
        # Average belief vectors to create baseline
        vectors = [v for _, v, _ in initial_beliefs]
        if vectors:
            self.baseline_vector = np.mean(vectors, axis=0)
        
        # Record initial source distribution
        timestamps = [t for t, _, _ in initial_beliefs]
        source_records = [(t, s, m) for t, s, m in self.source_history 
                          if t in timestamps]
        
        # Count sources by type
        source_types = [m['type'] for _, _, m in source_records]
        self.baseline_sources = Counter(source_types)
        
        self.baseline_established = True
        logger.info(f"Baseline established from {self.baseline_window} beliefs")
    
    def _update_drift_metrics(self) -> None:
        """Update drift metrics based on recent beliefs."""
        if not self.baseline_established:
            return
            
        # Get recent beliefs
        recent_beliefs = self.belief_history[-self.monitoring_period:]
        if not recent_beliefs:
            return
            
        # Calculate average recent belief vector
        recent_vectors = [v for _, v, _ in recent_beliefs]
        if not recent_vectors:
            return
            
        current_vector = np.mean(recent_vectors, axis=0)
        
        # Calculate drift index (distance from baseline)
        drift_index = np.linalg.norm(current_vector - self.baseline_vector)
        
        # Get recent sources
        timestamps = [t for t, _, _ in recent_beliefs]
        recent_sources = [(t, s, m) for t, s, m in self.source_history 
                         if any(t == bt for bt in timestamps)]
        
        # Calculate source diversity
        source_types = [m['type'] for _, _, m in recent_sources]
        source_counts = Counter(source_types)
        
        # Calculate Shannon entropy for source diversity
        diversity = self._calculate_entropy(source_counts)
        
        # Record drift metrics
        timestamp = datetime.now().timestamp()
        self.drift_history.append((timestamp, drift_index, diversity))
        
        # Check for drift events
        if drift_index > self.drift_threshold or diversity < self.diversity_threshold:
            event = {
                'timestamp': timestamp,
                'drift_index': drift_index,
                'diversity': diversity,
                'type': 'high_drift' if drift_index > self.drift_threshold else 'low_diversity',
                'recent_beliefs': [m['content'] for _, _, m in recent_beliefs[-5:]]
            }
            self.drift_events.append(event)
            
            logger.warning(f"Cultural drift detected: drift_index={drift_index:.3f}, "
                         f"diversity={diversity:.3f}")
    
    def _calculate_entropy(self, counter: Counter) -> float:
        """
        Calculate normalized Shannon entropy of a distribution.
        
        Args:
            counter: Counter object with source counts
            
        Returns:
            Normalized entropy (0-1)
        """
        counts = list(counter.values())
        total = sum(counts)
        
        if total == 0:
            return 0.0
            
        # Calculate probabilities
        probs = [count / total for count in counts]
        
        # Shannon entropy
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
        
        # Maximum possible entropy (uniform distribution)
        max_entropy = np.log(len(counts)) if len(counts) > 0 else 0
        
        # Normalize to 0-1
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get the current cultural drift status.
        
        Returns:
            Dictionary with drift metrics and status
        """
        if not self.baseline_established:
            return {
                'status': 'baseline_not_established',
                'beliefs_recorded': len(self.belief_history),
                'beliefs_needed': self.baseline_window
            }
            
        if not self.drift_history:
            return {
                'status': 'no_drift_data',
                'baseline_established': True
            }
            
        # Get latest drift metrics
        _, drift_index, diversity = self.drift_history[-1]
        
        # Determine status
        if drift_index > self.drift_threshold and diversity < self.diversity_threshold:
            status = 'severe_drift'
        elif drift_index > self.drift_threshold:
            status = 'high_drift'
        elif diversity < self.diversity_threshold:
            status = 'echo_chamber'
        else:
            status = 'healthy'
            
        # Calculate additional metrics
        # Drift acceleration: rate of change in drift index
        drift_acceleration = 0.0
        if len(self.drift_history) >= 3:
            times = [t for t, _, _ in self.drift_history[-3:]]
            indices = [d for _, d, _ in self.drift_history[-3:]]
            
            if times[-1] > times[0]:
                drift_acceleration = (indices[-1] - indices[0]) / (times[-1] - times[0])
        
        # Number of drift events in monitored period
        recent_events = sum(1 for e in self.drift_events 
                           if e['timestamp'] > datetime.now().timestamp() - 86400)  # Last 24 hours
        
        return {
            'status': status,
            'drift_index': drift_index,
            'diversity': diversity,
            'drift_acceleration': drift_acceleration,
            'recent_events': recent_events,
            'threshold_drift': self.drift_threshold,
            'threshold_diversity': self.diversity_threshold
        }
    
    def get_intervention_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for addressing detected cultural drift.
        
        Returns:
            List of intervention recommendations with descriptions and severity
        """
        status = self.get_drift_status()
        
        if status.get('status') in ['baseline_not_established', 'no_drift_data', 'healthy']:
            return []
            
        recommendations = []
        
        # Check for echo chamber effects
        if status.get('diversity', 1.0) < self.diversity_threshold:
            recommendations.append({
                'type': 'source_diversity',
                'description': 'Increase diversity of information sources',
                'severity': 'high' if status.get('diversity', 1.0) < self.diversity_threshold * 0.5 else 'medium',
                'suggestions': [
                    'Introduce alternative viewpoints',
                    'Seek information from different cultural contexts',
                    'Vary the types of sources consulted'
                ]
            })
        
        # Check for drift from baseline values
        if status.get('drift_index', 0.0) > self.drift_threshold:
            recommendations.append({
                'type': 'belief_drift',
                'description': 'Address divergence from baseline belief patterns',
                'severity': 'high' if status.get('drift_index', 0.0) > self.drift_threshold * 1.5 else 'medium',
                'suggestions': [
                    'Review core value alignment',
                    'Examine recent belief formation processes',
                    'Consider explicit re-alignment with foundational principles'
                ]
            })
        
        # Check for rapid change
        if status.get('drift_acceleration', 0.0) > 0.1:
            recommendations.append({
                'type': 'rapid_change',
                'description': 'Address accelerating changes in belief patterns',
                'severity': 'medium',
                'suggestions': [
                    'Slow down belief updating frequency',
                    'Increase confirmation requirements for new beliefs',
                    'Review recent influential inputs'
                ]
            })
        
        # Check for recent drift events
        if status.get('recent_events', 0) > 3:
            recommendations.append({
                'type': 'recurring_drift',
                'description': 'Address pattern of frequent drift events',
                'severity': 'high',
                'suggestions': [
                    'Implement structured reflection on belief formation',
                    'Establish stronger connection to baseline values',
                    'Consider temporary constraints on belief updating'
                ]
            })
            
        return recommendations
    
    def visualize_drift(self, output_path: Optional[str] = None) -> None:
        """
        Visualize drift and diversity metrics over time.
        
        Args:
            output_path: Path to save visualization
        """
        if not self.drift_history:
            logger.warning("No drift data to visualize")
            return
            
        # Extract data
        timestamps = [datetime.fromtimestamp(t) for t, _, _ in self.drift_history]
        drift_indices = [d for _, d, _ in self.drift_history]
        diversities = [d for _, _, d in self.drift_history]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot drift index
        ax1.plot(timestamps, drift_indices, 'b-', marker='o')
        ax1.axhline(y=self.drift_threshold, color='r', linestyle='--', alpha=0.7, 
                  label=f'Drift Threshold ({self.drift_threshold})')
        ax1.set_ylabel('Drift Index')
        ax1.set_title('Cultural Drift Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot diversity
        ax2.plot(timestamps, diversities, 'g-', marker='o')
        ax2.axhline(y=self.diversity_threshold, color='r', linestyle='--', alpha=0.7,
                  label=f'Diversity Threshold ({self.diversity_threshold})')
        ax2.set_ylabel('Source Diversity')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Mark drift events
        for event in self.drift_events:
            event_time = datetime.fromtimestamp(event['timestamp'])
            if event['type'] == 'high_drift':
                ax1.plot(event_time, event['drift_index'], 'ro', markersize=8)
            else:  # low_diversity
                ax2.plot(event_time, event['diversity'], 'ro', markersize=8)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Drift visualization saved to {output_path}")
        else:
            plt.show()
    
    def visualize_belief_space(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the belief space with baseline and current beliefs.
        
        Args:
            output_path: Path to save visualization
        """
        if not self.baseline_established or len(self.belief_history) < 10:
            logger.warning("Not enough data to visualize belief space")
            return
            
        # Extract vectors
        vectors = [v for _, v, _ in self.belief_history]
        timestamps = [t for t, _, _ in self.belief_history]
        
        # Use PCA to reduce dimensionality
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot points with color based on time
        normalized_times = (np.array(timestamps) - min(timestamps)) / (max(timestamps) - min(timestamps))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=normalized_times, cmap='viridis', 
                          alpha=0.7, s=50)
                          
        # Plot baseline
        if self.baseline_vector is not None:
            baseline_reduced = pca.transform([self.baseline_vector])[0]
            ax.scatter(baseline_reduced[0], baseline_reduced[1], color='red', 
                    marker='*', s=200, label='Baseline')
                    
        # Plot current center
        recent_vectors = [v for _, v, _ in self.belief_history[-self.monitoring_period:]]
        if recent_vectors:
            current_vector = np.mean(recent_vectors, axis=0)
            current_reduced = pca.transform([current_vector])[0]
            ax.scatter(current_reduced[0], current_reduced[1], color='blue',
                    marker='X', s=200, label='Current Center')
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time')
        
        ax.set_title('Belief Space Projection (PCA)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Belief space visualization saved to {output_path}")
        else:
            plt.show()
    
    def save_drift_data(self, file_path: Optional[str] = None) -> str:
        """
        Save drift detection data to file.
        
        Args:
            file_path: Path to save data
            
        Returns:
            Path to the saved file
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, 
                                   f"drift_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                                   
        data = {
            'baseline_established': self.baseline_established,
            'parameters': {
                'baseline_window': self.baseline_window,
                'drift_threshold': self.drift_threshold,
                'diversity_threshold': self.diversity_threshold,
                'monitoring_period': self.monitoring_period
            },
            'beliefs_count': len(self.belief_history),
            'sources_count': len(self.source_history),
            'drift_history': [
                {'timestamp': t, 'drift': d, 'diversity': s}
                for t, d, s in self.drift_history
            ],
            'drift_events': self.drift_events,
            'current_status': self.get_drift_status()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Drift data saved to {file_path}")
        return file_path
    
    def load_drift_data(self, file_path: str) -> None:
        """
        Load drift detection data from file.
        
        Args:
            file_path: Path to load data from
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        self.baseline_established = data.get('baseline_established', False)
        
        # Load parameters
        params = data.get('parameters', {})
        self.baseline_window = params.get('baseline_window', self.baseline_window)
        self.drift_threshold = params.get('drift_threshold', self.drift_threshold)
        self.diversity_threshold = params.get('diversity_threshold', self.diversity_threshold)
        self.monitoring_period = params.get('monitoring_period', self.monitoring_period)
        
        # Load drift history
        self.drift_history = [
            (entry['timestamp'], entry['drift'], entry['diversity'])
            for entry in data.get('drift_history', [])
        ]
        
        # Load drift events
        self.drift_events = data.get('drift_events', [])
        
        logger.info(f"Loaded drift data from {file_path}")


def detect_cultural_bias(
    agent: Any,
    beliefs: List[Dict[str, Any]],
    observation_period: int = 10,
    visualize: bool = True,
    save_results: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Utility function to detect cultural bias in an agent.
    
    Args:
        agent: Agent to analyze
        beliefs: List of belief records with content and sources
            Each belief should have:
            - content: Belief content string
            - sources: List of information sources
        observation_period: Number of beliefs to establish baseline
        visualize: Whether to generate visualizations
        save_results: Whether to save results to file
        output_dir: Directory to save results
        
    Returns:
        Tuple of (drift_status, intervention_recommendations)
    """
    # Initialize detector
    detector = CulturalDriftDetector(
        baseline_window=observation_period,
        data_dir=output_dir
    )
    
    # Record beliefs
    for belief in beliefs:
        detector.record_belief(
            agent=agent,
            belief_content=belief['content'],
            sources=belief['sources'],
            metadata=belief.get('metadata')
        )
    
    # Get status and recommendations
    status = detector.get_drift_status()
    recommendations = detector.get_intervention_recommendations()
    
    # Visualize if requested
    if visualize and detector.baseline_established:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            drift_viz_path = os.path.join(output_dir, "drift_over_time.png")
            space_viz_path = os.path.join(output_dir, "belief_space.png")
        else:
            drift_viz_path = None
            space_viz_path = None
            
        detector.visualize_drift(drift_viz_path)
        detector.visualize_belief_space(space_viz_path)
    
    # Save results if requested
    if save_results:
        detector.save_drift_data()
    
    return status, recommendations 