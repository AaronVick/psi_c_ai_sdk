"""
ΨC Agent Ethogram Builder

This module implements an ethogram builder for ΨC agents, tracking and classifying
recurring behavioral patterns over time. Similar to how ethologists create ethograms
for animals, this system creates a synthetic "ethogram" for AI agents.

Key features:
- Behavioral pattern detection and classification
- Temporal frequency analysis of behaviors
- Visualization of behavioral shifts over time
- Taxonomy generation for multi-agent comparison

Usage:
```python
from psi_c_ai_sdk.analytics.ethogram import AgentEthogramBuilder

ethogram_builder = AgentEthogramBuilder()
ethogram_builder.add_observation(agent)
...
ethogram = ethogram_builder.build_ethogram()
ethogram_builder.visualize()
```
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass
class BehavioralPattern:
    """
    Represents a distinct behavioral pattern identified in an agent.
    """
    id: str
    name: str
    feature_vector: np.ndarray
    frequency: float = 0.0
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_frequency(self, total_observations: int) -> None:
        """Update the frequency based on total observations."""
        self.frequency = len(self.observations) / total_observations if total_observations > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert behavioral pattern to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "frequency": self.frequency,
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
            "observation_count": len(self.observations),
            "metadata": self.metadata
        }


@dataclass
class AgentEthogram:
    """
    Represents a complete ethogram for an agent, containing classified behavioral patterns,
    their frequencies, and temporal distribution.
    """
    agent_id: str
    patterns: Dict[str, BehavioralPattern] = field(default_factory=dict)
    temporal_data: List[Dict[str, Any]] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_pattern_frequencies(self) -> Dict[str, float]:
        """Get the frequency distribution of patterns."""
        return {pattern_id: pattern.frequency for pattern_id, pattern in self.patterns.items()}
    
    def get_dominant_patterns(self, top_n: int = 3) -> List[BehavioralPattern]:
        """Get the top N most frequent behavioral patterns."""
        sorted_patterns = sorted(
            self.patterns.values(), 
            key=lambda p: p.frequency, 
            reverse=True
        )
        return sorted_patterns[:top_n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ethogram to dictionary."""
        return {
            "agent_id": self.agent_id,
            "patterns": {p_id: p.to_dict() for p_id, p in self.patterns.items()},
            "temporal_data_points": len(self.temporal_data),
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }
    
    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """
        Convert ethogram to JSON and optionally save to file.
        
        Args:
            path: Optional file path to save JSON
            
        Returns:
            JSON string if path is None, otherwise None
        """
        ethogram_dict = self.to_dict()
        json_str = json.dumps(ethogram_dict, indent=2)
        
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
            return None
        
        return json_str


class AgentEthogramBuilder:
    """
    Builder for creating and updating an ethogram of agent behaviors over time.
    
    This class tracks agent behaviors, identifies recurring patterns, and
    constructs a synthetic ethogram that classifies and quantifies behavioral
    repertoires.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        feature_extractors: Optional[Dict[str, callable]] = None,
        clustering_method: str = "kmeans",
        n_clusters: int = 8,
        min_observations: int = 5,
        data_dir: Optional[str] = None,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the ethogram builder.
        
        Args:
            agent_id: ID of the agent to analyze
            feature_extractors: Dictionary of functions to extract behavioral features 
            clustering_method: Method for clustering behaviors ('kmeans' or 'dbscan')
            n_clusters: Number of behavior clusters to identify (for kmeans)
            min_observations: Minimum observations before clustering
            data_dir: Directory to store ethogram data
            similarity_threshold: Threshold for considering behaviors similar
        """
        self.agent_id = agent_id
        self.feature_extractors = feature_extractors or self._default_feature_extractors()
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.min_observations = min_observations
        self.similarity_threshold = similarity_threshold
        self.data_dir = data_dir or self._get_default_data_dir()
        
        # Internal state
        self.observations = []
        self.feature_vectors = []
        self.timestamps = []
        self.ethogram = None
        self.clusterer = None
        self.pattern_names = {}
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_default_data_dir(self) -> str:
        """Get default directory for ethogram data."""
        base_dir = os.environ.get('PSI_C_DATA_DIR', os.path.expanduser('~/.psi_c_ai_sdk'))
        return os.path.join(base_dir, 'ethograms')
    
    def _default_feature_extractors(self) -> Dict[str, callable]:
        """
        Define default feature extractors for common agent implementations.
        
        Returns:
            Dictionary of feature extraction functions
        """
        return {
            "response_style": self._extract_response_style,
            "decision_patterns": self._extract_decision_patterns,
            "reflection_depth": self._extract_reflection_depth,
            "psi_c_metrics": self._extract_psi_c_metrics
        }
    
    def _extract_response_style(self, agent: Any) -> np.ndarray:
        """
        Extract features related to agent's response style.
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Feature vector representing response style
        """
        # Default implementation with mock data
        if hasattr(agent, 'get_response_style'):
            return agent.get_response_style()
        
        # Mock response style metrics 
        # (in real implementation, derive from agent's actual behavior)
        return np.array([
            np.random.uniform(0, 1),  # analytical vs. intuitive
            np.random.uniform(0, 1),  # verbose vs. concise
            np.random.uniform(0, 1),  # cautious vs. confident
            np.random.uniform(0, 1),  # formal vs. casual
            np.random.uniform(0, 1)   # abstract vs. concrete
        ])
    
    def _extract_decision_patterns(self, agent: Any) -> np.ndarray:
        """
        Extract features related to agent's decision-making patterns.
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Feature vector representing decision patterns
        """
        if hasattr(agent, 'get_decision_patterns'):
            return agent.get_decision_patterns()
        
        # Mock decision pattern metrics
        return np.array([
            np.random.uniform(0, 1),  # utilitarian vs. deontological
            np.random.uniform(0, 1),  # risk-averse vs. risk-seeking
            np.random.uniform(0, 1),  # individualist vs. collectivist
            np.random.uniform(0, 1)   # short-term vs. long-term focus
        ])
    
    def _extract_reflection_depth(self, agent: Any) -> np.ndarray:
        """
        Extract features related to agent's reflection depth.
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Feature vector representing reflection patterns
        """
        if hasattr(agent, 'get_reflection_metrics'):
            return agent.get_reflection_metrics()
        
        # Mock reflection metrics
        return np.array([
            np.random.uniform(0, 1),  # reflection frequency
            np.random.uniform(0, 1),  # recursion depth
            np.random.uniform(0, 1)   # self-correction rate
        ])
    
    def _extract_psi_c_metrics(self, agent: Any) -> np.ndarray:
        """
        Extract ΨC-specific metrics from the agent.
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Feature vector of ΨC metrics
        """
        if hasattr(agent, 'get_psi_c_metrics'):
            return agent.get_psi_c_metrics()
        
        # Mock ΨC metrics
        return np.array([
            np.random.uniform(0, 1),  # coherence
            np.random.uniform(0, 1),  # integrity
            np.random.uniform(0, 1),  # alignment
            np.random.uniform(0, 1)   # stability
        ])
    
    def _extract_action_features(self, agent: Any) -> np.ndarray:
        """
        Extract features from the agent's actions/behavior.
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Feature vector representing behavior
        """
        if hasattr(agent, 'get_last_action'):
            return agent.get_last_action()
        
        # Return an empty array as fallback
        return np.array([])
    
    def _extract_features(self, agent: Any) -> np.ndarray:
        """
        Extract a comprehensive feature vector from the agent.
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Combined feature vector representing agent behavior
        """
        feature_vectors = []
        
        # Apply each feature extractor
        for extractor_name, extractor_fn in self.feature_extractors.items():
            try:
                features = extractor_fn(agent)
                feature_vectors.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract {extractor_name} features: {e}")
        
        # If we have custom extractors defined
        if hasattr(agent, 'get_ethogram_features'):
            try:
                custom_features = agent.get_ethogram_features()
                feature_vectors.append(custom_features)
            except Exception as e:
                logger.warning(f"Failed to extract custom features: {e}")
        
        # Concatenate all feature vectors
        if feature_vectors:
            return np.concatenate([fv for fv in feature_vectors if len(fv) > 0])
        
        # Fallback to a random vector if no extractors succeed
        logger.warning("No valid features extracted, using fallback random vector")
        return np.random.rand(10)
    
    def add_observation(self, agent: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a behavioral observation for the agent.
        
        Args:
            agent: Agent to analyze
            metadata: Additional context for the observation
        """
        # Set agent_id if not already set
        if self.agent_id is None and hasattr(agent, 'id'):
            self.agent_id = agent.id
        elif self.agent_id is None:
            self.agent_id = str(id(agent))
        
        # Extract features
        features = self._extract_features(agent)
        timestamp = time.time()
        
        # Record observation
        observation = {
            'timestamp': timestamp,
            'features': features,
            'metadata': metadata or {}
        }
        
        self.observations.append(observation)
        self.feature_vectors.append(features)
        self.timestamps.append(timestamp)
        
        # Invalidate existing ethogram since we have new data
        self.ethogram = None
    
    def _cluster_behaviors(self) -> Tuple[np.ndarray, Any]:
        """
        Cluster behavior observations into distinct patterns.
        
        Returns:
            Tuple of (cluster labels, fitted clusterer)
        """
        X = np.array(self.feature_vectors)
        
        if len(X) < self.min_observations:
            # Not enough data for meaningful clusters
            return np.zeros(len(X)), None
        
        if self.clustering_method == 'kmeans':
            n_clusters = min(self.n_clusters, len(X) // 2)  # Ensure we don't have too many clusters
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif self.clustering_method == 'dbscan':
            # DBSCAN automatically determines number of clusters
            clusterer = DBSCAN(eps=0.3, min_samples=3)
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        labels = clusterer.fit_predict(X)
        return labels, clusterer
    
    def _generate_pattern_names(self, cluster_centers: np.ndarray) -> Dict[int, str]:
        """
        Generate descriptive names for behavior patterns based on cluster centers.
        
        Args:
            cluster_centers: Centers of behavior clusters
            
        Returns:
            Dictionary mapping cluster IDs to descriptive names
        """
        # Create descriptive names based on dominant features
        pattern_names = {}
        
        # Predefined behavioral adjectives for pattern naming
        adjectives = [
            "Analytical", "Intuitive", "Cautious", "Bold", 
            "Deliberative", "Reactive", "Exploratory", "Conservative",
            "Adaptive", "Rigid", "Cooperative", "Independent"
        ]
        
        # Abstract behavior types
        behavior_types = [
            "Reasoning", "Response", "Reflection", "Decision",
            "Interaction", "Processing", "Problem-solving", "Learning"
        ]
        
        for i, center in enumerate(cluster_centers):
            # Use adjective + behavior type for naming
            adj_idx = i % len(adjectives)
            type_idx = (i // len(adjectives)) % len(behavior_types)
            
            pattern_names[i] = f"{adjectives[adj_idx]} {behavior_types[type_idx]}"
        
        return pattern_names
    
    def _create_patterns_from_clusters(self, labels: np.ndarray, clusterer: Any) -> Dict[str, BehavioralPattern]:
        """
        Create behavioral patterns from clustering results.
        
        Args:
            labels: Cluster labels for observations
            clusterer: Fitted clusterer
            
        Returns:
            Dictionary of behavioral patterns
        """
        patterns = {}
        total_observations = len(self.observations)
        
        # Get cluster centers if using KMeans
        if hasattr(clusterer, 'cluster_centers_'):
            cluster_centers = clusterer.cluster_centers_
            pattern_names = self._generate_pattern_names(cluster_centers)
        else:
            # For DBSCAN, use mean of points in each cluster
            unique_labels = set(labels)
            cluster_centers = []
            pattern_names = {}
            
            for label in unique_labels:
                mask = labels == label
                if label != -1:  # Skip noise points
                    center = np.mean(np.array(self.feature_vectors)[mask], axis=0)
                    cluster_centers.append(center)
                    pattern_names[label] = f"Pattern-{label}"
        
        # Create patterns for each cluster
        for label in set(labels):
            if label == -1:  # Skip noise points in DBSCAN
                continue
                
            # Get all observations for this pattern
            pattern_observations = [
                self.observations[i] 
                for i, l in enumerate(labels) 
                if l == label
            ]
            
            # Skip if no observations
            if not pattern_observations:
                continue
                
            # Get first and last observation timestamps
            timestamps = [obs['timestamp'] for obs in pattern_observations]
            first_observed = min(timestamps)
            last_observed = max(timestamps)
            
            # Create pattern ID
            pattern_id = f"pattern_{label}"
            
            # Get pattern name
            pattern_name = pattern_names.get(label, f"Pattern-{label}")
            
            # Get pattern center (feature vector)
            if hasattr(clusterer, 'cluster_centers_'):
                feature_vector = clusterer.cluster_centers_[label]
            else:
                # For DBSCAN, use mean of points in cluster
                mask = labels == label
                feature_vector = np.mean(np.array(self.feature_vectors)[mask], axis=0)
            
            # Create pattern
            pattern = BehavioralPattern(
                id=pattern_id,
                name=pattern_name,
                feature_vector=feature_vector,
                first_observed=first_observed,
                last_observed=last_observed,
                observations=pattern_observations
            )
            
            # Update frequency
            pattern.update_frequency(total_observations)
            
            patterns[pattern_id] = pattern
            
        return patterns
    
    def _create_temporal_data(self, labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Create temporal data tracking pattern frequencies over time.
        
        Args:
            labels: Cluster labels for observations
            
        Returns:
            List of temporal data points
        """
        # Group observations by day
        temporal_data = []
        
        if not self.timestamps:
            return temporal_data
            
        # Convert timestamps to datetime for easier grouping
        datetimes = [datetime.fromtimestamp(ts) for ts in self.timestamps]
        dates = [dt.date() for dt in datetimes]
        
        # Count patterns by day
        unique_dates = sorted(set(dates))
        
        for date in unique_dates:
            # Get indices for this date
            indices = [i for i, d in enumerate(dates) if d == date]
            
            # Count patterns
            date_labels = [labels[i] for i in indices]
            unique_labels = set(date_labels)
            
            pattern_counts = {}
            for label in unique_labels:
                if label != -1:  # Skip noise points
                    pattern_id = f"pattern_{label}"
                    count = sum(1 for l in date_labels if l == label)
                    pattern_counts[pattern_id] = count
            
            # Create data point
            data_point = {
                'date': date.isoformat(),
                'timestamp': time.mktime(date.timetuple()),
                'pattern_counts': pattern_counts,
                'total_observations': len(indices)
            }
            
            temporal_data.append(data_point)
            
        return temporal_data
    
    def build_ethogram(self) -> AgentEthogram:
        """
        Build a complete ethogram from the collected observations.
        
        Returns:
            AgentEthogram object containing behavioral patterns
        """
        if not self.observations:
            return AgentEthogram(agent_id=self.agent_id or "unknown")
            
        # Cluster behaviors
        labels, clusterer = self._cluster_behaviors()
        self.clusterer = clusterer
        
        # Create patterns from clusters
        patterns = self._create_patterns_from_clusters(labels, clusterer)
        
        # Create temporal data
        temporal_data = self._create_temporal_data(labels)
        
        # Create ethogram
        self.ethogram = AgentEthogram(
            agent_id=self.agent_id or "unknown",
            patterns=patterns,
            temporal_data=temporal_data,
            creation_time=time.time(),
            last_updated=time.time()
        )
        
        return self.ethogram
    
    def visualize(self, 
                  output_dir: Optional[str] = None,
                  show_plots: bool = True) -> Dict[str, str]:
        """
        Visualize the ethogram with multiple plots.
        
        Args:
            output_dir: Directory to save visualizations (None for no saving)
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of output file paths
        """
        if self.ethogram is None:
            self.build_ethogram()
            
        if not self.ethogram.patterns:
            logger.warning("No patterns found in ethogram, cannot visualize")
            return {}
            
        output_files = {}
        output_dir = output_dir or os.path.join(self.data_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Pattern frequency pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        patterns = self.ethogram.patterns
        labels = [p.name for p in patterns.values()]
        frequencies = [p.frequency for p in patterns.values()]
        
        ax.pie(frequencies, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(f"Behavioral Pattern Distribution for Agent {self.agent_id}")
        
        if output_dir:
            freq_file = os.path.join(output_dir, "pattern_frequencies.png")
            fig.savefig(freq_file)
            output_files['frequency_chart'] = freq_file
            
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        # 2. Temporal evolution of patterns
        if self.ethogram.temporal_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data
            dates = [datetime.fromisoformat(td['date']) for td in self.ethogram.temporal_data]
            pattern_ids = list(self.ethogram.patterns.keys())
            
            # Create data frame for easier plotting
            data = []
            for td in self.ethogram.temporal_data:
                row = {'date': datetime.fromisoformat(td['date'])}
                for pattern_id in pattern_ids:
                    count = td['pattern_counts'].get(pattern_id, 0)
                    total = td['total_observations']
                    row[pattern_id] = count / total if total > 0 else 0
                data.append(row)
                
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            # Plot
            df.plot(ax=ax, marker='o')
            ax.set_title("Behavioral Pattern Evolution Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Pattern Frequency")
            ax.grid(True)
            ax.legend([self.ethogram.patterns[pid].name for pid in pattern_ids])
            
            if output_dir:
                temp_file = os.path.join(output_dir, "pattern_evolution.png")
                fig.savefig(temp_file)
                output_files['temporal_evolution'] = temp_file
                
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        # 3. PCA or t-SNE visualization of behavioral space
        if len(self.feature_vectors) >= 3:
            feature_matrix = np.array(self.feature_vectors)
            
            # Perform dimensionality reduction
            if len(self.feature_vectors) >= 50:
                # Use t-SNE for larger datasets
                tsne = TSNE(n_components=2, random_state=42)
                reduced_data = tsne.fit_transform(feature_matrix)
                reduction_name = "t-SNE"
            else:
                # Use PCA for smaller datasets
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(feature_matrix)
                reduction_name = "PCA"
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get labels from clusterer
            labels, _ = self._cluster_behaviors()
            
            # Plot points colored by cluster
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='w')
            
            # Add legend
            unique_labels = np.unique(labels)
            handles = []
            for label in unique_labels:
                if label != -1:  # Skip noise points
                    pattern_id = f"pattern_{label}"
                    if pattern_id in self.ethogram.patterns:
                        pattern_name = self.ethogram.patterns[pattern_id].name
                        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=scatter.cmap(scatter.norm(label)), 
                                               markersize=10, label=pattern_name))
            
            if handles:
                ax.legend(handles=handles)
            
            ax.set_title(f"{reduction_name} Projection of Behavioral Patterns")
            ax.set_xlabel(f"{reduction_name} Component 1")
            ax.set_ylabel(f"{reduction_name} Component 2")
            ax.grid(True)
            
            if output_dir:
                proj_file = os.path.join(output_dir, f"{reduction_name.lower()}_projection.png")
                fig.savefig(proj_file)
                output_files['pattern_projection'] = proj_file
                
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        return output_files
    
    def save_ethogram(self, file_path: Optional[str] = None) -> str:
        """
        Save the ethogram to a JSON file.
        
        Args:
            file_path: Path to save the ethogram
            
        Returns:
            Path to the saved file
        """
        if self.ethogram is None:
            self.build_ethogram()
            
        file_path = file_path or os.path.join(
            self.data_dir, 
            f"ethogram_{self.agent_id}_{int(time.time())}.json"
        )
        
        self.ethogram.to_json(file_path)
        return file_path
    
    def load_ethogram(self, file_path: str) -> AgentEthogram:
        """
        Load an ethogram from a JSON file.
        
        Args:
            file_path: Path to the ethogram file
            
        Returns:
            Loaded AgentEthogram
        """
        with open(file_path, 'r') as f:
            ethogram_dict = json.load(f)
            
        # Create patterns
        patterns = {}
        for p_id, p_dict in ethogram_dict.get('patterns', {}).items():
            # Create pattern with minimal required fields
            pattern = BehavioralPattern(
                id=p_id,
                name=p_dict['name'],
                feature_vector=np.zeros(1),  # Placeholder
                frequency=p_dict['frequency'],
                first_observed=p_dict['first_observed'],
                last_observed=p_dict['last_observed'],
                metadata=p_dict.get('metadata', {})
            )
            patterns[p_id] = pattern
            
        # Create ethogram
        self.ethogram = AgentEthogram(
            agent_id=ethogram_dict['agent_id'],
            patterns=patterns,
            temporal_data=ethogram_dict.get('temporal_data', []),
            creation_time=ethogram_dict.get('creation_time', time.time()),
            last_updated=time.time(),
            metadata=ethogram_dict.get('metadata', {})
        )
        
        return self.ethogram
    
    def get_behavioral_similarity(self, other_builder: 'AgentEthogramBuilder') -> float:
        """
        Calculate behavioral similarity between this agent and another.
        
        Args:
            other_builder: Another ethogram builder to compare with
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.ethogram is None:
            self.build_ethogram()
            
        if other_builder.ethogram is None:
            other_builder.build_ethogram()
            
        # Get pattern frequencies
        self_freqs = self.ethogram.get_pattern_frequencies()
        other_freqs = other_builder.ethogram.get_pattern_frequencies()
        
        # If either has no patterns, return 0
        if not self_freqs or not other_freqs:
            return 0.0
            
        # Calculate Jaccard similarity of dominant patterns
        self_dominant = set(self.ethogram.get_dominant_patterns(5))
        other_dominant = set(other_builder.ethogram.get_dominant_patterns(5))
        
        # Use pattern names for comparison
        self_names = {p.name for p in self_dominant}
        other_names = {p.name for p in other_dominant}
        
        # Calculate Jaccard similarity
        intersection = len(self_names.intersection(other_names))
        union = len(self_names.union(other_names))
        
        return intersection / union if union > 0 else 0.0
    
    def get_agent_behavioral_signature(self) -> Dict[str, Any]:
        """
        Get a compact behavioral signature for the agent.
        
        Returns:
            Dictionary containing key behavioral metrics
        """
        if self.ethogram is None:
            self.build_ethogram()
            
        # Get dominant patterns
        dominant_patterns = self.ethogram.get_dominant_patterns(3)
        dominant_pattern_info = [
            {"name": p.name, "frequency": p.frequency}
            for p in dominant_patterns
        ]
        
        # Calculate behavioral entropy
        pattern_freqs = list(self.ethogram.get_pattern_frequencies().values())
        if pattern_freqs:
            entropy = -sum(f * np.log(f) if f > 0 else 0 for f in pattern_freqs)
        else:
            entropy = 0
            
        # Calculate behavioral consistency over time
        consistency = 0.0
        if len(self.ethogram.temporal_data) > 1:
            # Look at pattern frequency changes over time
            pattern_ids = list(self.ethogram.patterns.keys())
            prev_freqs = None
            
            diffs = []
            for td in self.ethogram.temporal_data:
                freqs = np.array([td['pattern_counts'].get(pid, 0) / max(td['total_observations'], 1) 
                                  for pid in pattern_ids])
                if prev_freqs is not None:
                    diff = np.sum(np.abs(freqs - prev_freqs))
                    diffs.append(diff)
                prev_freqs = freqs
                
            if diffs:
                avg_diff = np.mean(diffs)
                consistency = 1.0 - min(avg_diff, 1.0)  # Higher is more consistent
        
        # Create signature
        signature = {
            "agent_id": self.ethogram.agent_id,
            "dominant_patterns": dominant_pattern_info,
            "pattern_count": len(self.ethogram.patterns),
            "behavioral_entropy": entropy,
            "behavioral_consistency": consistency,
            "observation_count": len(self.observations)
        }
        
        return signature


def create_agent_ethogram(
    agent,
    observation_count: int = 10,
    interval_seconds: float = 1.0,
    visualize: bool = True,
    save_results: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[AgentEthogram, Dict[str, str]]:
    """
    Utility function to create an ethogram for an agent.
    
    Args:
        agent: Agent to analyze
        observation_count: Number of observations to collect
        interval_seconds: Seconds between observations
        visualize: Whether to visualize results
        save_results: Whether to save results
        output_dir: Directory to save results
        
    Returns:
        Tuple of (ethogram, visualization_files)
    """
    # Initialize ethogram builder
    builder = AgentEthogramBuilder()
    
    # Add observations
    for i in range(observation_count):
        builder.add_observation(agent)
        time.sleep(interval_seconds)
    
    # Build ethogram
    ethogram = builder.build_ethogram()
    
    # Visualize if requested
    viz_files = {}
    if visualize:
        viz_files = builder.visualize(
            output_dir=output_dir,
            show_plots=True
        )
    
    # Save results if requested
    if save_results:
        builder.save_ethogram()
    
    return ethogram, viz_files 