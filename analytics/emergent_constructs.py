#!/usr/bin/env python3
"""
Novelty Map of Emergent Constructs

This module tracks and visualizes unexpected concept clusters that emerge during an agent's
reflective processes, providing insights into novel constructs that were not explicitly
programmed (unlike traditional cognitive architectures like SOAR).

Key features:
- Tracking concept drift in schema vector space over time
- Detecting emergent clusters that exceed threshold distance from seed concepts
- Visualizing the evolution and emergence of novel constructs
- Analyzing the relationship between reflection and emergent concept formation

Detection is based on measuring the distance between current concept clusters and 
their seed concepts: 
    Δ_construct = || C_emergent - C_seed ||

Constructs are flagged as novel when they exceed threshold distance from their origin.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import os
import sys
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure we can import from the psi_c_ai_sdk package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from psi_c_ai_sdk.schema import schema_manager
    from psi_c_ai_sdk.belief import belief_network
    MOCK_MODE = False
except ImportError:
    logger.warning("Could not import ΨC SDK modules. Running in mock mode with simulated data.")
    MOCK_MODE = True


@dataclass
class Concept:
    """Represents a concept in the schema space."""
    id: str
    name: str
    vector: np.ndarray
    creation_time: float = field(default_factory=time.time)
    is_seed: bool = False
    related_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other_concept: 'Concept') -> float:
        """Calculate the distance to another concept."""
        return np.linalg.norm(self.vector - other_concept.vector)


@dataclass
class ConceptCluster:
    """Represents a cluster of related concepts."""
    id: str
    concepts: List[str]  # Concept IDs
    centroid: np.ndarray
    creation_time: float = field(default_factory=time.time)
    ancestor_clusters: List[str] = field(default_factory=list)  # Parent cluster IDs
    novelty_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_novelty(self, seed_centroids: List[np.ndarray]) -> float:
        """
        Calculate novelty score based on distance from seed centroids.
        
        Args:
            seed_centroids: List of seed concept centroids
            
        Returns:
            Novelty score (0-1, higher = more novel)
        """
        if not seed_centroids:
            return 0.0
        
        # Calculate minimum distance to any seed centroid
        min_dist = min(np.linalg.norm(self.centroid - seed) for seed in seed_centroids)
        
        # Normalize to a 0-1 score using a sigmoid function
        novelty = 1.0 / (1.0 + np.exp(-0.5 * (min_dist - 2.0)))
        
        self.novelty_score = novelty
        return novelty


@dataclass
class EmergenceEvent:
    """Represents the emergence of a novel construct."""
    id: str
    timestamp: float
    cluster_id: str
    novelty_score: float
    description: str
    concepts_involved: List[str]
    reflective_trigger: Optional[str] = None  # ID of the reflection that triggered this
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "cluster_id": self.cluster_id,
            "novelty_score": self.novelty_score,
            "description": self.description,
            "concepts_involved": self.concepts_involved,
            "reflective_trigger": self.reflective_trigger
        }


class NoveltyMap:
    """
    Main class for tracking and visualizing emergent concept constructs.
    
    This class monitors concept clusters as they evolve, detecting when they
    drift significantly from their original seed concepts to form novel
    emergent constructs.
    """
    
    def __init__(self, 
                 schema_manager=None, 
                 vector_dim: int = 128,
                 novelty_threshold: float = 0.7):
        """
        Initialize the novelty map.
        
        Args:
            schema_manager: Optional schema manager to integrate with
            vector_dim: Dimensionality of concept vectors
            novelty_threshold: Threshold for detecting novel constructs (0-1)
        """
        self.schema_manager = schema_manager
        self.vector_dim = vector_dim
        self.novelty_threshold = novelty_threshold
        
        # Storage
        self.concepts: Dict[str, Concept] = {}
        self.clusters: Dict[str, ConceptCluster] = {}
        self.emergence_events: List[EmergenceEvent] = []
        
        # Seed tracking
        self.seed_concepts: List[str] = []  # IDs of seed concepts
        self.seed_centroids: List[np.ndarray] = []  # Centroids of seed clusters
        
        # Temporal tracking
        self.cluster_history: Dict[str, List[Tuple[float, np.ndarray]]] = {}  # cluster_id -> [(time, centroid)]
        self.concept_count_history: List[Tuple[float, int]] = []  # (time, count)
        self.novelty_history: List[Tuple[float, float]] = []  # (time, avg_novelty)
        
        # Visualization
        self.fig = None
        self.axes = None
        
        # Check if we're in mock mode
        if MOCK_MODE or schema_manager is None:
            logger.info("Initializing with mock data...")
            self._create_mock_data()
    
    def _create_mock_data(self, seed_count: int = 4, concept_count: int = 40):
        """Create mock data for demonstration."""
        # Create seed concepts in different regions of vector space
        seed_regions = [
            {"name": "logic", "center": np.random.normal(5, 0.2, self.vector_dim)},
            {"name": "ethics", "center": np.random.normal(-5, 0.2, self.vector_dim)},
            {"name": "identity", "center": np.random.normal(0, 0.2, self.vector_dim) + np.random.normal(5, 0.2, self.vector_dim)},
            {"name": "causality", "center": np.random.normal(0, 0.2, self.vector_dim) - np.random.normal(5, 0.2, self.vector_dim)}
        ]
        
        # Create seed concepts
        for i, region in enumerate(seed_regions[:seed_count]):
            seed_id = f"seed_{i}"
            
            # Create a normalized vector
            vector = region["center"].copy()
            vector = vector / np.linalg.norm(vector)
            
            # Create concept
            concept = Concept(
                id=seed_id,
                name=f"Seed Concept: {region['name']}",
                vector=vector,
                is_seed=True,
                metadata={"type": "seed", "theme": region["name"]}
            )
            
            self.concepts[seed_id] = concept
            self.seed_concepts.append(seed_id)
            self.seed_centroids.append(vector)
            
            # Create a cluster for each seed
            cluster_id = f"cluster_{i}"
            cluster = ConceptCluster(
                id=cluster_id,
                concepts=[seed_id],
                centroid=vector,
                metadata={"type": "seed_cluster", "theme": region["name"]}
            )
            
            self.clusters[cluster_id] = cluster
            self.cluster_history[cluster_id] = [(time.time(), vector)]
        
        # Create non-seed concepts
        for i in range(concept_count):
            concept_id = f"concept_{i}"
            
            # Pick a random seed to be "near"
            seed_idx = np.random.randint(len(self.seed_concepts))
            seed_concept = self.concepts[self.seed_concepts[seed_idx]]
            
            # Create a vector near the seed, with some noise
            # Higher noise for concepts created later (representing drift)
            noise_level = 0.2 + (i / concept_count) * 1.0  # Gradually increase noise
            vector = seed_concept.vector + np.random.normal(0, noise_level, self.vector_dim)
            vector = vector / np.linalg.norm(vector)
            
            # Create concept
            concept = Concept(
                id=concept_id,
                name=f"Concept near {seed_concept.metadata.get('theme', 'unknown')} #{i}",
                vector=vector,
                related_concepts=[seed_concept.id],
                metadata={"type": "derived", "parent_theme": seed_concept.metadata.get("theme", "unknown")}
            )
            
            self.concepts[concept_id] = concept
        
        # Run clustering to detect emergent clusters
        self._update_clusters()
        
        # Record initial metrics
        self._update_history()
        
        logger.info(f"Created {len(self.seed_concepts)} seed concepts and {len(self.concepts) - len(self.seed_concepts)} derived concepts")
        logger.info(f"Detected {len(self.clusters)} concept clusters")
    
    def _update_clusters(self):
        """Update concept clusters based on current concepts."""
        if len(self.concepts) < 3:
            logger.warning("Not enough concepts to form clusters.")
            return
        
        # Extract vectors and IDs
        vectors = []
        concept_ids = []
        
        for concept_id, concept in self.concepts.items():
            vectors.append(concept.vector)
            concept_ids.append(concept_id)
        
        vectors = np.array(vectors)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.8, min_samples=2)
        labels = clustering.fit_predict(vectors)
        
        # Create clusters
        new_clusters = {}
        cluster_concepts = {}
        
        for i, label in enumerate(labels):
            if label == -1:  # Noise points
                continue
                
            if label not in cluster_concepts:
                cluster_concepts[label] = []
            
            cluster_concepts[label].append(concept_ids[i])
        
        # Create cluster objects
        old_clusters = self.clusters.copy()
        self.clusters = {}
        
        for label, concepts in cluster_concepts.items():
            # Calculate centroid
            centroid = np.mean([self.concepts[cid].vector for cid in concepts], axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            
            # Determine if this is a continuation of an existing cluster
            best_match = None
            best_match_distance = float('inf')
            
            for old_id, old_cluster in old_clusters.items():
                dist = np.linalg.norm(centroid - old_cluster.centroid)
                if dist < best_match_distance:
                    best_match_distance = dist
                    best_match = old_id
            
            # Create or update cluster
            if best_match and best_match_distance < 0.7:
                # Continue existing cluster
                cluster_id = best_match
                ancestor_clusters = old_clusters[best_match].ancestor_clusters
            else:
                # New cluster
                cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
                ancestor_clusters = []
                if best_match:
                    ancestor_clusters = [best_match]
            
            cluster = ConceptCluster(
                id=cluster_id,
                concepts=concepts,
                centroid=centroid,
                ancestor_clusters=ancestor_clusters,
                creation_time=time.time()
            )
            
            # Calculate novelty
            cluster.calculate_novelty(self.seed_centroids)
            
            # Check if this is an emergence event
            if cluster.novelty_score >= self.novelty_threshold:
                # This is a novel construct
                if cluster_id not in self.clusters and not any(e.cluster_id == cluster_id for e in self.emergence_events):
                    # New emergence
                    event = EmergenceEvent(
                        id=f"event_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        cluster_id=cluster_id,
                        novelty_score=cluster.novelty_score,
                        description=f"Novel concept cluster emerged with novelty score {cluster.novelty_score:.2f}",
                        concepts_involved=concepts
                    )
                    
                    self.emergence_events.append(event)
                    logger.info(f"Detected novel construct: {event.description}")
            
            self.clusters[cluster_id] = cluster
            
            # Update cluster history
            if cluster_id not in self.cluster_history:
                self.cluster_history[cluster_id] = []
            
            self.cluster_history[cluster_id].append((time.time(), centroid))
    
    def _update_history(self):
        """Update historical tracking metrics."""
        current_time = time.time()
        
        # Update concept count
        self.concept_count_history.append((current_time, len(self.concepts)))
        
        # Update average novelty
        if self.clusters:
            avg_novelty = np.mean([c.novelty_score for c in self.clusters.values()])
            self.novelty_history.append((current_time, avg_novelty))
    
    def add_concept(self, name: str, vector: np.ndarray, is_seed: bool = False, 
                   related_concepts: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Add a new concept to the novelty map.
        
        Args:
            name: Name of the concept
            vector: Vector representation
            is_seed: Whether this is a seed concept
            related_concepts: IDs of related concepts
            metadata: Additional metadata
            
        Returns:
            ID of the added concept
        """
        # Normalize vector
        vector = vector / np.linalg.norm(vector)
        
        # Generate ID
        concept_id = f"concept_{uuid.uuid4().hex[:8]}"
        
        # Create concept
        concept = Concept(
            id=concept_id,
            name=name,
            vector=vector,
            is_seed=is_seed,
            related_concepts=related_concepts or [],
            metadata=metadata or {}
        )
        
        # Add to storage
        self.concepts[concept_id] = concept
        
        if is_seed:
            self.seed_concepts.append(concept_id)
            self.seed_centroids.append(vector)
        
        # Update clusters
        self._update_clusters()
        
        # Update history
        self._update_history()
        
        return concept_id 

    def visualize_novelty_map(self, figsize=(14, 10), output_file=None):
        """
        Visualize the novelty map showing concept clusters and emergent constructs.
        
        Args:
            figsize: Figure size in inches
            output_file: If provided, save visualization to this file
        """
        if not self.concepts:
            logger.warning("No concepts to visualize.")
            return
        
        # Create figure if needed
        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Clear axes
        for ax in self.axes.flatten():
            ax.clear()
        
        # Extract vectors for visualization
        vectors = np.array([concept.vector for concept in self.concepts.values()])
        
        # Project to 2D using PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(vectors)
        
        # Plot 1: Concept map with clusters
        ax1 = self.axes[0, 0]
        
        # Collect points by cluster
        cluster_points = {}
        unclustered_points = []
        unclustered_ids = []
        
        for i, (concept_id, concept) in enumerate(self.concepts.items()):
            # Find which cluster this concept belongs to
            found_cluster = False
            for cluster_id, cluster in self.clusters.items():
                if concept_id in cluster.concepts:
                    if cluster_id not in cluster_points:
                        cluster_points[cluster_id] = {"x": [], "y": [], "ids": []}
                    
                    cluster_points[cluster_id]["x"].append(pca_result[i, 0])
                    cluster_points[cluster_id]["y"].append(pca_result[i, 1])
                    cluster_points[cluster_id]["ids"].append(concept_id)
                    found_cluster = True
                    break
            
            if not found_cluster:
                unclustered_points.append((pca_result[i, 0], pca_result[i, 1]))
                unclustered_ids.append(concept_id)
        
        # Plot clusters
        for cluster_id, points in cluster_points.items():
            cluster = self.clusters[cluster_id]
            color = plt.cm.viridis(cluster.novelty_score)
            
            ax1.scatter(points["x"], points["y"], color=color, alpha=0.7, 
                      label=f"Cluster {cluster_id[:4]}... (novelty: {cluster.novelty_score:.2f})")
            
            # Plot centroid
            centroid_pca = pca.transform([cluster.centroid])[0]
            ax1.scatter(centroid_pca[0], centroid_pca[1], color=color, marker='*', s=200, 
                      edgecolor='black', linewidth=1)
            
            # Mark seed concepts
            for i, cid in enumerate(points["ids"]):
                if cid in self.seed_concepts:
                    ax1.scatter(points["x"][i], points["y"][i], color='red', marker='x', s=100)
        
        # Plot unclustered points
        if unclustered_points:
            x, y = zip(*unclustered_points)
            ax1.scatter(x, y, color='gray', alpha=0.5, marker='o', label="Unclustered")
        
        # Add legend if not too crowded
        if len(cluster_points) <= 5:
            ax1.legend(loc='upper right')
        
        ax1.set_title("Concept Clusters in Vector Space")
        ax1.set_xlabel("PCA Dimension 1")
        ax1.set_ylabel("PCA Dimension 2")
        
        # Plot 2: Novelty scores by cluster
        ax2 = self.axes[0, 1]
        
        if self.clusters:
            cluster_ids = list(self.clusters.keys())
            novelty_scores = [self.clusters[cid].novelty_score for cid in cluster_ids]
            
            # Sort by novelty
            sorted_indices = np.argsort(novelty_scores)[::-1]
            sorted_ids = [cluster_ids[i] for i in sorted_indices]
            sorted_scores = [novelty_scores[i] for i in sorted_indices]
            
            # Use shortened IDs for clarity
            short_ids = [cid[:6] + "..." for cid in sorted_ids]
            
            # Plot bars
            bars = ax2.bar(short_ids, sorted_scores, color=plt.cm.viridis(sorted_scores))
            
            # Add novelty threshold line
            ax2.axhline(y=self.novelty_threshold, color='red', linestyle='--', 
                       alpha=0.7, label="Novelty Threshold")
            
            # Highlight emergent clusters
            for i, score in enumerate(sorted_scores):
                if score >= self.novelty_threshold:
                    bars[i].set_edgecolor('red')
                    bars[i].set_linewidth(2)
            
            ax2.set_xlabel("Cluster ID")
            ax2.set_ylabel("Novelty Score")
            ax2.set_title("Cluster Novelty Scores")
            ax2.set_ylim(0, 1)
            ax2.legend()
            
            # Rotate x-tick labels if many clusters
            if len(sorted_ids) > 5:
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, "No clusters available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Concept count and novelty over time
        ax3 = self.axes[1, 0]
        
        if self.concept_count_history and self.novelty_history:
            # Setup two y-axes
            ax3_twin = ax3.twinx()
            
            # Convert to relative time
            start_time = min(self.concept_count_history[0][0], self.novelty_history[0][0])
            concept_times = [(t - start_time) / 60 for t, _ in self.concept_count_history]  # minutes
            novelty_times = [(t - start_time) / 60 for t, _ in self.novelty_history]  # minutes
            
            concept_counts = [c for _, c in self.concept_count_history]
            novelty_values = [n for _, n in self.novelty_history]
            
            # Plot concept count
            line1 = ax3.plot(concept_times, concept_counts, 'b-', marker='o', label="Concept Count")
            ax3.set_xlabel("Time (minutes)")
            ax3.set_ylabel("Concept Count", color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            
            # Plot average novelty
            line2 = ax3_twin.plot(novelty_times, novelty_values, 'r-', marker='x', label="Avg Novelty")
            ax3_twin.set_ylabel("Average Novelty", color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
            ax3_twin.set_ylim(0, 1)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left')
            
            ax3.set_title("Concept Growth and Novelty Over Time")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No history data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Emergence events timeline
        ax4 = self.axes[1, 1]
        
        if self.emergence_events:
            # Convert to relative time
            start_time = min(e.timestamp for e in self.emergence_events)
            event_times = [(e.timestamp - start_time) / 60 for e in self.emergence_events]  # minutes
            novelty_scores = [e.novelty_score for e in self.emergence_events]
            
            # Plot events
            scatter = ax4.scatter(event_times, novelty_scores, c=novelty_scores, 
                                cmap='plasma', s=100, alpha=0.7)
            
            # Add labels for each event
            for i, event in enumerate(self.emergence_events):
                ax4.annotate(f"E{i+1}", 
                           (event_times[i], novelty_scores[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_xlabel("Time (minutes)")
            ax4.set_ylabel("Novelty Score")
            ax4.set_title("Emergence Events Timeline")
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label("Novelty Score")
            
            # Add event legend (shortened)
            legend_text = "\n".join([f"E{i+1}: {len(e.concepts_involved)} concepts, score={e.novelty_score:.2f}" 
                                    for i, e in enumerate(self.emergence_events)])
            ax4.text(1.05, 0.5, legend_text, transform=ax4.transAxes, fontsize=8,
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax4.text(0.5, 0.5, "No emergence events detected", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        
        # Show the figure
        plt.show()
    
    def generate_report(self, output_file=None):
        """
        Generate a report on emergent constructs detected.
        
        Args:
            output_file: If provided, save report to this file
        """
        # Create report text
        report_text = f"""
NOVELTY MAP OF EMERGENT CONSTRUCTS REPORT
=========================================

System Overview:
--------------
Total Concepts: {len(self.concepts)}
Seed Concepts: {len(self.seed_concepts)}
Total Clusters: {len(self.clusters)}
Novelty Threshold: {self.novelty_threshold}

Emergent Constructs Summary:
--------------------------
Total Emergence Events: {len(self.emergence_events)}
Average Novelty Score: {np.mean([c.novelty_score for c in self.clusters.values()]):.4f}
Novel Clusters: {sum(1 for c in self.clusters.values() if c.novelty_score >= self.novelty_threshold)}

Most Novel Clusters:
------------------
"""
        
        # Add details on most novel clusters
        if self.clusters:
            sorted_clusters = sorted(
                self.clusters.values(), 
                key=lambda c: c.novelty_score, 
                reverse=True
            )
            
            # Show top 5 most novel clusters
            for i, cluster in enumerate(sorted_clusters[:5]):
                concepts_in_cluster = [self.concepts[cid].name for cid in cluster.concepts if cid in self.concepts]
                concepts_text = ", ".join(concepts_in_cluster[:3])
                if len(concepts_in_cluster) > 3:
                    concepts_text += f", and {len(concepts_in_cluster) - 3} more"
                
                report_text += f"""
Cluster {i+1}: {cluster.id}
Novelty Score: {cluster.novelty_score:.4f}
Concepts: {len(cluster.concepts)}
Sample Concepts: {concepts_text}
Ancestor Clusters: {', '.join(cluster.ancestor_clusters) if cluster.ancestor_clusters else 'None'}
------------------------------
"""
        
        # Add details on emergence events
        report_text += f"""
\nEmergence Events:
----------------
"""
        
        if self.emergence_events:
            for i, event in enumerate(self.emergence_events):
                concepts_involved = [self.concepts[cid].name for cid in event.concepts_involved if cid in self.concepts]
                concepts_text = ", ".join(concepts_involved[:3])
                if len(concepts_involved) > 3:
                    concepts_text += f", and {len(concepts_involved) - 3} more"
                
                report_text += f"""
Event {i+1}: {event.id}
Timestamp: {datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
Cluster: {event.cluster_id}
Novelty Score: {event.novelty_score:.4f}
Concepts Involved: {len(event.concepts_involved)}
Sample Concepts: {concepts_text}
Description: {event.description}
------------------------------
"""
        else:
            report_text += "No emergence events detected.\n"
        
        # Add recommendations
        report_text += f"""
\nRecommendations:
---------------
"""
        
        if self.emergence_events:
            if len(self.emergence_events) > 2:
                report_text += """
- High level of emergent concept formation detected. Consider investigating the patterns
  of emergence to understand the factors driving novel concept creation.
- The system is showing strong signs of generative creativity. Monitor for potential
  concept drift that might affect alignment.
"""
            else:
                report_text += """
- Limited emergence events detected. Consider enriching the concept space with more
  diverse seed concepts or lowering the novelty threshold.
- Current emergence patterns suggest incremental rather than transformative novelty.
"""
        else:
            report_text += """
- No emergence events detected. Consider:
  1. Lowering the novelty threshold
  2. Adding more diverse seed concepts
  3. Increasing the variability in concept generation
  4. Running the system for longer to allow emergence to develop
"""
        
        # Print to console
        print(report_text)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
    
    def export_data(self, output_file: str):
        """
        Export novelty map data to a JSON file.
        
        Args:
            output_file: File to save data to
        """
        data = {
            "concepts": {},
            "clusters": {},
            "emergence_events": [],
            "seed_concepts": self.seed_concepts,
            "novelty_threshold": self.novelty_threshold,
            "concept_count_history": self.concept_count_history,
            "novelty_history": self.novelty_history
        }
        
        # Export concepts
        for concept_id, concept in self.concepts.items():
            data["concepts"][concept_id] = {
                "id": concept.id,
                "name": concept.name,
                "vector": concept.vector.tolist(),
                "creation_time": concept.creation_time,
                "is_seed": concept.is_seed,
                "related_concepts": concept.related_concepts,
                "metadata": concept.metadata
            }
        
        # Export clusters
        for cluster_id, cluster in self.clusters.items():
            data["clusters"][cluster_id] = {
                "id": cluster.id,
                "concepts": cluster.concepts,
                "centroid": cluster.centroid.tolist(),
                "creation_time": cluster.creation_time,
                "ancestor_clusters": cluster.ancestor_clusters,
                "novelty_score": cluster.novelty_score,
                "metadata": cluster.metadata
            }
        
        # Export emergence events
        for event in self.emergence_events:
            data["emergence_events"].append(event.to_dict())
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data exported to {output_file}")
    
    def import_data(self, input_file: str):
        """
        Import novelty map data from a JSON file.
        
        Args:
            input_file: File to load data from
        """
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Clear existing data
            self.concepts = {}
            self.clusters = {}
            self.emergence_events = []
            self.seed_concepts = []
            self.seed_centroids = []
            self.cluster_history = {}
            self.concept_count_history = []
            self.novelty_history = []
            
            # Import concepts
            for concept_id, concept_data in data["concepts"].items():
                concept = Concept(
                    id=concept_id,
                    name=concept_data["name"],
                    vector=np.array(concept_data["vector"]),
                    creation_time=concept_data["creation_time"],
                    is_seed=concept_data["is_seed"],
                    related_concepts=concept_data["related_concepts"],
                    metadata=concept_data["metadata"]
                )
                
                self.concepts[concept_id] = concept
                
                if concept.is_seed:
                    self.seed_concepts.append(concept_id)
                    self.seed_centroids.append(concept.vector)
            
            # Import clusters
            for cluster_id, cluster_data in data["clusters"].items():
                cluster = ConceptCluster(
                    id=cluster_id,
                    concepts=cluster_data["concepts"],
                    centroid=np.array(cluster_data["centroid"]),
                    creation_time=cluster_data["creation_time"],
                    ancestor_clusters=cluster_data["ancestor_clusters"],
                    novelty_score=cluster_data["novelty_score"],
                    metadata=cluster_data["metadata"]
                )
                
                self.clusters[cluster_id] = cluster
            
            # Import emergence events
            for event_data in data["emergence_events"]:
                event = EmergenceEvent(
                    id=event_data["id"],
                    timestamp=event_data["timestamp"],
                    cluster_id=event_data["cluster_id"],
                    novelty_score=event_data["novelty_score"],
                    description=event_data["description"],
                    concepts_involved=event_data["concepts_involved"],
                    reflective_trigger=event_data.get("reflective_trigger")
                )
                
                self.emergence_events.append(event)
            
            # Import other data
            self.novelty_threshold = data.get("novelty_threshold", self.novelty_threshold)
            self.concept_count_history = data.get("concept_count_history", [])
            self.novelty_history = data.get("novelty_history", [])
            
            logger.info(f"Imported {len(self.concepts)} concepts, {len(self.clusters)} clusters, and {len(self.emergence_events)} emergence events")
            
            return True
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            return False
    
    def generate_dashboard(self, output_dir: str):
        """
        Generate a comprehensive dashboard for novelty analysis.
        
        Args:
            output_dir: Directory to save dashboard files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate visualization
        viz_file = os.path.join(output_dir, "novelty_map.png")
        self.visualize_novelty_map(output_file=viz_file)
        
        # Generate report
        report_file = os.path.join(output_dir, "novelty_report.txt")
        self.generate_report(output_file=report_file)
        
        # Export data
        data_file = os.path.join(output_dir, "novelty_data.json")
        self.export_data(data_file)
        
        # Generate HTML dashboard
        html_file = os.path.join(output_dir, "novelty_dashboard.html")
        
        # Create HTML content
        with open(html_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Novelty Map of Emergent Constructs</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #3f51b5; color: white; padding: 20px; margin-bottom: 20px; }}
        .viz-container {{ margin-bottom: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; margin-right: 15px; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .cluster-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .cluster-table th, .cluster-table td {{ padding: 10px; border: 1px solid #ddd; }}
        .cluster-table th {{ background-color: #f2f2f2; }}
        .event-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .event-table th, .event-table td {{ padding: 10px; border: 1px solid #ddd; }}
        .event-table th {{ background-color: #f2f2f2; }}
        .timestamp {{ font-size: 12px; color: #666; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Novelty Map of Emergent Constructs</h1>
            <p>Tracking and visualizing unexpected concept clusters that emerge from reflection</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{len(self.concepts)}</div>
                <div class="metric-label">Total Concepts</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(self.clusters)}</div>
                <div class="metric-label">Concept Clusters</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sum(1 for c in self.clusters.values() if c.novelty_score >= self.novelty_threshold)}</div>
                <div class="metric-label">Novel Clusters</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(self.emergence_events)}</div>
                <div class="metric-label">Emergence Events</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{np.mean([c.novelty_score for c in self.clusters.values()]):.2f}</div>
                <div class="metric-label">Average Novelty</div>
            </div>
        </div>
        
        <div class="viz-container">
            <h2>Novelty Map Visualization</h2>
            <img src="novelty_map.png" alt="Novelty Map" style="max-width: 100%;" />
        </div>
        
        <div>
            <h2>Top Novel Clusters</h2>
            <table class="cluster-table">
                <tr>
                    <th>Cluster ID</th>
                    <th>Novelty Score</th>
                    <th>Concepts</th>
                    <th>Creation Time</th>
                </tr>
                {"".join([f"<tr><td>{c.id}</td><td>{c.novelty_score:.3f}</td><td>{len(c.concepts)}</td><td>{datetime.fromtimestamp(c.creation_time).strftime('%Y-%m-%d %H:%M:%S')}</td></tr>" for c in sorted(self.clusters.values(), key=lambda x: x.novelty_score, reverse=True)[:5]])}
            </table>
        </div>
        
        <div>
            <h2>Emergence Events</h2>
            <table class="event-table">
                <tr>
                    <th>Event ID</th>
                    <th>Timestamp</th>
                    <th>Cluster</th>
                    <th>Novelty Score</th>
                    <th>Concepts Involved</th>
                </tr>
                {"".join([f"<tr><td>{e.id}</td><td>{datetime.fromtimestamp(e.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</td><td>{e.cluster_id}</td><td>{e.novelty_score:.3f}</td><td>{len(e.concepts_involved)}</td></tr>" for e in self.emergence_events])}
            </table>
        </div>
        
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>""")
        
        logger.info(f"Dashboard generated in {output_dir}")


def main():
    """Main function to run the novelty map."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Novelty Map of Emergent Constructs")
    parser.add_argument("--seeds", type=int, default=4, help="Number of seed concepts")
    parser.add_argument("--concepts", type=int, default=40, help="Number of total concepts")
    parser.add_argument("--threshold", type=float, default=0.7, help="Novelty threshold (0-1)")
    parser.add_argument("--import-data", type=str, help="Import data from JSON file")
    parser.add_argument("--export-data", type=str, help="Export data to JSON file")
    parser.add_argument("--output-dir", type=str, help="Directory to save dashboard")
    
    args = parser.parse_args()
    
    logger.info(f"Initializing novelty map with threshold {args.threshold}")
    novelty_map = NoveltyMap(novelty_threshold=args.threshold)
    
    # Import data if specified
    if args.import_data:
        logger.info(f"Importing data from {args.import_data}...")
        novelty_map.import_data(args.import_data)
    else:
        # Create mock data
        logger.info(f"Creating mock data with {args.seeds} seeds and {args.concepts} concepts...")
        novelty_map._create_mock_data(seed_count=args.seeds, concept_count=args.concepts)
    
    # Export data if specified
    if args.export_data:
        logger.info(f"Exporting data to {args.export_data}...")
        novelty_map.export_data(args.export_data)
    
    # Generate dashboard if specified
    if args.output_dir:
        logger.info(f"Generating dashboard in {args.output_dir}...")
        novelty_map.generate_dashboard(args.output_dir)
    else:
        # Just show visualization
        logger.info("Generating visualization...")
        novelty_map.visualize_novelty_map()
        
        # Generate report
        logger.info("Generating report...")
        novelty_map.generate_report()
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 