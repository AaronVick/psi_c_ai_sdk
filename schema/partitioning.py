#!/usr/bin/env python3
"""
Internal Workspace Partitioning System

This module enables soft modularization of schema space without requiring fixed symbolic slots
as seen in traditional cognitive architectures like ACT-R. Instead, it uses dynamic clustering
over coherence vectors to create flexible, emergent partitions that adapt to the agent's
evolving knowledge and cognitive state.

Key capabilities:
- Clustering schema elements based on coherence vectors
- Dynamic partition boundary adjustment as new information is integrated
- Reflective routing of cognitive operations based on ΨC zone activation
- Visualization of schema partitioning and activation patterns
- Partition stability metrics with temporal tracking

Mathematical foundations:
- Partition optimization using k-means:
  P_k = argmin ∑_{i ∈ P_k} ||M_i - c_k||²
- Dynamic reflective routing based on partition activation
- Partition coherence scoring for stability assessment

This approach contrasts with traditional architectures that use predefined, fixed module
structures, allowing for more flexible, emergent specialization while maintaining
integration across the schema space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import distance
import networkx as nx
import pandas as pd
import json
import os
import logging
import time
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ΨC SDK components if available
try:
    from psi_c_ai_sdk.schema import schema_manager
    from psi_c_ai_sdk.belief import belief_network
    MOCK_MODE = False
except ImportError:
    logger.warning("ΨC SDK components not found. Running in mock mode.")
    MOCK_MODE = True


@dataclass
class SchemaElement:
    """A schema element with vector representation and metadata."""
    id: str
    vector: np.ndarray
    content: Any  # The actual content (belief, concept, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0  # Current activation level
    last_accessed: float = field(default_factory=time.time)
    partition_id: Optional[int] = None  # Assigned partition
    coherence_scores: Dict[str, float] = field(default_factory=dict)
    
    def update_activation(self, value: float, decay_factor: float = 0.05):
        """Update the activation level with decay."""
        current_time = time.time()
        time_diff = current_time - self.last_accessed
        self.activation = self.activation * np.exp(-decay_factor * time_diff) + value
        self.last_accessed = current_time
        
    def distance_to(self, other: 'SchemaElement') -> float:
        """Calculate vector distance to another schema element."""
        return distance.cosine(self.vector, other.vector)


@dataclass
class Partition:
    """A partition (cluster) of schema elements."""
    id: int
    centroid: np.ndarray
    elements: List[SchemaElement] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    stability_history: List[float] = field(default_factory=list)
    activation_level: float = 0.0  # Current overall activation
    specialization: Dict[str, float] = field(default_factory=dict)  # What this partition specializes in
    
    def add_element(self, element: SchemaElement):
        """Add an element to this partition."""
        self.elements.append(element)
        element.partition_id = self.id
        self.last_modified = time.time()
        
    def remove_element(self, element_id: str):
        """Remove an element from this partition."""
        self.elements = [e for e in self.elements if e.id != element_id]
        self.last_modified = time.time()
        
    def calculate_coherence(self) -> float:
        """Calculate internal coherence of this partition."""
        if len(self.elements) <= 1:
            return 1.0  # Perfect coherence for single elements
        
        # Average pairwise similarity
        similarities = []
        vectors = np.array([e.vector for e in self.elements])
        
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                similarity = 1 - distance.cosine(vectors[i], vectors[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def update_centroid(self):
        """Update the centroid based on current elements."""
        if not self.elements:
            return
        
        vectors = np.array([e.vector for e in self.elements])
        self.centroid = np.mean(vectors, axis=0)
        self.last_modified = time.time()
        
    def update_activation(self, decay_factor: float = 0.01):
        """Update the overall activation based on elements."""
        current_time = time.time()
        time_diff = current_time - self.last_modified
        
        # Decay existing activation
        self.activation = self.activation * np.exp(-decay_factor * time_diff)
        
        # Add current element activations
        if self.elements:
            self.activation += np.mean([e.activation for e in self.elements])
        
        self.last_modified = current_time
        
    def calculate_stability(self) -> float:
        """Calculate stability metric for this partition."""
        coherence = self.calculate_coherence()
        
        # Record for history
        self.stability_history.append(coherence)
        
        # Keep only the last 10 values
        if len(self.stability_history) > 10:
            self.stability_history = self.stability_history[-10:]
            
        return coherence
    
    def get_top_elements(self, n: int = 5) -> List[SchemaElement]:
        """Get the most central elements to this partition."""
        if not self.elements:
            return []
        
        # Calculate distances to centroid
        distances = [(e, distance.cosine(e.vector, self.centroid)) for e in self.elements]
        distances.sort(key=lambda x: x[1])  # Sort by distance (ascending)
        
        return [e for e, _ in distances[:n]]
    
    def identify_specialization(self, feature_extractor: Callable = None):
        """Identify what this partition specializes in."""
        if not self.elements or len(self.elements) < 3:
            return
        
        # Extract features from elements if provided
        if feature_extractor:
            features = [feature_extractor(e.content) for e in self.elements]
            if all(features):
                # Count occurrences of each feature
                feature_counts = {}
                for f_dict in features:
                    for k, v in f_dict.items():
                        if k not in feature_counts:
                            feature_counts[k] = 0
                        feature_counts[k] += v
                
                # Normalize by number of elements
                self.specialization = {k: v / len(self.elements) for k, v in feature_counts.items()}
                return
                
        # Default: use element metadata if available
        keyword_counts = {}
        for element in self.elements:
            if 'keywords' in element.metadata:
                for kw in element.metadata['keywords']:
                    if kw not in keyword_counts:
                        keyword_counts[kw] = 0
                    keyword_counts[kw] += 1
        
        # Find most common keywords
        if keyword_counts:
            total = sum(keyword_counts.values())
            self.specialization = {k: v / total for k, v in keyword_counts.items()}


class WorkspacePartitioner:
    """
    System for soft modularization of schema space using clustering over coherence vectors.
    
    This class implements dynamic partitioning of the workspace based on coherence relationships
    rather than fixed architectural boundaries, allowing for emergent specialization while
    maintaining integration.
    """
    
    def __init__(self, 
                 schema_manager=None,
                 vector_dim: int = 128,
                 min_partition_size: int = 3,
                 max_partitions: int = 8,
                 stability_threshold: float = 0.6,
                 partition_method: str = 'kmeans'):
        """
        Initialize the workspace partitioner.
        
        Args:
            schema_manager: The schema manager to use (if None, will use mock data)
            vector_dim: Dimensionality of the schema element vectors
            min_partition_size: Minimum elements for a partition to be valid
            max_partitions: Maximum number of partitions to create
            stability_threshold: Threshold for partition stability
            partition_method: Clustering method ('kmeans' or 'dbscan')
        """
        self.schema_manager = schema_manager
        self.vector_dim = vector_dim
        self.min_partition_size = min_partition_size
        self.max_partitions = max_partitions
        self.stability_threshold = stability_threshold
        self.partition_method = partition_method
        
        # Storage for schema elements and partitions
        self.elements: Dict[str, SchemaElement] = {}
        self.partitions: Dict[int, Partition] = {}
        
        # Tracking data
        self.stability_history: List[float] = []
        self.partition_count_history: List[int] = []
        self.last_partition_time = time.time()
        self.next_partition_id = 0
        
        # Initialize visualization
        self.fig = None
        self.axes = None
        self.projections = {}  # For caching dimensionality reduction
        
        # Check if we are in mock mode
        if MOCK_MODE or schema_manager is None:
            # Create mock data for demonstration
            logger.info("Initializing with mock data...")
            self._create_mock_data()
    
    def _create_mock_data(self, num_elements: int = 50):
        """Create mock data for demonstration."""
        themes = [
            {"name": "ethics", "center": np.random.normal(0, 0.1, self.vector_dim)},
            {"name": "science", "center": np.random.normal(1, 0.1, self.vector_dim)},
            {"name": "art", "center": np.random.normal(2, 0.1, self.vector_dim)},
            {"name": "technology", "center": np.random.normal(3, 0.1, self.vector_dim)},
            {"name": "philosophy", "center": np.random.normal(4, 0.1, self.vector_dim)},
        ]
        
        for i in range(num_elements):
            # Select a theme with some randomness
            theme = themes[i % len(themes)]
            
            # Create a vector near the theme center
            vector = theme["center"] + np.random.normal(0, 0.2, self.vector_dim)
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            # Create element
            element_id = f"element_{i}"
            element = SchemaElement(
                id=element_id,
                vector=vector,
                content=f"Content related to {theme['name']} #{i}",
                metadata={
                    "keywords": [theme["name"], f"topic_{i % 10}", f"concept_{i % 20}"],
                    "theme": theme["name"],
                    "created_at": time.time()
                },
                activation=np.random.random() * 0.5  # Random initial activation
            )
            
            self.elements[element_id] = element
    
    def sync_with_schema_manager(self):
        """Synchronize elements with the schema manager."""
        if MOCK_MODE or self.schema_manager is None:
            return
        
        # In a real implementation, this would retrieve schema elements
        # from the schema manager and convert them to our format
        raise NotImplementedError("Integration with actual schema manager not implemented")
    
    def partition_workspace(self):
        """Partition the schema elements into coherent clusters."""
        if not self.elements:
            logger.warning("No elements to partition.")
            return
        
        # Extract vectors from elements
        element_ids = list(self.elements.keys())
        vectors = np.array([self.elements[eid].vector for eid in element_ids])
        
        # Perform clustering
        if self.partition_method == 'kmeans':
            # Determine optimal number of clusters with silhouette score
            max_score = -1
            best_k = 2  # Default
            
            # Try different numbers of clusters
            for k in range(2, min(self.max_partitions + 1, len(vectors) // 2 + 1)):
                if len(vectors) < k:
                    continue
                    
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(vectors)
                
                # Count elements per cluster
                counts = np.bincount(labels)
                
                # Check if any cluster is too small
                if np.any(counts < self.min_partition_size):
                    continue
                
                # Calculate silhouette score if more than one cluster
                if k > 1:
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(vectors, labels)
                    
                    if score > max_score:
                        max_score = score
                        best_k = k
            
            # Use the best k
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            labels = kmeans.fit_predict(vectors)
            centroids = kmeans.cluster_centers_
            
        elif self.partition_method == 'dbscan':
            # Try to automatically determine eps
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(20, len(vectors)))
            nn.fit(vectors)
            distances, _ = nn.kneighbors(vectors)
            
            # Sort and find "elbow"
            knee_distances = np.sort(distances[:, -1])
            eps = np.mean(knee_distances) * 0.5  # Heuristic
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=self.min_partition_size)
            labels = dbscan.fit_predict(vectors)
            
            # Handle noise points (-1 labels) by assigning to nearest cluster
            if -1 in labels:
                # Get valid cluster labels
                valid_clusters = set(labels) - {-1}
                
                if valid_clusters:  # Only if we have valid clusters
                    # Calculate centroids for valid clusters
                    centroids = []
                    for cluster in valid_clusters:
                        mask = labels == cluster
                        centroids.append(np.mean(vectors[mask], axis=0))
                    centroids = np.array(centroids)
                    
                    # Assign noise points to nearest centroid
                    noise_indices = np.where(labels == -1)[0]
                    for idx in noise_indices:
                        distances = [np.linalg.norm(vectors[idx] - centroid) for centroid in centroids]
                        nearest_cluster = list(valid_clusters)[np.argmin(distances)]
                        labels[idx] = nearest_cluster
                else:
                    # If no valid clusters, create one cluster with everything
                    labels = np.zeros(len(vectors), dtype=int)
                    centroids = [np.mean(vectors, axis=0)]
                    centroids = np.array(centroids)
            else:
                # Calculate centroids for clusters
                centroids = []
                for cluster in set(labels):
                    mask = labels == cluster
                    centroids.append(np.mean(vectors[mask], axis=0))
                centroids = np.array(centroids)
        else:
            raise ValueError(f"Unknown partition method: {self.partition_method}")
        
        # Clear old partitions
        self.partitions = {}
        
        # Create new partitions
        for cluster_id in range(len(centroids)):
            partition = Partition(
                id=self.next_partition_id,
                centroid=centroids[cluster_id],
                elements=[]
            )
            self.partitions[self.next_partition_id] = partition
            self.next_partition_id += 1
        
        # Assign elements to partitions
        for idx, element_id in enumerate(element_ids):
            if idx < len(labels):  # Safety check
                label = labels[idx]
                # Find partition with this label
                for partition_id, partition in self.partitions.items():
                    if np.array_equal(partition.centroid, centroids[label]):
                        partition.add_element(self.elements[element_id])
                        break
        
        # Calculate overall stability
        stability = np.mean([p.calculate_stability() for p in self.partitions.values()])
        self.stability_history.append(stability)
        self.partition_count_history.append(len(self.partitions))
        
        # Update specializations
        for partition in self.partitions.values():
            partition.identify_specialization()
        
        self.last_partition_time = time.time()
        logger.info(f"Partitioned workspace into {len(self.partitions)} partitions with stability {stability:.3f}")
    
    def route_query(self, query_vector: np.ndarray, top_n: int = 2) -> List[int]:
        """
        Route a query to the most relevant partitions.
        
        Args:
            query_vector: Vector representation of the query
            top_n: Number of top partitions to return
            
        Returns:
            List of partition IDs most relevant to the query
        """
        if not self.partitions:
            return []
        
        # Calculate similarity to each partition centroid
        similarities = []
        for partition_id, partition in self.partitions.items():
            similarity = 1 - distance.cosine(query_vector, partition.centroid)
            similarities.append((partition_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N partition IDs
        return [pid for pid, _ in similarities[:top_n]]
    
    def add_element(self, element_id: str, vector: np.ndarray, content: Any, metadata: Dict[str, Any] = None):
        """
        Add a new schema element to the workspace.
        
        Args:
            element_id: Unique identifier for the element
            vector: Vector representation
            content: The actual content
            metadata: Additional metadata
        """
        if element_id in self.elements:
            logger.warning(f"Element {element_id} already exists. Updating...")
            self.elements[element_id].vector = vector
            self.elements[element_id].content = content
            if metadata:
                self.elements[element_id].metadata.update(metadata)
            return
        
        element = SchemaElement(
            id=element_id,
            vector=vector,
            content=content,
            metadata=metadata or {},
            activation=0.2  # Initial activation
        )
        
        self.elements[element_id] = element
        
        # Assign to best matching partition if we have partitions
        if self.partitions:
            best_partition_id = self.route_query(vector, top_n=1)[0]
            self.partitions[best_partition_id].add_element(element)
            
            # Check if we need to repartition
            if time.time() - self.last_partition_time > 300:  # Every 5 minutes
                self.partition_workspace()
        else:
            # Initial partitioning
            self.partition_workspace()
    
    def activate_elements(self, element_ids: List[str], activation_value: float = 0.5):
        """
        Activate specific schema elements.
        
        Args:
            element_ids: List of element IDs to activate
            activation_value: Activation value to add
        """
        for eid in element_ids:
            if eid in self.elements:
                self.elements[eid].update_activation(activation_value)
        
        # Update partition activations
        for partition in self.partitions.values():
            partition.update_activation()
    
    def get_active_partitions(self, threshold: float = 0.2) -> List[int]:
        """
        Get currently active partitions.
        
        Args:
            threshold: Activation threshold
            
        Returns:
            List of active partition IDs
        """
        active = []
        for pid, partition in self.partitions.items():
            if partition.activation >= threshold:
                active.append(pid)
        
        return active
    
    def visualize_workspace(self, show_inactive: bool = False, output_file: Optional[str] = None):
        """
        Visualize the current workspace partitioning.
        
        Args:
            show_inactive: Whether to show inactive elements
            output_file: If provided, save visualization to this file
        """
        if not self.elements:
            logger.warning("No elements to visualize.")
            return
        
        # Project vectors to 2D for visualization if needed
        element_ids = list(self.elements.keys())
        vectors = np.array([self.elements[eid].vector for eid in element_ids])
        
        # Check if we need to recompute projection
        compute_projection = True
        if 'tsne' in self.projections:
            if self.projections['tsne_elements'] == set(element_ids):
                compute_projection = False
        
        if compute_projection:
            # Use PCA first to reduce dimensions, then t-SNE
            if vectors.shape[1] > 50:
                pca = PCA(n_components=50)
                reduced_vectors = pca.fit_transform(vectors)
            else:
                reduced_vectors = vectors
                
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(3, len(vectors) // 5)))
            projections = tsne.fit_transform(reduced_vectors)
            
            # Cache the projection
            self.projections['tsne'] = projections
            self.projections['tsne_elements'] = set(element_ids)
        else:
            projections = self.projections['tsne']
        
        # Create a figure if needed
        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Clear axes
        for ax in self.axes.flatten():
            ax.clear()
        
        # Visualization 1: Elements colored by partition
        ax1 = self.axes[0, 0]
        for pid, partition in self.partitions.items():
            # Get elements in this partition
            partition_element_ids = [e.id for e in partition.elements]
            
            # Filter to elements we have projections for
            valid_indices = [i for i, eid in enumerate(element_ids) if eid in partition_element_ids]
            
            if valid_indices:
                # Get projection points for these elements
                x = projections[valid_indices, 0]
                y = projections[valid_indices, 1]
                
                # Plot elements
                ax1.scatter(x, y, label=f"Partition {pid}", alpha=0.7)
                
                # Plot centroid
                centroid_proj = np.mean(projections[valid_indices], axis=0)
                ax1.scatter(centroid_proj[0], centroid_proj[1], marker='*', s=200, 
                           edgecolor='black', linewidth=1, alpha=0.8)
                ax1.text(centroid_proj[0], centroid_proj[1], str(pid), fontsize=12)
        
        ax1.set_title("Schema Elements by Partition")
        ax1.legend(loc='upper right')
        ax1.set_xlabel("t-SNE Dimension 1")
        ax1.set_ylabel("t-SNE Dimension 2")
        
        # Visualization 2: Elements colored by activation
        ax2 = self.axes[0, 1]
        
        # Get activations
        activations = np.array([self.elements[eid].activation for eid in element_ids])
        
        # Filter by activation if requested
        if not show_inactive:
            active_mask = activations > 0.1
            if np.any(active_mask):
                x = projections[active_mask, 0]
                y = projections[active_mask, 1]
                c = activations[active_mask]
                ax2.scatter(x, y, c=c, cmap='plasma', alpha=0.7)
            
                # Add colorbar
                cbar = plt.colorbar(ax2.collections[0], ax=ax2)
                cbar.set_label('Activation Level')
        else:
            # Show all elements
            x = projections[:, 0]
            y = projections[:, 1]
            c = activations
            sc = ax2.scatter(x, y, c=c, cmap='plasma', alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax2)
            cbar.set_label('Activation Level')
            
        ax2.set_title("Schema Elements by Activation")
        ax2.set_xlabel("t-SNE Dimension 1")
        ax2.set_ylabel("t-SNE Dimension 2")
        
        # Visualization 3: Partition stability over time
        ax3 = self.axes[1, 0]
        x = range(len(self.stability_history))
        ax3.plot(x, self.stability_history, 'b-', marker='o')
        ax3.set_title("Partition Stability Over Time")
        ax3.set_xlabel("Partition Iterations")
        ax3.set_ylabel("Stability Score")
        ax3.grid(True, alpha=0.3)
        
        # Add threshold line
        ax3.axhline(y=self.stability_threshold, color='r', linestyle='--', alpha=0.7)
        ax3.text(0, self.stability_threshold + 0.02, f"Threshold: {self.stability_threshold}", color='r')
        
        # Visualization 4: Partition specialization
        ax4 = self.axes[1, 1]
        
        # Get active partitions
        active_partitions = self.get_active_partitions()
        
        if active_partitions:
            # Get specializations of active partitions
            specializations = []
            partition_names = []
            
            for pid in active_partitions:
                partition = self.partitions[pid]
                if partition.specialization:
                    # Get top 3 specializations
                    top_spec = sorted(partition.specialization.items(), key=lambda x: x[1], reverse=True)[:3]
                    spec_str = ', '.join([f"{k}: {v:.2f}" for k, v in top_spec])
                    specializations.append(spec_str)
                    partition_names.append(f"Partition {pid}")
            
            if specializations:
                # Create table
                ax4.axis('tight')
                ax4.axis('off')
                table = ax4.table(
                    cellText=[[s] for s in specializations],
                    rowLabels=partition_names,
                    colLabels=["Top Specializations"],
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
        
        ax4.set_title("Partition Specializations")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        
        # Show
        plt.show()
    
    def export_partition_data(self, output_file: str):
        """
        Export partitioning data for external visualization or analysis.
        
        Args:
            output_file: File to write the data to
        """
        data = {
            "partitions": {},
            "elements": {},
            "metrics": {
                "stability_history": self.stability_history,
                "partition_count_history": self.partition_count_history,
                "last_partition_time": self.last_partition_time
            }
        }
        
        # Export partition data
        for pid, partition in self.partitions.items():
            data["partitions"][str(pid)] = {
                "id": partition.id,
                "element_count": len(partition.elements),
                "creation_time": partition.creation_time,
                "last_modified": partition.last_modified,
                "stability": partition.calculate_stability(),
                "activation_level": partition.activation_level,
                "element_ids": [e.id for e in partition.elements],
                "specialization": partition.specialization
            }
        
        # Export element data (top 1000 most active to keep size reasonable)
        sorted_elements = sorted(
            self.elements.values(), 
            key=lambda e: e.activation,
            reverse=True
        )[:1000]
        
        for element in sorted_elements:
            data["elements"][element.id] = {
                "id": element.id,
                "partition_id": element.partition_id,
                "activation": element.activation,
                "last_accessed": element.last_accessed,
                "metadata": {k: v for k, v in element.metadata.items() if isinstance(v, (str, int, float, bool))}
            }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Partition data exported to {output_file}")
    
    def dashboard(self, show_inactive: bool = False, output_dir: str = None):
        """
        Generate a dashboard visualization of the workspace partitioning.
        
        Args:
            show_inactive: Whether to show inactive elements
            output_dir: Directory to save dashboard files to
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save visualization
            viz_file = os.path.join(output_dir, "workspace_partitioning.png")
            self.visualize_workspace(show_inactive=show_inactive, output_file=viz_file)
            
            # Export data
            data_file = os.path.join(output_dir, "partitioning_data.json")
            self.export_partition_data(data_file)
            
            # Generate HTML dashboard if matplotlib has HTML capabilities
            try:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.backends.backend_svg import FigureCanvasSVG
                
                dashboard_file = os.path.join(output_dir, "dashboard.html")
                
                # Create SVG visualization
                fig_file = os.path.join(output_dir, "workspace_viz.svg")
                self.fig.savefig(fig_file, format='svg', bbox_inches='tight')
                
                # Create HTML dashboard
                with open(dashboard_file, 'w') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>ΨC Workspace Partitioning Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
        .viz-container {{ margin-bottom: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; margin-right: 15px; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .partition-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .partition-table th, .partition-table td {{ padding: 10px; border: 1px solid #ddd; }}
        .partition-table th {{ background-color: #f2f2f2; }}
        .timestamp {{ font-size: 12px; color: #666; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ΨC Workspace Partitioning Dashboard</h1>
            <p>Visualizing schema partitioning based on coherence vectors</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{len(self.partitions)}</div>
                <div class="metric-label">Active Partitions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(self.elements)}</div>
                <div class="metric-label">Total Schema Elements</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.stability_history[-1]:.2f}</div>
                <div class="metric-label">Current Stability</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(self.get_active_partitions())}</div>
                <div class="metric-label">Currently Active Partitions</div>
            </div>
        </div>
        
        <div class="viz-container">
            <h2>Workspace Visualization</h2>
            <img src="workspace_viz.svg" alt="Workspace Visualization" style="max-width: 100%;" />
        </div>
        
        <div>
            <h2>Partition Information</h2>
            <table class="partition-table">
                <tr>
                    <th>ID</th>
                    <th>Elements</th>
                    <th>Stability</th>
                    <th>Activation</th>
                    <th>Top Specializations</th>
                </tr>
                {"".join([f"<tr><td>{p.id}</td><td>{len(p.elements)}</td><td>{p.calculate_stability():.2f}</td><td>{p.activation_level:.2f}</td><td>{'<br>'.join([f'{k}: {v:.2f}' for k, v in sorted(p.specialization.items(), key=lambda x: x[1], reverse=True)[:3]])}</td></tr>" for p in self.partitions.values()])}
            </table>
        </div>
        
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>""")
                
                logger.info(f"Dashboard saved to {dashboard_file}")
            except Exception as e:
                logger.error(f"Failed to generate HTML dashboard: {e}")
        else:
            # Just show the visualization
            self.visualize_workspace(show_inactive=show_inactive)


def main():
    """Main function to demonstrate the partitioning system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ΨC Workspace Partitioning System")
    parser.add_argument("--vector-dim", type=int, default=128, help="Vector dimension for mock data")
    parser.add_argument("--elements", type=int, default=100, help="Number of elements for mock data")
    parser.add_argument("--partitions", type=int, default=5, help="Maximum number of partitions")
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "dbscan"],
                       help="Partitioning method")
    parser.add_argument("--output-dir", type=str, help="Directory to save dashboard")
    parser.add_argument("--show-inactive", action="store_true", help="Show inactive elements")
    
    args = parser.parse_args()
    
    logger.info("Initializing workspace partitioner...")
    partitioner = WorkspacePartitioner(
        vector_dim=args.vector_dim,
        max_partitions=args.partitions,
        partition_method=args.method
    )
    
    # Create mock data if not already done
    if len(partitioner.elements) < args.elements:
        logger.info(f"Creating {args.elements} mock elements...")
        partitioner._create_mock_data(args.elements)
    
    # Partition the workspace
    logger.info("Partitioning workspace...")
    partitioner.partition_workspace()
    
    # Simulate some activations
    logger.info("Simulating element activations...")
    element_ids = list(partitioner.elements.keys())
    np.random.shuffle(element_ids)
    partitioner.activate_elements(element_ids[:20], 0.8)
    
    # Generate dashboard
    logger.info("Generating dashboard...")
    partitioner.dashboard(show_inactive=args.show_inactive, output_dir=args.output_dir)
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 