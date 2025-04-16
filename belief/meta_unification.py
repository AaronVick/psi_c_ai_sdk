#!/usr/bin/env python3
"""
Unification Tracker for Meta-Beliefs

This module implements a system for detecting and tracking meta-beliefs (beliefs about beliefs)
that serve to unify lower-level memory and belief structures. Unlike traditional cognitive
architectures like SOAR/ACT-R that use explicit top-down rule systems, the ΨC approach
discovers unifying meta-structures emergently through coherence optimization.

Key capabilities:
- Detection of meta-beliefs that increase overall schema coherence
- Tracking of belief cluster entropy to identify unification events
- Visualization of meta-belief formation and influence on belief networks
- Analysis of emergent top-down organization without explicit symbolic rules

Mathematical foundation:
- Entropy of belief clusters: H_meta = -∑ p_k log p_k
  where p_k = |C_k|/∑_k |C_k|
- Unification strength: U(m, C) = H(C) - H(C|m)
- Meta-belief influence radius: R_m = |{b ∈ B | coh(b|m) > coh(b)}|

This system provides insights into how higher-order cognitive structures emerge organically
from lower-level beliefs through coherence maximization rather than through explicit
programming or rule-based architectures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import pandas as pd
import seaborn as sns
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
    from psi_c_ai_sdk.belief import belief_network
    from psi_c_ai_sdk.schema import schema_manager
    MOCK_MODE = False
except ImportError:
    logger.warning("ΨC SDK components not found. Running in mock mode.")
    MOCK_MODE = True


@dataclass
class Belief:
    """Represents a single belief in the system."""
    id: str
    content: str
    vector: np.ndarray
    confidence: float
    is_meta: bool = False
    sources: List[str] = field(default_factory=list)
    influenced_by: List[str] = field(default_factory=list)
    influences: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "confidence": self.confidence,
            "is_meta": self.is_meta,
            "sources": self.sources,
            "influenced_by": self.influenced_by,
            "influences": self.influences,
            "created_at": self.created_at,
            "modified_at": self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Belief':
        """Create from dictionary."""
        vector = np.array(data["vector"]) if "vector" in data else np.zeros(128)
        return cls(
            id=data["id"],
            content=data["content"],
            vector=vector,
            confidence=data["confidence"],
            is_meta=data.get("is_meta", False),
            sources=data.get("sources", []),
            influenced_by=data.get("influenced_by", []),
            influences=data.get("influences", []),
            created_at=data.get("created_at", time.time()),
            modified_at=data.get("modified_at", time.time())
        )


@dataclass
class BeliefCluster:
    """A cluster of related beliefs."""
    id: str
    beliefs: List[str]  # Belief IDs
    centroid: np.ndarray
    meta_beliefs: List[str] = field(default_factory=list)  # Meta-belief IDs
    entropy: float = 0.0
    coherence: float = 0.0
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "beliefs": self.beliefs,
            "centroid": self.centroid.tolist() if isinstance(self.centroid, np.ndarray) else self.centroid,
            "meta_beliefs": self.meta_beliefs,
            "entropy": self.entropy,
            "coherence": self.coherence,
            "created_at": self.created_at,
            "modified_at": self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BeliefCluster':
        """Create from dictionary."""
        centroid = np.array(data["centroid"]) if "centroid" in data else np.zeros(128)
        return cls(
            id=data["id"],
            beliefs=data["beliefs"],
            centroid=centroid,
            meta_beliefs=data.get("meta_beliefs", []),
            entropy=data.get("entropy", 0.0),
            coherence=data.get("coherence", 0.0),
            created_at=data.get("created_at", time.time()),
            modified_at=data.get("modified_at", time.time())
        )


@dataclass
class MetaUnificationEvent:
    """Represents a unification event where a meta-belief unifies existing beliefs."""
    id: str
    meta_belief_id: str
    unified_beliefs: List[str]  # Belief IDs
    entropy_before: float
    entropy_after: float
    unification_strength: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "meta_belief_id": self.meta_belief_id,
            "unified_beliefs": self.unified_beliefs,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "unification_strength": self.unification_strength,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MetaUnificationEvent':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            meta_belief_id=data["meta_belief_id"],
            unified_beliefs=data["unified_beliefs"],
            entropy_before=data["entropy_before"],
            entropy_after=data["entropy_after"],
            unification_strength=data["unification_strength"],
            timestamp=data.get("timestamp", time.time())
        )

class MetaUnificationTracker:
    """
    Main class for tracking and analyzing meta-beliefs that unify lower-level beliefs.
    
    This tracker identifies emergent meta-beliefs that provide unifying frameworks for
    collections of lower-level beliefs, calculates their impact on belief coherence,
    and tracks the evolution of these meta-structures over time.
    """
    
    def __init__(self, belief_network=None, vector_dim: int = 128):
        """
        Initialize the meta-belief unification tracker.
        
        Args:
            belief_network: Optional existing belief network to integrate with
            vector_dim: Dimensionality of belief vectors
        """
        self.vector_dim = vector_dim
        self.belief_network = belief_network
        
        # Storage
        self.beliefs: Dict[str, Belief] = {}
        self.clusters: Dict[str, BeliefCluster] = {}
        self.meta_beliefs: Dict[str, Belief] = {}
        self.unification_events: List[MetaUnificationEvent] = []
        
        # Tracking metrics
        self.system_entropy_history: List[Tuple[float, float]] = []  # (timestamp, entropy)
        self.meta_belief_count_history: List[Tuple[float, int]] = []  # (timestamp, count)
        
        # Graph representation
        self.belief_graph = nx.DiGraph()
        
        # Visualization
        self.fig = None
        self.axes = None
        
        # Check if we're in mock mode
        if MOCK_MODE or belief_network is None:
            logger.info("Initializing with mock data...")
            self._create_mock_data()
    
    def _create_mock_data(self, num_beliefs: int = 50):
        """Create mock data for demonstration purposes."""
        # Define themes and their associated beliefs
        themes = {
            "ethics": [
                "Honesty is important",
                "People should be treated fairly",
                "Harm reduction is a moral good",
                "Intentions matter in moral judgments",
                "Justice requires equitable treatment"
            ],
            "epistemology": [
                "Knowledge requires justification",
                "Empirical evidence is important",
                "Some beliefs can be justified a priori",
                "Skepticism about certain knowledge is reasonable",
                "Scientific consensus has epistemic weight"
            ],
            "identity": [
                "Personal identity persists through time",
                "Memory is central to identity",
                "Identity has social components",
                "Values shape personal identity",
                "Self-concept evolves over time"
            ],
            "agency": [
                "Free will is compatible with determinism",
                "Agency requires capacity for choice",
                "People are responsible for their actions",
                "External factors can constrain agency",
                "Autonomy is valuable for well-being"
            ]
        }
        
        # Create meta-beliefs (beliefs about the themes)
        meta_beliefs_content = {
            "ethics": "Ethical principles guide how we should treat others and evaluate actions",
            "epistemology": "Knowledge claims should be evaluated based on evidence and justification methods",
            "identity": "Personal identity comprises psychological continuity, memories, and social dimensions",
            "agency": "Human agency involves the capacity to make choices and act according to values"
        }
        
        # Create vectors for each theme (for clustering purposes)
        theme_vectors = {}
        for theme in themes:
            theme_vectors[theme] = np.random.normal(0, 0.5, self.vector_dim)
            theme_vectors[theme] = theme_vectors[theme] / np.linalg.norm(theme_vectors[theme])
        
        # Add meta-beliefs
        for theme, content in meta_beliefs_content.items():
            meta_id = f"meta_{theme}"
            meta_vector = theme_vectors[theme] + np.random.normal(0, 0.1, self.vector_dim)
            meta_vector = meta_vector / np.linalg.norm(meta_vector)
            
            meta_belief = Belief(
                id=meta_id,
                content=content,
                vector=meta_vector,
                confidence=0.85 + np.random.random() * 0.1,
                is_meta=True,
                sources=["reflection", "abstraction"]
            )
            
            self.beliefs[meta_id] = meta_belief
            self.meta_beliefs[meta_id] = meta_belief
            self.belief_graph.add_node(meta_id, belief=meta_belief)
        
        # Add regular beliefs
        belief_count = 0
        for theme, belief_contents in themes.items():
            cluster_beliefs = []
            meta_id = f"meta_{theme}"
            
            for content in belief_contents:
                # Create a vector close to the theme vector
                vector = theme_vectors[theme] + np.random.normal(0, 0.2, self.vector_dim)
                vector = vector / np.linalg.norm(vector)
                
                belief_id = f"belief_{belief_count}"
                belief = Belief(
                    id=belief_id,
                    content=content,
                    vector=vector,
                    confidence=0.7 + np.random.random() * 0.2,
                    influenced_by=[meta_id],
                )
                
                self.beliefs[belief_id] = belief
                self.belief_graph.add_node(belief_id, belief=belief)
                self.belief_graph.add_edge(meta_id, belief_id, weight=0.7 + np.random.random() * 0.2)
                
                # Update the meta-belief influences
                self.beliefs[meta_id].influences.append(belief_id)
                
                cluster_beliefs.append(belief_id)
                belief_count += 1
            
            # Create a cluster for this theme
            cluster_id = f"cluster_{theme}"
            cluster = BeliefCluster(
                id=cluster_id,
                beliefs=cluster_beliefs,
                centroid=theme_vectors[theme],
                meta_beliefs=[meta_id],
                coherence=0.7 + np.random.random() * 0.2
            )
            
            self.clusters[cluster_id] = cluster
        
        # Add random beliefs not strongly connected to any meta-belief
        for i in range(belief_count, num_beliefs):
            vector = np.random.normal(0, 1, self.vector_dim)
            vector = vector / np.linalg.norm(vector)
            
            belief_id = f"belief_{i}"
            belief = Belief(
                id=belief_id,
                content=f"Miscellaneous belief #{i}",
                vector=vector,
                confidence=0.5 + np.random.random() * 0.3
            )
            
            self.beliefs[belief_id] = belief
            self.belief_graph.add_node(belief_id, belief=belief)
        
        # Calculate initial entropy
        self._update_system_metrics()
    
    def _update_system_metrics(self):
        """Update system-wide metrics like entropy and meta-belief count."""
        # Calculate system entropy
        entropy = self.calculate_system_entropy()
        self.system_entropy_history.append((time.time(), entropy))
        
        # Update meta-belief count
        meta_count = len(self.meta_beliefs)
        self.meta_belief_count_history.append((time.time(), meta_count))
    
    def calculate_system_entropy(self) -> float:
        """
        Calculate the entropy of the belief system based on cluster distributions.
        
        Returns:
            System entropy value
        """
        if not self.clusters:
            return 0.0
        
        # Get total number of beliefs in clusters
        total_beliefs = sum(len(cluster.beliefs) for cluster in self.clusters.values())
        
        if total_beliefs == 0:
            return 0.0
        
        # Calculate entropy using cluster probabilities
        entropy = 0.0
        for cluster in self.clusters.values():
            p_k = len(cluster.beliefs) / total_beliefs
            if p_k > 0:  # Avoid log(0)
                entropy -= p_k * np.log(p_k)
        
        return entropy
    
    def calculate_cluster_entropy(self, cluster_id: str) -> float:
        """
        Calculate the entropy within a specific belief cluster.
        
        Args:
            cluster_id: ID of the cluster to analyze
            
        Returns:
            Entropy value for the cluster
        """
        if cluster_id not in self.clusters:
            return 0.0
        
        cluster = self.clusters[cluster_id]
        
        if not cluster.beliefs:
            return 0.0
        
        # Use vector similarity to calculate entropy
        vectors = []
        for belief_id in cluster.beliefs:
            if belief_id in self.beliefs:
                vectors.append(self.beliefs[belief_id].vector)
        
        if not vectors:
            return 0.0
        
        # Convert to numpy array
        vectors = np.array(vectors)
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Calculate entropy based on similarity distribution
        mean_sim = np.mean(similarities)
        variance = np.var(similarities)
        
        # Higher variance = higher entropy
        return variance
    
    def detect_meta_beliefs(self, threshold: float = 0.6) -> List[str]:
        """
        Detect potential meta-beliefs that unify existing beliefs.
        
        Args:
            threshold: Minimum confidence threshold for meta-belief detection
            
        Returns:
            List of detected meta-belief IDs
        """
        # This is a simplified implementation for demonstration
        # In a real system, this would use more sophisticated detection methods
        
        # Get all non-meta beliefs
        regular_beliefs = {bid: b for bid, b in self.beliefs.items() if not b.is_meta}
        
        # Group beliefs by similarity
        from sklearn.cluster import AgglomerativeClustering
        
        if len(regular_beliefs) < 3:
            return []
        
        # Extract vectors and IDs
        vectors = []
        belief_ids = []
        
        for bid, belief in regular_beliefs.items():
            vectors.append(belief.vector)
            belief_ids.append(bid)
        
        vectors = np.array(vectors)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.4,  # Adjust based on vector similarity distribution
            linkage='average'
        )
        
        labels = clustering.fit_predict(vectors)
        
        # Find clusters with more than 2 beliefs
        cluster_dict = {}
        for i, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(belief_ids[i])
        
        meta_belief_candidates = []
        
        # For each potential cluster, check if it needs a meta-belief
        for label, cluster_beliefs in cluster_dict.items():
            if len(cluster_beliefs) < 3:
                continue
                
            # Check if this cluster already has a meta-belief influencing it
            has_meta = False
            for mb_id in self.meta_beliefs:
                influence_count = sum(1 for bid in cluster_beliefs if mb_id in self.beliefs[bid].influenced_by)
                if influence_count / len(cluster_beliefs) > 0.5:
                    has_meta = True
                    break
            
            if not has_meta:
                # Create a new meta-belief for this cluster
                meta_id = f"meta_{uuid.uuid4().hex[:8]}"
                
                # Get the beliefs to create a coherent meta-belief
                belief_contents = [self.beliefs[bid].content for bid in cluster_beliefs]
                
                # In a real system, this would use LLM or other methods to generate a coherent meta-belief
                meta_content = f"Unifying principle for: {', '.join(belief_contents[:3])}..."
                
                # Calculate centroid
                centroid = np.mean([self.beliefs[bid].vector for bid in cluster_beliefs], axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                
                meta_belief = Belief(
                    id=meta_id,
                    content=meta_content,
                    vector=centroid,
                    confidence=threshold,
                    is_meta=True,
                    influences=cluster_beliefs,
                    sources=["emergent_detection"]
                )
                
                self.beliefs[meta_id] = meta_belief
                self.meta_beliefs[meta_id] = meta_belief
                self.belief_graph.add_node(meta_id, belief=meta_belief)
                
                # Connect to beliefs in the cluster
                for bid in cluster_beliefs:
                    self.belief_graph.add_edge(meta_id, bid, weight=0.6)
                    self.beliefs[bid].influenced_by.append(meta_id)
                
                meta_belief_candidates.append(meta_id)
                
                # Calculate unification strength
                entropy_before = self.calculate_system_entropy()
                
                # Record unification event
                event = MetaUnificationEvent(
                    id=f"event_{uuid.uuid4().hex[:8]}",
                    meta_belief_id=meta_id,
                    unified_beliefs=cluster_beliefs,
                    entropy_before=entropy_before,
                    entropy_after=entropy_before * 0.9,  # Simulated reduction
                    unification_strength=len(cluster_beliefs) / 10  # Simplified calculation
                )
                
                self.unification_events.append(event)
        
        # Update system metrics
        self._update_system_metrics()
        
        return meta_belief_candidates
    
    def calculate_unification_strength(self, meta_belief_id: str) -> float:
        """
        Calculate how strongly a meta-belief unifies its influenced beliefs.
        
        Args:
            meta_belief_id: ID of the meta-belief to analyze
            
        Returns:
            Unification strength score (higher = stronger unification)
        """
        if meta_belief_id not in self.meta_beliefs:
            return 0.0
        
        meta_belief = self.meta_beliefs[meta_belief_id]
        
        if not meta_belief.influences:
            return 0.0
        
        # Get the beliefs influenced by this meta-belief
        influenced_beliefs = []
        for bid in meta_belief.influences:
            if bid in self.beliefs:
                influenced_beliefs.append(self.beliefs[bid])
        
        if not influenced_beliefs:
            return 0.0
        
        # Calculate average similarity to meta-belief
        similarities = []
        for belief in influenced_beliefs:
            sim = np.dot(meta_belief.vector, belief.vector) / (np.linalg.norm(meta_belief.vector) * np.linalg.norm(belief.vector))
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        # Calculate variance of similarities (lower variance = stronger unification)
        variance = np.var(similarities) if len(similarities) > 1 else 0.0
        
        # Calculate pairwise similarities between influenced beliefs
        pairwise_sims = []
        for i in range(len(influenced_beliefs)):
            for j in range(i+1, len(influenced_beliefs)):
                sim = np.dot(influenced_beliefs[i].vector, influenced_beliefs[j].vector) / (
                    np.linalg.norm(influenced_beliefs[i].vector) * np.linalg.norm(influenced_beliefs[j].vector))
                pairwise_sims.append(sim)
        
        avg_pairwise = np.mean(pairwise_sims) if pairwise_sims else 0.0
        
        # Unification strength combines direct similarity, consistency (low variance), 
        # and increased coherence among influenced beliefs
        strength = (avg_similarity * 0.4) + ((1 - variance) * 0.3) + (avg_pairwise * 0.3)
        
        return strength

    def visualize_belief_network(self, figsize=(12, 10), output_file=None):
        """
        Visualize the belief network showing meta-beliefs and their influences.
        
        Args:
            figsize: Figure size in inches
            output_file: If provided, save visualization to this file
        """
        if not self.beliefs:
            logger.warning("No beliefs to visualize.")
            return
        
        # Create figure if needed
        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Clear all axes
        for ax in self.axes.flatten():
            ax.clear()
        
        # Plot 1: Belief network graph
        ax1 = self.axes[0, 0]
        
        # Prepare node attributes
        node_colors = []
        node_sizes = []
        
        for node in self.belief_graph.nodes():
            if node in self.meta_beliefs:
                node_colors.append('red')
                node_sizes.append(300)
            else:
                node_colors.append('blue')
                node_sizes.append(100)
        
        # Use spring layout for the graph
        pos = nx.spring_layout(self.belief_graph, seed=42)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(
            self.belief_graph, pos, ax=ax1,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            self.belief_graph, pos, ax=ax1,
            arrows=True,
            arrowstyle='->',
            alpha=0.6
        )
        
        # Draw labels only for meta-beliefs
        labels = {node: node for node in self.belief_graph.nodes() if node in self.meta_beliefs}
        nx.draw_networkx_labels(
            self.belief_graph, pos, ax=ax1,
            labels=labels,
            font_size=10
        )
        
        ax1.set_title("Meta-Belief Influence Network")
        ax1.axis('off')
        
        # Plot 2: System entropy over time
        ax2 = self.axes[0, 1]
        
        if self.system_entropy_history:
            times, entropies = zip(*self.system_entropy_history)
            # Convert absolute times to relative times
            start_time = times[0]
            rel_times = [(t - start_time) / 60 for t in times]  # Convert to minutes
            
            ax2.plot(rel_times, entropies, 'b-', marker='o')
            ax2.set_xlabel("Time (minutes)")
            ax2.set_ylabel("System Entropy")
            ax2.set_title("Belief System Entropy Over Time")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No entropy data available", horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Unification strength by meta-belief
        ax3 = self.axes[1, 0]
        
        if self.meta_beliefs:
            meta_ids = list(self.meta_beliefs.keys())
            strengths = [self.calculate_unification_strength(mid) for mid in meta_ids]
            
            # Sort by strength
            sorted_indices = np.argsort(strengths)[::-1]  # Descending
            sorted_meta_ids = [meta_ids[i] for i in sorted_indices]
            sorted_strengths = [strengths[i] for i in sorted_indices]
            
            # Create shortened labels
            labels = [f"MB{i+1}" for i in range(len(sorted_meta_ids))]
            
            ax3.bar(labels, sorted_strengths, color='green', alpha=0.7)
            ax3.set_xlabel("Meta-Belief")
            ax3.set_ylabel("Unification Strength")
            ax3.set_title("Meta-Belief Unification Strength")
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add legend mapping
            legend_text = "\n".join([f"MB{i+1}: {mid}" for i, mid in enumerate(sorted_meta_ids)])
            ax3.text(1.05, 0.5, legend_text, transform=ax3.transAxes, fontsize=8,
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax3.text(0.5, 0.5, "No meta-beliefs available", horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Meta-belief count over time
        ax4 = self.axes[1, 1]
        
        if self.meta_belief_count_history:
            times, counts = zip(*self.meta_belief_count_history)
            # Convert absolute times to relative times
            start_time = times[0]
            rel_times = [(t - start_time) / 60 for t in times]  # Convert to minutes
            
            ax4.plot(rel_times, counts, 'r-', marker='x')
            ax4.set_xlabel("Time (minutes)")
            ax4.set_ylabel("Meta-Belief Count")
            ax4.set_title("Emergence of Meta-Beliefs Over Time")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No meta-belief history available", horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        
        # Show plot
        plt.show()
    
    def visualize_meta_influence(self, meta_belief_id: str, figsize=(10, 8), output_file=None):
        """
        Visualize the influence of a specific meta-belief on other beliefs.
        
        Args:
            meta_belief_id: ID of the meta-belief to analyze
            figsize: Figure size in inches
            output_file: If provided, save visualization to this file
        """
        if meta_belief_id not in self.meta_beliefs:
            logger.warning(f"Meta-belief {meta_belief_id} not found.")
            return
        
        meta_belief = self.meta_beliefs[meta_belief_id]
        
        # Create a subgraph of the meta-belief and influenced beliefs
        influenced_ids = meta_belief.influences
        subgraph_nodes = [meta_belief_id] + influenced_ids
        subgraph = self.belief_graph.subgraph(subgraph_nodes)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create positions - meta-belief in center, others around
        pos = nx.spring_layout(subgraph, seed=42)
        
        # Draw nodes
        node_colors = ['red' if node == meta_belief_id else 'blue' for node in subgraph.nodes()]
        node_sizes = [300 if node == meta_belief_id else 100 for node in subgraph.nodes()]
        
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos,
            arrows=True,
            arrowstyle='->',
            alpha=0.6
        )
        
        # Draw labels
        labels = {}
        for node in subgraph.nodes():
            if node == meta_belief_id:
                labels[node] = f"META: {self.beliefs[node].content[:20]}..."
            else:
                labels[node] = f"{self.beliefs[node].content[:20]}..."
        
        nx.draw_networkx_labels(
            subgraph, pos,
            labels=labels,
            font_size=8,
            font_color='black',
            font_family='sans-serif',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7)
        )
        
        # Add title
        plt.title(f"Influence of Meta-Belief: {meta_belief.content[:30]}...")
        plt.axis('off')
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Meta-influence visualization saved to {output_file}")
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, output_file=None):
        """
        Generate a detailed report of meta-belief unification analysis.
        
        Args:
            output_file: If provided, save report to this file
        """
        # Create a report DataFrame
        meta_data = []
        
        for mid, meta in self.meta_beliefs.items():
            unification_strength = self.calculate_unification_strength(mid)
            influence_count = len(meta.influences)
            
            # Find unification events for this meta-belief
            events = [e for e in self.unification_events if e.meta_belief_id == mid]
            entropy_reduction = np.mean([e.entropy_before - e.entropy_after for e in events]) if events else 0
            
            meta_data.append({
                'meta_belief_id': mid,
                'content': meta.content,
                'unification_strength': unification_strength,
                'influence_count': influence_count,
                'entropy_reduction': entropy_reduction,
                'creation_time': meta.created_at
            })
        
        if not meta_data:
            logger.warning("No meta-beliefs to report on.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(meta_data)
        
        # Sort by unification strength
        df = df.sort_values('unification_strength', ascending=False)
        
        # Generate summary statistics
        summary = {
            'total_meta_beliefs': len(self.meta_beliefs),
            'total_beliefs': len(self.beliefs),
            'avg_unification_strength': df['unification_strength'].mean(),
            'max_unification_strength': df['unification_strength'].max(),
            'avg_influenced_beliefs': df['influence_count'].mean(),
            'total_unification_events': len(self.unification_events),
            'current_system_entropy': self.system_entropy_history[-1][1] if self.system_entropy_history else 0,
            'emergence_rate': len(self.meta_beliefs) / (1 + len(self.beliefs) - len(self.meta_beliefs))
        }
        
        # Print to console
        print(f"\n{'='*50}")
        print(f"META-BELIEF UNIFICATION ANALYSIS REPORT")
        print(f"{'='*50}")
        
        print(f"\nSYSTEM SUMMARY:")
        print(f"Total Meta-Beliefs: {summary['total_meta_beliefs']}")
        print(f"Total Beliefs: {summary['total_beliefs']}")
        print(f"Meta-Belief Emergence Rate: {summary['emergence_rate']:.4f}")
        print(f"Current System Entropy: {summary['current_system_entropy']:.4f}")
        print(f"Total Unification Events: {summary['total_unification_events']}")
        
        print(f"\nMETA-BELIEF STATISTICS:")
        print(f"Average Unification Strength: {summary['avg_unification_strength']:.4f}")
        print(f"Maximum Unification Strength: {summary['max_unification_strength']:.4f}")
        print(f"Average Influenced Beliefs: {summary['avg_influenced_beliefs']:.2f}")
        
        print(f"\nTOP 5 META-BELIEFS BY UNIFICATION STRENGTH:")
        for _, row in df.head(5).iterrows():
            print(f"ID: {row['meta_belief_id']}")
            print(f"Content: {row['content']}")
            print(f"Unification Strength: {row['unification_strength']:.4f}")
            print(f"Influences: {row['influence_count']} beliefs")
            print(f"Entropy Reduction: {row['entropy_reduction']:.4f}")
            print("-" * 40)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"META-BELIEF UNIFICATION ANALYSIS REPORT\n")
                f.write(f"{'='*50}\n\n")
                
                f.write(f"SYSTEM SUMMARY:\n")
                for k, v in summary.items():
                    f.write(f"{k.replace('_', ' ').title()}: {v}\n")
                
                f.write(f"\nMETA-BELIEF DETAILS:\n")
                f.write(df.to_string(index=False))
                
                f.write(f"\n\nUNIFICATION EVENTS:\n")
                for event in self.unification_events:
                    f.write(f"Event ID: {event.id}\n")
                    f.write(f"Meta-Belief: {event.meta_belief_id}\n")
                    f.write(f"Unified Beliefs: {len(event.unified_beliefs)}\n")
                    f.write(f"Entropy Before: {event.entropy_before:.4f}\n")
                    f.write(f"Entropy After: {event.entropy_after:.4f}\n")
                    f.write(f"Unification Strength: {event.unification_strength:.4f}\n")
                    f.write(f"Timestamp: {datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 40 + "\n")
            
            logger.info(f"Report saved to {output_file}")
        
        return summary, df

    def add_belief(self, belief: Belief):
        """Add a new belief to the system."""
        if belief.id in self.beliefs:
            logger.warning(f"Belief {belief.id} already exists. Updating...")
            self.beliefs[belief.id] = belief
            if belief.is_meta:
                self.meta_beliefs[belief.id] = belief
            return
        
        self.beliefs[belief.id] = belief
        self.belief_graph.add_node(belief.id, belief=belief)
        
        if belief.is_meta:
            self.meta_beliefs[belief.id] = belief
        
        # Update connections
        for influence_id in belief.influences:
            if influence_id in self.beliefs:
                self.belief_graph.add_edge(belief.id, influence_id, weight=0.7)
                self.beliefs[influence_id].influenced_by.append(belief.id)
        
        for influencer_id in belief.influenced_by:
            if influencer_id in self.beliefs:
                self.belief_graph.add_edge(influencer_id, belief.id, weight=0.7)
                self.beliefs[influencer_id].influences.append(belief.id)
        
        # Update system metrics
        self._update_system_metrics()
    
    def merge_clusters(self, cluster_id1: str, cluster_id2: str) -> str:
        """
        Merge two belief clusters into one.
        
        Args:
            cluster_id1: First cluster ID
            cluster_id2: Second cluster ID
            
        Returns:
            ID of the merged cluster
        """
        if cluster_id1 not in self.clusters or cluster_id2 not in self.clusters:
            logger.warning(f"One or both clusters not found: {cluster_id1}, {cluster_id2}")
            return ""
        
        cluster1 = self.clusters[cluster_id1]
        cluster2 = self.clusters[cluster_id2]
        
        # Create a new merged cluster
        merged_id = f"cluster_merged_{uuid.uuid4().hex[:8]}"
        
        # Combine belief lists, removing duplicates
        combined_beliefs = list(set(cluster1.beliefs + cluster2.beliefs))
        
        # Calculate new centroid
        vectors = []
        for bid in combined_beliefs:
            if bid in self.beliefs:
                vectors.append(self.beliefs[bid].vector)
        
        if not vectors:
            logger.warning("No valid beliefs to calculate centroid.")
            return ""
        
        new_centroid = np.mean(vectors, axis=0)
        new_centroid = new_centroid / np.linalg.norm(new_centroid)
        
        # Combine meta-beliefs
        combined_meta = list(set(cluster1.meta_beliefs + cluster2.meta_beliefs))
        
        # Create the merged cluster
        merged_cluster = BeliefCluster(
            id=merged_id,
            beliefs=combined_beliefs,
            centroid=new_centroid,
            meta_beliefs=combined_meta,
            created_at=time.time()
        )
        
        # Calculate new metrics
        merged_cluster.entropy = self.calculate_cluster_entropy(merged_id)
        
        # Add to clusters
        self.clusters[merged_id] = merged_cluster
        
        # Remove old clusters
        del self.clusters[cluster_id1]
        del self.clusters[cluster_id2]
        
        # Update system metrics
        self._update_system_metrics()
        
        return merged_id
    
    def export_data(self, output_file: str):
        """
        Export all tracker data to JSON for persistence.
        
        Args:
            output_file: File to save the data to
        """
        data = {
            "beliefs": {},
            "clusters": {},
            "unification_events": [],
            "system_entropy_history": self.system_entropy_history,
            "meta_belief_count_history": self.meta_belief_count_history
        }
        
        # Export beliefs
        for bid, belief in self.beliefs.items():
            data["beliefs"][bid] = belief.to_dict()
        
        # Export clusters
        for cid, cluster in self.clusters.items():
            data["clusters"][cid] = cluster.to_dict()
        
        # Export unification events
        for event in self.unification_events:
            data["unification_events"].append(event.to_dict())
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data exported to {output_file}")
    
    def import_data(self, input_file: str):
        """
        Import tracker data from JSON.
        
        Args:
            input_file: File to load the data from
        """
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Clear current data
            self.beliefs = {}
            self.meta_beliefs = {}
            self.clusters = {}
            self.unification_events = []
            self.belief_graph = nx.DiGraph()
            
            # Import beliefs
            for bid, belief_data in data["beliefs"].items():
                belief = Belief.from_dict(belief_data)
                self.beliefs[bid] = belief
                self.belief_graph.add_node(bid, belief=belief)
                if belief.is_meta:
                    self.meta_beliefs[bid] = belief
            
            # Import clusters
            for cid, cluster_data in data["clusters"].items():
                self.clusters[cid] = BeliefCluster.from_dict(cluster_data)
            
            # Import unification events
            for event_data in data["unification_events"]:
                self.unification_events.append(MetaUnificationEvent.from_dict(event_data))
            
            # Import history data
            if "system_entropy_history" in data:
                self.system_entropy_history = data["system_entropy_history"]
            
            if "meta_belief_count_history" in data:
                self.meta_belief_count_history = data["meta_belief_count_history"]
            
            # Rebuild graph edges
            for bid, belief in self.beliefs.items():
                for influence_id in belief.influences:
                    if influence_id in self.beliefs:
                        self.belief_graph.add_edge(bid, influence_id, weight=0.7)
            
            logger.info(f"Data imported from {input_file}")
        except Exception as e:
            logger.error(f"Error importing data: {e}")
    
    def generate_dashboard(self, output_dir: str):
        """
        Generate a comprehensive dashboard for meta-belief analysis.
        
        Args:
            output_dir: Directory to save dashboard files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate visualizations
        network_file = os.path.join(output_dir, "meta_belief_network.png")
        self.visualize_belief_network(output_file=network_file)
        
        # Generate individual meta-belief influence visualizations
        meta_viz_dir = os.path.join(output_dir, "meta_influences")
        if not os.path.exists(meta_viz_dir):
            os.makedirs(meta_viz_dir)
        
        for mid in list(self.meta_beliefs.keys())[:5]:  # Limit to 5 for brevity
            meta_file = os.path.join(meta_viz_dir, f"meta_{mid}.png")
            self.visualize_meta_influence(mid, output_file=meta_file)
        
        # Generate report
        report_file = os.path.join(output_dir, "meta_report.txt")
        self.generate_report(output_file=report_file)
        
        # Export data
        data_file = os.path.join(output_dir, "meta_data.json")
        self.export_data(data_file)
        
        # Generate HTML dashboard
        html_file = os.path.join(output_dir, "meta_dashboard.html")
        
        # Get summary statistics for the dashboard
        summary, df = self.generate_report()
        
        # Generate events table HTML
        events_html = ""
        for event in self.unification_events[:10]:  # Limit to 10 most recent
            meta_content = self.beliefs[event.meta_belief_id].content if event.meta_belief_id in self.beliefs else "Unknown"
            events_html += f"""
            <tr>
                <td>{event.meta_belief_id}</td>
                <td>{meta_content[:50]}...</td>
                <td>{len(event.unified_beliefs)}</td>
                <td>{event.unification_strength:.3f}</td>
                <td>{event.entropy_before:.3f} → {event.entropy_after:.3f}</td>
                <td>{datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
            """
        
        # Generate top meta-beliefs HTML
        meta_html = ""
        for _, row in df.head(5).iterrows():
            meta_html += f"""
            <tr>
                <td>{row['meta_belief_id']}</td>
                <td>{row['content'][:50]}...</td>
                <td>{row['unification_strength']:.3f}</td>
                <td>{row['influence_count']}</td>
                <td>{row['entropy_reduction']:.3f}</td>
            </tr>
            """
        
        # Create HTML
        with open(html_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Meta-Belief Unification Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #4a148c; color: white; padding: 20px; margin-bottom: 20px; }}
        .viz-container {{ margin-bottom: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; margin-right: 15px; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .meta-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .meta-table th, .meta-table td {{ padding: 10px; border: 1px solid #ddd; }}
        .meta-table th {{ background-color: #f2f2f2; }}
        .events-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .events-table th, .events-table td {{ padding: 10px; border: 1px solid #ddd; }}
        .events-table th {{ background-color: #f2f2f2; }}
        .timestamp {{ font-size: 12px; color: #666; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Meta-Belief Unification Dashboard</h1>
            <p>Tracking the emergence of unifying meta-beliefs and their effect on belief system coherence</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{summary['total_meta_beliefs']}</div>
                <div class="metric-label">Meta-Beliefs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['total_beliefs']}</div>
                <div class="metric-label">Total Beliefs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['avg_unification_strength']:.3f}</div>
                <div class="metric-label">Avg Unification Strength</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['emergence_rate']:.3f}</div>
                <div class="metric-label">Meta-Belief Emergence Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['current_system_entropy']:.3f}</div>
                <div class="metric-label">Current System Entropy</div>
            </div>
        </div>
        
        <div class="viz-container">
            <h2>Meta-Belief Network</h2>
            <img src="meta_belief_network.png" alt="Meta-Belief Network" style="max-width: 100%;" />
        </div>
        
        <div>
            <h2>Top Meta-Beliefs by Unification Strength</h2>
            <table class="meta-table">
                <tr>
                    <th>ID</th>
                    <th>Content</th>
                    <th>Unification Strength</th>
                    <th>Influenced Beliefs</th>
                    <th>Entropy Reduction</th>
                </tr>
                {meta_html}
            </table>
        </div>
        
        <div>
            <h2>Recent Unification Events</h2>
            <table class="events-table">
                <tr>
                    <th>Meta-Belief ID</th>
                    <th>Content</th>
                    <th>Unified Beliefs</th>
                    <th>Strength</th>
                    <th>Entropy Change</th>
                    <th>Timestamp</th>
                </tr>
                {events_html}
            </table>
        </div>
        
        <div class="viz-container">
            <h2>Sample Meta-Belief Influence Networks</h2>
            <div style="display: flex; flex-wrap: wrap;">
                {"".join([f'<div style="flex: 1; min-width: 300px; margin: 10px;"><img src="meta_influences/meta_{mid}.png" alt="Meta-Belief {mid}" style="width: 100%;" /><p>Meta-Belief: {mid}</p></div>' for mid in list(self.meta_beliefs.keys())[:3]])}
            </div>
        </div>
        
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>""")
        
        logger.info(f"Dashboard generated in {output_dir}")


def main():
    """Main function to demonstrate the meta-belief unification tracker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Belief Unification Tracker")
    parser.add_argument("--vector-dim", type=int, default=128, help="Vector dimension for beliefs")
    parser.add_argument("--beliefs", type=int, default=50, help="Number of beliefs for mock data")
    parser.add_argument("--import-data", type=str, help="Import data from JSON file")
    parser.add_argument("--export-data", type=str, help="Export data to JSON file")
    parser.add_argument("--output-dir", type=str, help="Directory to save dashboard")
    parser.add_argument("--report", type=str, help="Generate and save report to file")
    
    args = parser.parse_args()
    
    logger.info("Initializing Meta-Belief Unification Tracker...")
    tracker = MetaUnificationTracker(vector_dim=args.vector_dim)
    
    # Import data if specified
    if args.import_data:
        logger.info(f"Importing data from {args.import_data}...")
        tracker.import_data(args.import_data)
    else:
        # Create mock data with specified number of beliefs
        logger.info(f"Creating mock data with {args.beliefs} beliefs...")
        tracker._create_mock_data(args.beliefs)
    
    # Detect potential meta-beliefs
    logger.info("Detecting potential meta-beliefs...")
    new_meta_beliefs = tracker.detect_meta_beliefs()
    if new_meta_beliefs:
        logger.info(f"Detected {len(new_meta_beliefs)} new meta-beliefs")
    
    # Generate report if specified
    if args.report:
        logger.info(f"Generating report to {args.report}...")
        tracker.generate_report(args.report)
    
    # Export data if specified
    if args.export_data:
        logger.info(f"Exporting data to {args.export_data}...")
        tracker.export_data(args.export_data)
    
    # Generate dashboard if specified
    if args.output_dir:
        logger.info(f"Generating dashboard in {args.output_dir}...")
        tracker.generate_dashboard(args.output_dir)
    else:
        # Just show the visualization
        logger.info("Visualizing belief network...")
        tracker.visualize_belief_network()
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 