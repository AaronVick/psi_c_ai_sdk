"""
Schema Fork Detector (Post-Mutation Splits)
------------------------------------------

This module detects when a schema mutation has functionally forked the agent's
identity by tracking and analyzing schema embeddings over time.

The fork detector compares the current schema embedding with the historical
trajectory to identify significant deviations that may indicate an identity 
divergence or functional fork.

Mathematical basis:
- Fork detection via clustering:
  - Compare current schema embedding with historical trajectory:
    Δ_trajectory = ||Σ_t - E[Σ_(t-1..t-n)]||
  - Flag if drift exceeds historical entropy bounds:
    Δ > 3σ_Σ
"""

import logging
import os
import math
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json
import warnings

# Setup logging
logger = logging.getLogger(__name__)


class SchemaForkDetector:
    """
    Detector for identifying functional forks in an agent's identity after schema mutations.
    
    The detector tracks schema embeddings over time and uses statistical methods to
    detect when a schema has diverged significantly from its historical trajectory,
    which may indicate a functional fork in the agent's identity.
    """
    
    def __init__(
        self, 
        history_window: int = 10,
        drift_threshold: float = 3.0,  # Number of standard deviations
        min_history_required: int = 3,
        embedding_dim: int = 64,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the schema fork detector.
        
        Args:
            history_window: Number of historical schema states to consider
            drift_threshold: Threshold multiplier for standard deviation
            min_history_required: Minimum number of history points needed before detection
            embedding_dim: Dimension of the schema embedding vector
            data_dir: Directory to store fork detection data
        """
        self.history_window = history_window
        self.drift_threshold = drift_threshold
        self.min_history_required = min_history_required
        self.embedding_dim = embedding_dim
        self.data_dir = data_dir or self._get_default_data_dir()
        
        # Internal state
        self.schema_embeddings = []  # List of (timestamp, embedding) tuples
        self.detected_forks = []     # List of detected forks
        self.current_fork_id = None  # ID of the current fork branch
        
        # Statistics
        self.mean_vector = None      # Historical mean embedding vector
        self.std_vector = None       # Historical std deviation of embedding
        self.trajectory = []         # Drift measurements over time
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_default_data_dir(self) -> str:
        """Get the default directory for storing fork detection data."""
        # First check if there's a data directory in the package
        package_dir = Path(__file__).parent.parent
        data_dir = package_dir / "data" / "fork_detection"
        if data_dir.exists():
            return str(data_dir)
        
        # If not, use a directory in the user's home
        home_dir = Path.home() / ".psi_c_ai_sdk" / "data" / "fork_detection"
        home_dir.mkdir(parents=True, exist_ok=True)
        return str(home_dir)

    def compute_schema_embedding(self, agent) -> np.ndarray:
        """
        Compute an embedding vector for the agent's current schema.
        
        Args:
            agent: The ΨC agent to analyze
            
        Returns:
            Embedding vector as numpy array
        """
        # Extract schema graph from agent
        schema_graph = self._extract_schema_graph(agent)
        
        # Compute embedding
        embedding = self._embed_schema_graph(schema_graph)
        
        return embedding

    def _extract_schema_graph(self, agent) -> nx.Graph:
        """
        Extract schema graph from the agent.
        
        Args:
            agent: The ΨC agent to analyze
            
        Returns:
            NetworkX graph representing the schema
        """
        try:
            # Try different approaches based on agent implementation
            if hasattr(agent, 'schema_graph'):
                return agent.schema_graph
            
            elif hasattr(agent, 'get_schema_graph'):
                return agent.get_schema_graph()
            
            elif hasattr(agent, 'schema') and hasattr(agent.schema, 'graph'):
                return agent.schema.graph
                
            else:
                # Generate mock data for testing
                logger.warning("Using mock schema - couldn't access agent schema")
                return self._generate_mock_schema()
                
        except Exception as e:
            logger.error(f"Failed to extract schema graph: {e}")
            return nx.Graph()  # Return empty graph as fallback

    def _embed_schema_graph(self, graph: nx.Graph) -> np.ndarray:
        """
        Compute an embedding vector for a schema graph.
        
        This embedding captures the structural and semantic properties of the schema
        in a fixed-length vector that can be compared across time.
        
        Args:
            graph: NetworkX graph representing the schema
            
        Returns:
            Embedding vector as numpy array
        """
        # Empty graph handling
        if len(graph) == 0:
            return np.zeros(self.embedding_dim)
        
        # Initialize embedding vector
        embedding = np.zeros(self.embedding_dim)
        
        # Extract graph features (using multiple approaches for robustness)
        
        # 1. Basic graph statistics
        n_nodes = len(graph)
        n_edges = len(graph.edges())
        
        # Normalize by log to handle different scales
        embedding[0] = math.log(n_nodes + 1)
        embedding[1] = math.log(n_edges + 1)
        
        # 2. Connectivity metrics
        if n_nodes > 1:
            try:
                avg_clustering = nx.average_clustering(graph)
                embedding[2] = avg_clustering
            except:
                embedding[2] = 0.0
                
            try:
                # Use approximate values for larger graphs
                if n_nodes > 1000:
                    avg_path_length = nx.approximation.average_shortest_path_length(graph)
                else:
                    # Handle disconnected graphs
                    if nx.is_connected(graph):
                        avg_path_length = nx.average_shortest_path_length(graph)
                    else:
                        # Use the largest connected component
                        largest_cc = max(nx.connected_components(graph), key=len)
                        largest_subgraph = graph.subgraph(largest_cc)
                        avg_path_length = nx.average_shortest_path_length(largest_subgraph)
                        
                # Normalize path length
                embedding[3] = min(1.0, avg_path_length / 10.0)
            except:
                embedding[3] = 0.5  # Default value
        
        # 3. Centrality distributions
        try:
            degree_centrality = list(nx.degree_centrality(graph).values())
            embedding[4] = np.mean(degree_centrality) if degree_centrality else 0
            embedding[5] = np.std(degree_centrality) if len(degree_centrality) > 1 else 0
        except:
            embedding[4] = 0.0
            embedding[5] = 0.0
            
        try:
            # Use approximate eigenvector centrality for large graphs
            if n_nodes > 1000:
                eigenvector_centrality = list(nx.eigenvector_centrality_numpy(graph).values())
            else:
                eigenvector_centrality = list(nx.eigenvector_centrality(graph).values())
                
            embedding[6] = np.mean(eigenvector_centrality) if eigenvector_centrality else 0
            embedding[7] = np.std(eigenvector_centrality) if len(eigenvector_centrality) > 1 else 0
        except:
            embedding[6] = 0.0
            embedding[7] = 0.0
        
        # 4. Community structure
        try:
            communities = list(nx.community.greedy_modularity_communities(graph))
            embedding[8] = len(communities) / max(1, math.sqrt(n_nodes))
            embedding[9] = nx.community.modularity(graph, communities)
        except:
            embedding[8] = 1.0  # Default: one community
            embedding[9] = 0.0  # Default: no modularity
        
        # 5. Schema node type distributions (if available)
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Encode node type distribution in embedding
        type_index = 10
        for i, count in enumerate(sorted(node_types.values(), reverse=True)):
            if type_index + i < self.embedding_dim:
                embedding[type_index + i] = count / n_nodes
            else:
                break
        
        # 6. Edge weight statistics (if available)
        weights = []
        for _, _, data in graph.edges(data=True):
            if 'weight' in data:
                weights.append(data['weight'])
        
        if weights:
            weight_index = type_index + len(node_types)
            if weight_index < self.embedding_dim - 1:
                embedding[weight_index] = np.mean(weights)
                embedding[weight_index + 1] = np.std(weights) if len(weights) > 1 else 0
        
        # 7. Graph spectrum features
        try:
            # Use top-k eigenvalues of adjacency matrix for spectral fingerprint
            if n_nodes > 1000:
                # For large graphs, use approximate methods
                k = min(10, n_nodes - 1)
                laplacian = nx.normalized_laplacian_matrix(graph)
                eigenvalues = np.linalg.eigvals(laplacian.toarray())
                eigenvalues.sort()
                spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
            else:
                laplacian = nx.normalized_laplacian_matrix(graph).toarray()
                eigenvalues = np.linalg.eigvalsh(laplacian)
                eigenvalues.sort()
                spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
            
            # Spectral gap is informative about graph structure
            embedding[self.embedding_dim - 2] = spectral_gap
        except:
            embedding[self.embedding_dim - 2] = 0.0
        
        # 8. Hash-based stability marker
        try:
            # Use a deterministic graph hash as a stability marker
            graph_hash = hash(str(sorted(graph.edges())))
            embedding[self.embedding_dim - 1] = (graph_hash % 1000) / 1000.0
        except:
            embedding[self.embedding_dim - 1] = 0.0
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

    def _generate_mock_schema(self) -> nx.Graph:
        """Generate a mock schema graph for testing."""
        G = nx.Graph()
        
        # Add nodes with types
        node_types = ['belief', 'memory', 'concept', 'relation', 'rule']
        for i in range(30):
            node_type = node_types[i % len(node_types)]
            G.add_node(f"node_{i}", type=node_type, importance=np.random.uniform(0.3, 0.9))
        
        # Add edges with weights
        for i in range(30):
            # Connect to several random nodes
            for _ in range(3):
                j = np.random.randint(0, 30)
                if i != j:
                    G.add_edge(f"node_{i}", f"node_{j}", weight=np.random.uniform(0.5, 1.0))
        
        return G

    def add_schema_snapshot(self, agent, timestamp=None, embedding=None):
        """
        Add a snapshot of the agent's current schema to the history.
        
        Args:
            agent: The ΨC agent to analyze
            timestamp: Optional timestamp (defaults to current time)
            embedding: Pre-computed embedding (computed if not provided)
            
        Returns:
            Tuple of (embedding, drift_value)
        """
        # Use current time if not provided
        if timestamp is None:
            timestamp = datetime.now()
            
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.compute_schema_embedding(agent)
        
        # Add to history
        self.schema_embeddings.append((timestamp, embedding))
        
        # Trim history if needed
        if len(self.schema_embeddings) > self.history_window:
            self.schema_embeddings = self.schema_embeddings[-self.history_window:]
        
        # Update statistics
        self._update_statistics()
        
        # Calculate current drift
        drift = self._calculate_drift(embedding)
        
        # Add to trajectory
        self.trajectory.append((timestamp, drift))
        
        # Check for fork
        if self._check_for_fork(drift):
            self._record_fork(agent, timestamp, embedding, drift)
            logger.warning(f"FORK DETECTED: Schema drift of {drift:.4f} exceeds threshold")
        
        return embedding, drift

    def _update_statistics(self):
        """Update statistical measures based on embedding history."""
        if len(self.schema_embeddings) < 2:
            return
            
        # Extract embeddings
        embeddings = [e for _, e in self.schema_embeddings]
        
        # Calculate mean embedding vector
        self.mean_vector = np.mean(embeddings, axis=0)
        
        # Calculate standard deviation
        if len(embeddings) > 1:
            self.std_vector = np.std(embeddings, axis=0)
        else:
            self.std_vector = np.zeros_like(self.mean_vector)

    def _calculate_drift(self, embedding: np.ndarray) -> float:
        """
        Calculate drift from historical trajectory.
        
        Args:
            embedding: Current schema embedding
            
        Returns:
            Drift value (distance from expected trajectory)
        """
        if self.mean_vector is None or len(self.schema_embeddings) < self.min_history_required:
            return 0.0
        
        # Calculate Euclidean distance from mean, normalized by std
        # This is the Mahalanobis distance when using diagonal covariance
        delta = embedding - self.mean_vector
        
        # Handle zeros in std_vector to avoid division by zero
        std_safe = np.copy(self.std_vector)
        std_safe[std_safe == 0] = 1.0  # Replace zeros with ones
        
        # Normalized deviation in each dimension
        normalized_delta = delta / std_safe
        
        # Euclidean norm of normalized delta
        drift = np.linalg.norm(normalized_delta)
        
        return drift

    def _check_for_fork(self, drift: float) -> bool:
        """
        Check if current drift indicates a fork.
        
        Args:
            drift: Current drift value
            
        Returns:
            True if a fork is detected, False otherwise
        """
        # Need minimum history before detection
        if len(self.schema_embeddings) < self.min_history_required:
            return False
            
        # Check if drift exceeds threshold
        return drift > self.drift_threshold

    def _record_fork(self, agent, timestamp, embedding, drift):
        """
        Record a detected fork.
        
        Args:
            agent: The agent being analyzed
            timestamp: Time of fork detection
            embedding: Schema embedding at fork point
            drift: Measured drift value
        """
        # Generate fork ID
        fork_id = f"fork_{len(self.detected_forks) + 1}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        
        # Create fork record
        fork = {
            "id": fork_id,
            "timestamp": timestamp.isoformat(),
            "drift": float(drift),
            "threshold": float(self.drift_threshold),
            "previous_fork": self.current_fork_id,
            "schema_stats": self._get_schema_stats(agent)
        }
        
        # Update current fork ID
        self.current_fork_id = fork_id
        
        # Add to detected forks
        self.detected_forks.append(fork)
        
        # Save fork data
        self._save_fork_data(fork, embedding)
        
    def _get_schema_stats(self, agent) -> Dict:
        """
        Extract schema statistics for the fork record.
        
        Args:
            agent: The agent being analyzed
            
        Returns:
            Dictionary of schema statistics
        """
        # Extract schema graph
        schema_graph = self._extract_schema_graph(agent)
        
        stats = {
            "nodes": len(schema_graph),
            "edges": len(schema_graph.edges()),
            "density": nx.density(schema_graph),
            "connected_components": nx.number_connected_components(schema_graph)
        }
        
        # Try to get node type distribution
        node_types = {}
        for node, data in schema_graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        stats["node_types"] = node_types
        
        return stats

    def _save_fork_data(self, fork: Dict, embedding: np.ndarray):
        """
        Save fork data to disk.
        
        Args:
            fork: Fork record dictionary
            embedding: Schema embedding at fork point
        """
        # Create directory for this fork
        fork_dir = os.path.join(self.data_dir, fork["id"])
        os.makedirs(fork_dir, exist_ok=True)
        
        # Save fork metadata
        with open(os.path.join(fork_dir, "metadata.json"), 'w') as f:
            json.dump(fork, f, indent=2)
            
        # Save embedding
        np.save(os.path.join(fork_dir, "embedding.npy"), embedding)
        
        # Save trajectory data up to this point
        trajectory_data = [
            {"timestamp": t.isoformat(), "drift": float(d)}
            for t, d in self.trajectory
        ]
        with open(os.path.join(fork_dir, "trajectory.json"), 'w') as f:
            json.dump(trajectory_data, f, indent=2)

    def get_drift_history(self) -> List[Tuple[datetime, float]]:
        """
        Get the history of drift measurements.
        
        Returns:
            List of (timestamp, drift) tuples
        """
        return self.trajectory

    def get_recent_drift(self) -> float:
        """
        Get the most recent drift measurement.
        
        Returns:
            Most recent drift value, or 0.0 if no measurements
        """
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1][1]

    def get_fork_history(self) -> List[Dict]:
        """
        Get the history of detected forks.
        
        Returns:
            List of fork records
        """
        return self.detected_forks

    def reset_history(self):
        """Reset the detector's history."""
        self.schema_embeddings = []
        self.trajectory = []
        self.mean_vector = None
        self.std_vector = None
        # Note: We keep detected_forks and current_fork_id for reference

    def monitor_agent(self, agent, reset_on_fork=False):
        """
        Monitor an agent for potential schema forks.
        
        Args:
            agent: The ΨC agent to monitor
            reset_on_fork: Whether to reset history when a fork is detected
            
        Returns:
            Tuple of (drift, is_fork)
        """
        # Compute embedding and check drift
        embedding, drift = self.add_schema_snapshot(agent)
        
        # Check if fork was detected
        is_fork = False
        if self.detected_forks and self.detected_forks[-1]["id"] == self.current_fork_id:
            is_fork = True
            
            # Reset history if configured to do so
            if reset_on_fork:
                self.reset_history()
                # Keep the current embedding as first point in new history
                self.schema_embeddings = [(datetime.now(), embedding)]
                self._update_statistics()
        
        return drift, is_fork

    def visualize_trajectory(self, output_path=None, show=True):
        """
        Visualize the drift trajectory and detected forks.
        
        Args:
            output_path: Path to save the visualization (optional)
            show: Whether to display the plot
            
        Returns:
            Path to saved visualization if output_path is provided
        """
        if not self.trajectory:
            logger.warning("No trajectory data to visualize")
            return None
            
        # Extract data
        timestamps = [t for t, _ in self.trajectory]
        drifts = [d for _, d in self.trajectory]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot trajectory
        plt.plot(timestamps, drifts, 'b-', linewidth=2, label='Drift')
        
        # Plot threshold
        plt.axhline(y=self.drift_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.drift_threshold}σ)')
        
        # Mark forks
        fork_timestamps = []
        fork_drifts = []
        
        for fork in self.detected_forks:
            # Convert ISO timestamp string to datetime
            fork_time = datetime.fromisoformat(fork["timestamp"])
            
            # Find closest trajectory point
            closest_idx = min(range(len(timestamps)), 
                             key=lambda i: abs((timestamps[i] - fork_time).total_seconds()))
            
            fork_timestamps.append(timestamps[closest_idx])
            fork_drifts.append(drifts[closest_idx])
        
        if fork_timestamps:
            plt.plot(fork_timestamps, fork_drifts, 'ro', markersize=10, label='Forks')
        
        # Labels and formatting
        plt.title('Schema Drift Trajectory')
        plt.xlabel('Time')
        plt.ylabel('Drift (σ)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Format x-axis to show readable dates
        plt.gcf().autofmt_xdate()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory visualization to {output_path}")
        
        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path if output_path else None

    def visualize_embedding_pca(self, output_path=None, show=True):
        """
        Visualize the schema embeddings using PCA projection.
        
        Args:
            output_path: Path to save the visualization (optional)
            show: Whether to display the plot
            
        Returns:
            Path to saved visualization if output_path is provided
        """
        if len(self.schema_embeddings) < 2:
            logger.warning("Not enough embedding data for PCA visualization")
            return None
            
        try:
            from sklearn.decomposition import PCA
            
            # Extract embeddings
            embeddings = [e for _, e in self.schema_embeddings]
            timestamps = [t for t, _ in self.schema_embeddings]
            
            # Apply PCA
            pca = PCA(n_components=2)
            projected = pca.fit_transform(embeddings)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot embeddings as a trajectory
            plt.plot(projected[:, 0], projected[:, 1], 'b-', alpha=0.5)
            
            # Plot each point with timestamp label
            for i, (time, _) in enumerate(self.schema_embeddings):
                plt.scatter(projected[i, 0], projected[i, 1], c='blue', s=50)
                plt.annotate(time.strftime("%H:%M:%S"), 
                            (projected[i, 0], projected[i, 1]),
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center')
            
            # Add current mean
            if self.mean_vector is not None:
                mean_projected = pca.transform([self.mean_vector])[0]
                plt.scatter(mean_projected[0], mean_projected[1], c='red', s=100, marker='*')
                plt.annotate("Mean", 
                            (mean_projected[0], mean_projected[1]),
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center',
                            color='red',
                            weight='bold')
            
            # Labels and formatting
            plt.title('Schema Embedding Evolution (PCA Projection)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save if output path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved PCA visualization to {output_path}")
            
            # Show plot if requested
            if show:
                plt.show()
            else:
                plt.close()
                
            return output_path if output_path else None
            
        except ImportError:
            logger.warning("scikit-learn is required for PCA visualization")
            return None
        except Exception as e:
            logger.error(f"Error in PCA visualization: {e}")
            return None

    def save_state(self, filepath=None):
        """
        Save the detector's state to a file.
        
        Args:
            filepath: Path to save state (optional)
            
        Returns:
            Path to saved state file
        """
        if filepath is None:
            filepath = os.path.join(self.data_dir, f"fork_detector_state_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
            
        # Prepare state
        state = {
            "history_window": self.history_window,
            "drift_threshold": self.drift_threshold,
            "min_history_required": self.min_history_required,
            "embedding_dim": self.embedding_dim,
            "current_fork_id": self.current_fork_id,
            "detected_forks": self.detected_forks,
            "trajectory": [
                {"timestamp": t.isoformat(), "drift": float(d)}
                for t, d in self.trajectory
            ],
            "schema_embeddings": [
                {"timestamp": t.isoformat(), "embedding_idx": i}
                for i, (t, _) in enumerate(self.schema_embeddings)
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save embeddings separately to handle numpy arrays
        embeddings_path = os.path.splitext(filepath)[0] + "_embeddings.npy"
        embeddings = np.array([e for _, e in self.schema_embeddings])
        if len(embeddings) > 0:
            np.save(embeddings_path, embeddings)
        
        # Save state
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved fork detector state to {filepath}")
        return filepath
        
    def load_state(self, filepath):
        """
        Load the detector's state from a file.
        
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
            self.history_window = state["history_window"]
            self.drift_threshold = state["drift_threshold"]
            self.min_history_required = state["min_history_required"]
            self.embedding_dim = state["embedding_dim"]
            self.current_fork_id = state["current_fork_id"]
            self.detected_forks = state["detected_forks"]
            
            # Load trajectory
            self.trajectory = [
                (datetime.fromisoformat(item["timestamp"]), item["drift"])
                for item in state["trajectory"]
            ]
            
            # Load embeddings
            embeddings_path = os.path.splitext(filepath)[0] + "_embeddings.npy"
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path)
                
                # Reconstruct schema_embeddings list
                self.schema_embeddings = []
                for item in state["schema_embeddings"]:
                    timestamp = datetime.fromisoformat(item["timestamp"])
                    idx = item["embedding_idx"]
                    if idx < len(embeddings):
                        self.schema_embeddings.append((timestamp, embeddings[idx]))
                
                # Update statistics
                self._update_statistics()
                
            logger.info(f"Loaded fork detector state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fork detector state: {e}")
            return False
            
    def get_metrics(self):
        """
        Get metrics about the fork detector's state.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "history_points": len(self.schema_embeddings),
            "detected_forks": len(self.detected_forks),
            "current_fork": self.current_fork_id,
            "current_drift": self.get_recent_drift(),
            "max_drift": max([d for _, d in self.trajectory]) if self.trajectory else 0.0,
            "avg_drift": np.mean([d for _, d in self.trajectory]) if self.trajectory else 0.0
        }


def analyze_agent_schema_fork(agent, visualize=True, save_results=False, output_dir=None):
    """
    Utility function to analyze potential schema forks in an agent.
    
    Args:
        agent: ΨC agent to analyze
        visualize: Whether to visualize results
        save_results: Whether to save results to disk
        output_dir: Directory to save results (optional)
        
    Returns:
        Tuple of (drift, is_fork, detector)
    """
    # Initialize detector
    detector = SchemaForkDetector()
    
    # Monitor for potential fork
    drift, is_fork = detector.monitor_agent(agent)
    
    # Print summary
    print(f"\nSchema Fork Analysis:")
    print(f"  Current drift: {drift:.3f}")
    print(f"  Drift threshold: {detector.drift_threshold:.3f}")
    print(f"  Fork detected: {is_fork}")
    
    # Show detailed information for detected forks
    forks = detector.get_fork_history()
    if forks:
        print(f"\nDetected Forks ({len(forks)}):")
        for fork in forks:
            print(f"  ID: {fork['id']}")
            print(f"  Timestamp: {fork['timestamp']}")
            print(f"  Drift: {fork['drift']:.3f}")
            print(f"  Schema nodes: {fork['schema_stats']['nodes']}")
            print(f"  Schema edges: {fork['schema_stats']['edges']}")
            print()
    
    # Visualize if requested
    if visualize:
        detector.visualize_trajectory(
            output_path=os.path.join(output_dir, "schema_drift.png") if output_dir and save_results else None
        )
        
        # PCA visualization if at least 3 points available
        if len(detector.schema_embeddings) >= 3:
            detector.visualize_embedding_pca(
                output_path=os.path.join(output_dir, "schema_embedding_pca.png") if output_dir and save_results else None
            )
    
    # Save results if requested
    if save_results:
        if output_dir is None:
            # Create a results directory
            output_dir = os.path.join(detector.data_dir, f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
        # Save detector state
        detector.save_state(os.path.join(output_dir, "fork_detector_state.json"))
    
    return drift, is_fork, detector


if __name__ == "__main__":
    # Simple demo with mock agent
    class MockAgent:
        def __init__(self):
            self.schema_graph = None
    
    # Create a mock agent
    agent = MockAgent()
    
    # Analyze for potential schema fork
    drift, is_fork, detector = analyze_agent_schema_fork(agent) 