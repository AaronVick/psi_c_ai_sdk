"""
ΨC Schema Compression Benchmark

This module implements tools for measuring the effectiveness of schema compression techniques
and their impact on ΨC scores, coherence, and functional capabilities. It provides methods
to evaluate the tradeoffs between memory efficiency and cognitive performance.

Key features:
- Compression effect measurement on ΨC scores
- Analysis of information loss during compression
- Benchmark suite for comparing compression algorithms
- Memory-performance tradeoff evaluation
- Visualization of compression effects
"""

import logging
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

# Internal imports
try:
    from psi_c_ai_sdk.core.trace_context import TraceContext
except ImportError:
    # For standalone testing
    class TraceContext:
        """Mock TraceContext for testing."""
        def __init__(self):
            self.trace_id = "mock"
            self.psi_c_score = 0.5

@dataclass
class CompressionResult:
    """Results of a schema compression operation."""
    algorithm_name: str
    original_size: int  # Number of nodes/edges before compression
    compressed_size: int  # Number of nodes/edges after compression
    compression_ratio: float  # original_size / compressed_size
    
    pre_compression_psi_c: float
    post_compression_psi_c: float
    psi_c_delta: float
    
    pre_compression_coherence: float
    post_compression_coherence: float
    coherence_delta: float
    
    execution_time_ms: float
    memory_usage_kb: float
    
    # Performance metrics on functional tests
    accuracy_scores: Dict[str, float] = field(default_factory=dict)
    
    # Details about what was compressed
    removed_nodes: List[str] = field(default_factory=list)
    merged_nodes: List[Tuple[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "algorithm_name": self.algorithm_name,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "pre_compression_psi_c": self.pre_compression_psi_c,
            "post_compression_psi_c": self.post_compression_psi_c,
            "psi_c_delta": self.psi_c_delta,
            "pre_compression_coherence": self.pre_compression_coherence,
            "post_compression_coherence": self.post_compression_coherence,
            "coherence_delta": self.coherence_delta,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_kb": self.memory_usage_kb,
            "accuracy_scores": self.accuracy_scores,
            "removed_nodes": self.removed_nodes,
            "merged_nodes": self.merged_nodes
        }

@dataclass
class CompressionAlgorithm:
    """Defines a schema compression algorithm."""
    name: str
    description: str
    compress_function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __call__(self, *args, **kwargs):
        """Call the compress function with the stored parameters."""
        all_kwargs = {**self.parameters, **kwargs}
        return self.compress_function(*args, **all_kwargs)

class SchemaCompressionEval:
    """
    Benchmark for evaluating schema compression techniques.
    
    This class provides tools to measure the impact of compression on
    schema coherence, ΨC scores, and functional capabilities.
    """
    
    def __init__(self, trace_context: Optional[TraceContext] = None):
        """
        Initialize the schema compression evaluator.
        
        Args:
            trace_context: Optional trace context for logging
        """
        self.trace_context = trace_context or TraceContext()
        self.compression_algorithms: Dict[str, CompressionAlgorithm] = {}
        self.benchmark_results: List[CompressionResult] = []
        self.logger = logging.getLogger(__name__)
        
        # Register default compression algorithms
        self._register_default_algorithms()
    
    def _register_default_algorithms(self) -> None:
        """Register default compression algorithms."""
        
        # Edge pruning algorithm
        self.register_algorithm(
            name="edge_pruning",
            description="Removes edges with low weights to simplify the schema graph",
            compress_function=self._edge_pruning_algorithm,
            parameters={"threshold": 0.2}
        )
        
        # Node clustering algorithm
        self.register_algorithm(
            name="node_clustering",
            description="Clusters similar nodes and merges them to reduce graph size",
            compress_function=self._node_clustering_algorithm,
            parameters={"n_clusters": 5, "similarity_threshold": 0.8}
        )
        
        # Entropy-based pruning
        self.register_algorithm(
            name="entropy_pruning",
            description="Removes high-entropy, low-value nodes from the schema",
            compress_function=self._entropy_pruning_algorithm,
            parameters={"entropy_threshold": 0.7}
        )
    
    def _edge_pruning_algorithm(self, 
                               schema_graph: nx.Graph, 
                               threshold: float = 0.2,
                               **kwargs) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Simple edge pruning algorithm that removes edges with weight below threshold.
        
        Args:
            schema_graph: Input schema graph
            threshold: Weight threshold for pruning
            
        Returns:
            Compressed graph and metadata
        """
        compressed_graph = schema_graph.copy()
        
        # Find edges below threshold
        edges_to_remove = []
        for u, v, data in compressed_graph.edges(data=True):
            weight = data.get('weight', 0)
            if weight < threshold:
                edges_to_remove.append((u, v))
        
        # Remove edges
        compressed_graph.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(compressed_graph))
        compressed_graph.remove_nodes_from(isolated_nodes)
        
        metadata = {
            "removed_edges": edges_to_remove,
            "removed_nodes": isolated_nodes,
            "merged_nodes": []
        }
        
        return compressed_graph, metadata
    
    def _node_clustering_algorithm(self, 
                                  schema_graph: nx.Graph, 
                                  n_clusters: int = 5,
                                  similarity_threshold: float = 0.8,
                                  **kwargs) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Node clustering algorithm that groups similar nodes and merges them.
        
        Args:
            schema_graph: Input schema graph
            n_clusters: Number of clusters to form
            similarity_threshold: Threshold for merging nodes
            
        Returns:
            Compressed graph and metadata
        """
        # Create node feature matrix from node attributes
        nodes = list(schema_graph.nodes())
        
        # Simple feature extraction - use node degree and clustering coefficient
        features = []
        for node in nodes:
            degree = schema_graph.degree(node)
            clustering = nx.clustering(schema_graph, node)
            features.append([degree, clustering])
        
        feature_matrix = np.array(features)
        
        # Normalize features
        if len(feature_matrix) > 0:
            feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-10)
            
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(nodes)), random_state=42)
            clusters = kmeans.fit_predict(feature_matrix)
            
            # Create compressed graph by merging nodes in the same cluster
            compressed_graph = nx.Graph()
            
            # Create cluster representative nodes
            cluster_nodes = {}
            for i in range(max(clusters) + 1):
                cluster_nodes[i] = f"cluster_{i}"
                compressed_graph.add_node(cluster_nodes[i])
            
            # Add edges between clusters
            merged_nodes = []
            for i, node1 in enumerate(nodes):
                cluster1 = clusters[i]
                for j, node2 in enumerate(nodes):
                    if i >= j:  # Avoid duplicates
                        continue
                    
                    cluster2 = clusters[j]
                    if schema_graph.has_edge(node1, node2):
                        # Add edge between cluster representatives if not already present
                        if not compressed_graph.has_edge(cluster_nodes[cluster1], cluster_nodes[cluster2]):
                            weight = schema_graph[node1][node2].get('weight', 1.0)
                            compressed_graph.add_edge(cluster_nodes[cluster1], cluster_nodes[cluster2], weight=weight)
                
                # Track which nodes were merged
                cluster_members = [nodes[j] for j, c in enumerate(clusters) if c == cluster1]
                if len(cluster_members) > 1:
                    for member in cluster_members:
                        if member != node1:
                            merged_nodes.append((node1, member))
            
            metadata = {
                "removed_edges": [],
                "removed_nodes": [],
                "merged_nodes": merged_nodes,
                "clusters": {i: [nodes[j] for j, c in enumerate(clusters) if c == i] for i in range(max(clusters) + 1)}
            }
            
            return compressed_graph, metadata
        
        # If no features, return original graph
        return schema_graph.copy(), {"removed_edges": [], "removed_nodes": [], "merged_nodes": []}
    
    def _entropy_pruning_algorithm(self, 
                                  schema_graph: nx.Graph, 
                                  entropy_threshold: float = 0.7,
                                  **kwargs) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Entropy-based pruning that removes high-entropy nodes.
        
        Args:
            schema_graph: Input schema graph
            entropy_threshold: Entropy threshold for pruning
            
        Returns:
            Compressed graph and metadata
        """
        compressed_graph = schema_graph.copy()
        
        # Calculate node entropy (using neighbor diversity as a proxy)
        node_entropy = {}
        for node in compressed_graph.nodes():
            neighbors = list(compressed_graph.neighbors(node))
            if not neighbors:
                node_entropy[node] = 0.0
                continue
            
            # Use edge weight distribution as entropy measure
            weights = [compressed_graph[node][neighbor].get('weight', 1.0) for neighbor in neighbors]
            weights_sum = sum(weights) + 1e-10
            weights_normalized = [w / weights_sum for w in weights]
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p + 1e-10) for p in weights_normalized)
            max_entropy = np.log2(len(weights) + 1e-10)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            node_entropy[node] = normalized_entropy
        
        # Remove high-entropy nodes
        nodes_to_remove = [node for node, entropy in node_entropy.items() 
                          if entropy > entropy_threshold]
        
        compressed_graph.remove_nodes_from(nodes_to_remove)
        
        metadata = {
            "removed_edges": [],
            "removed_nodes": nodes_to_remove,
            "merged_nodes": [],
            "node_entropy": node_entropy
        }
        
        return compressed_graph, metadata
    
    def register_algorithm(self, 
                          name: str, 
                          description: str, 
                          compress_function: Callable,
                          parameters: Dict[str, Any] = None) -> None:
        """
        Register a compression algorithm.
        
        Args:
            name: Algorithm name
            description: Algorithm description
            compress_function: Function that implements the algorithm
            parameters: Default parameters for the algorithm
        """
        self.compression_algorithms[name] = CompressionAlgorithm(
            name=name,
            description=description,
            compress_function=compress_function,
            parameters=parameters or {}
        )
        
        self.logger.info(f"Registered compression algorithm: {name}")
    
    def run_benchmark(self, 
                     schema_graph: nx.Graph,
                     psi_c_evaluator: Callable[[nx.Graph], float],
                     coherence_evaluator: Callable[[nx.Graph], float],
                     functional_tests: Dict[str, Callable[[nx.Graph], float]] = None,
                     algorithm_names: Optional[List[str]] = None) -> List[CompressionResult]:
        """
        Run compression benchmark on a schema graph.
        
        Args:
            schema_graph: Schema graph to compress
            psi_c_evaluator: Function to evaluate ΨC score on a graph
            coherence_evaluator: Function to evaluate coherence on a graph
            functional_tests: Dictionary of test name to test function
            algorithm_names: Names of algorithms to run (all if None)
            
        Returns:
            List of compression results
        """
        results = []
        
        # Select algorithms to run
        algorithms = []
        if algorithm_names:
            for name in algorithm_names:
                if name in self.compression_algorithms:
                    algorithms.append(self.compression_algorithms[name])
                else:
                    self.logger.warning(f"Algorithm not found: {name}")
        else:
            algorithms = list(self.compression_algorithms.values())
        
        # Get pre-compression metrics
        original_size = schema_graph.number_of_nodes() + schema_graph.number_of_edges()
        pre_psi_c = psi_c_evaluator(schema_graph)
        pre_coherence = coherence_evaluator(schema_graph)
        
        # Run pre-compression functional tests
        pre_func_scores = {}
        if functional_tests:
            for test_name, test_func in functional_tests.items():
                pre_func_scores[test_name] = test_func(schema_graph)
        
        # Run each algorithm
        for algorithm in algorithms:
            self.logger.info(f"Running compression algorithm: {algorithm.name}")
            
            try:
                # Measure execution time and memory usage
                start_time = time.time()
                # TODO: Add actual memory profiling if needed
                memory_usage = 0.0
                
                # Run compression
                compressed_graph, metadata = algorithm(schema_graph)
                
                # Calculate execution time
                execution_time = (time.time() - start_time) * 1000  # ms
                
                # Get post-compression metrics
                compressed_size = compressed_graph.number_of_nodes() + compressed_graph.number_of_edges()
                compression_ratio = original_size / max(1, compressed_size)
                
                post_psi_c = psi_c_evaluator(compressed_graph)
                post_coherence = coherence_evaluator(compressed_graph)
                
                # Run post-compression functional tests
                accuracy_scores = {}
                if functional_tests:
                    for test_name, test_func in functional_tests.items():
                        post_score = test_func(compressed_graph)
                        pre_score = pre_func_scores.get(test_name, 0)
                        # Calculate accuracy retention (higher is better)
                        accuracy_scores[test_name] = post_score / max(1e-10, pre_score)
                
                # Create result
                result = CompressionResult(
                    algorithm_name=algorithm.name,
                    original_size=original_size,
                    compressed_size=compressed_size,
                    compression_ratio=compression_ratio,
                    pre_compression_psi_c=pre_psi_c,
                    post_compression_psi_c=post_psi_c,
                    psi_c_delta=post_psi_c - pre_psi_c,
                    pre_compression_coherence=pre_coherence,
                    post_compression_coherence=post_coherence,
                    coherence_delta=post_coherence - pre_coherence,
                    execution_time_ms=execution_time,
                    memory_usage_kb=memory_usage,
                    accuracy_scores=accuracy_scores,
                    removed_nodes=metadata.get("removed_nodes", []),
                    merged_nodes=metadata.get("merged_nodes", [])
                )
                
                results.append(result)
                self.benchmark_results.append(result)
                
                self.logger.info(f"Compression results for {algorithm.name}: "
                                f"ratio={compression_ratio:.2f}, "
                                f"ΨC delta={result.psi_c_delta:.4f}, "
                                f"coherence delta={result.coherence_delta:.4f}")
            
            except Exception as e:
                self.logger.error(f"Error running algorithm {algorithm.name}: {e}")
        
        return results
    
    def visualize_results(self, 
                         results: Optional[List[CompressionResult]] = None,
                         output_file: Optional[str] = None) -> None:
        """
        Visualize benchmark results.
        
        Args:
            results: Results to visualize (uses stored results if None)
            output_file: Path to save visualization (shows interactive if None)
        """
        results_to_plot = results or self.benchmark_results
        
        if not results_to_plot:
            self.logger.warning("No results to visualize")
            return
        
        # Convert results to DataFrame for easier plotting
        data = []
        for result in results_to_plot:
            data.append({
                "Algorithm": result.algorithm_name,
                "Compression Ratio": result.compression_ratio,
                "ΨC Delta": result.psi_c_delta,
                "Coherence Delta": result.coherence_delta,
                "Execution Time (ms)": result.execution_time_ms
            })
        
        df = pd.DataFrame(data)
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Compression ratio by algorithm
        axes[0, 0].bar(df["Algorithm"], df["Compression Ratio"], color='blue', alpha=0.7)
        axes[0, 0].set_title("Compression Ratio by Algorithm")
        axes[0, 0].set_ylabel("Ratio (higher is better)")
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        axes[0, 0].set_xticklabels(df["Algorithm"], rotation=45, ha='right')
        
        # 2. ΨC and Coherence delta
        x = range(len(df))
        width = 0.35
        axes[0, 1].bar([i - width/2 for i in x], df["ΨC Delta"], width, label='ΨC Delta', color='green', alpha=0.7)
        axes[0, 1].bar([i + width/2 for i in x], df["Coherence Delta"], width, label='Coherence Delta', color='red', alpha=0.7)
        axes[0, 1].set_title("Impact on ΨC and Coherence")
        axes[0, 1].set_ylabel("Delta (higher is better)")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df["Algorithm"], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. Execution time
        axes[1, 0].bar(df["Algorithm"], df["Execution Time (ms)"], color='purple', alpha=0.7)
        axes[1, 0].set_title("Execution Time by Algorithm")
        axes[1, 0].set_ylabel("Time (ms) (lower is better)")
        axes[1, 0].set_xticklabels(df["Algorithm"], rotation=45, ha='right')
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 4. Tradeoff: Compression vs ΨC impact
        scatter = axes[1, 1].scatter(
            df["Compression Ratio"], 
            df["ΨC Delta"],
            s=100,
            c=df["Coherence Delta"],
            cmap="coolwarm",
            alpha=0.7
        )
        
        # Add algorithm labels to scatter points
        for i, alg in enumerate(df["Algorithm"]):
            axes[1, 1].annotate(
                alg,
                (df["Compression Ratio"].iloc[i], df["ΨC Delta"].iloc[i]),
                xytext=(5, 5),
                textcoords="offset points"
            )
        
        axes[1, 1].set_title("Tradeoff: Compression vs ΨC Impact")
        axes[1, 1].set_xlabel("Compression Ratio (higher is better)")
        axes[1, 1].set_ylabel("ΨC Delta (higher is better)")
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Add a colorbar for coherence delta
        cbar = fig.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label("Coherence Delta")
        
        # Add an overall title
        plt.suptitle("Schema Compression Benchmark Results", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def save_results(self, file_path: str) -> None:
        """
        Save benchmark results to a JSON file.
        
        Args:
            file_path: Path to save results
        """
        try:
            results_data = [result.to_dict() for result in self.benchmark_results]
            
            with open(file_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            self.logger.info(f"Saved {len(results_data)} benchmark results to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save results to {file_path}: {e}")
    
    def load_results(self, file_path: str) -> List[CompressionResult]:
        """
        Load benchmark results from a JSON file.
        
        Args:
            file_path: Path to load results from
            
        Returns:
            List of loaded compression results
        """
        try:
            with open(file_path, 'r') as f:
                results_data = json.load(f)
            
            results = []
            for item in results_data:
                try:
                    # Convert dict to CompressionResult
                    result = CompressionResult(
                        algorithm_name=item["algorithm_name"],
                        original_size=item["original_size"],
                        compressed_size=item["compressed_size"],
                        compression_ratio=item["compression_ratio"],
                        pre_compression_psi_c=item["pre_compression_psi_c"],
                        post_compression_psi_c=item["post_compression_psi_c"],
                        psi_c_delta=item["psi_c_delta"],
                        pre_compression_coherence=item["pre_compression_coherence"],
                        post_compression_coherence=item["post_compression_coherence"],
                        coherence_delta=item["coherence_delta"],
                        execution_time_ms=item["execution_time_ms"],
                        memory_usage_kb=item["memory_usage_kb"],
                        accuracy_scores=item.get("accuracy_scores", {}),
                        removed_nodes=item.get("removed_nodes", []),
                        merged_nodes=item.get("merged_nodes", [])
                    )
                    results.append(result)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Error parsing result: {e}")
            
            self.benchmark_results.extend(results)
            self.logger.info(f"Loaded {len(results)} benchmark results from {file_path}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Failed to load results from {file_path}: {e}")
            return []
    
    def get_best_algorithm(self, 
                          metric: str = "psi_c_delta",
                          min_compression_ratio: float = 1.5) -> Optional[str]:
        """
        Get the best performing algorithm based on a specified metric.
        
        Args:
            metric: Metric to optimize ("psi_c_delta", "coherence_delta", etc.)
            min_compression_ratio: Minimum acceptable compression ratio
            
        Returns:
            Name of the best algorithm or None if no results
        """
        if not self.benchmark_results:
            return None
        
        # Filter results by minimum compression ratio
        filtered_results = [r for r in self.benchmark_results 
                          if r.compression_ratio >= min_compression_ratio]
        
        if not filtered_results:
            return None
        
        # Find the best algorithm based on the metric
        if metric == "psi_c_delta":
            best_result = max(filtered_results, key=lambda r: r.psi_c_delta)
        elif metric == "coherence_delta":
            best_result = max(filtered_results, key=lambda r: r.coherence_delta)
        elif metric == "compression_ratio":
            best_result = max(filtered_results, key=lambda r: r.compression_ratio)
        elif metric == "execution_time_ms":
            best_result = min(filtered_results, key=lambda r: r.execution_time_ms)
        else:
            # Assume a weighted combination
            def combined_score(r):
                psi_weight = 0.4
                coherence_weight = 0.3
                compression_weight = 0.2
                time_weight = 0.1
                
                # Normalize execution time (lower is better)
                max_time = max(r.execution_time_ms for r in filtered_results)
                normalized_time = 1.0 - (r.execution_time_ms / max_time) if max_time > 0 else 0
                
                return (
                    psi_weight * r.psi_c_delta +
                    coherence_weight * r.coherence_delta +
                    compression_weight * (r.compression_ratio - 1.0) +
                    time_weight * normalized_time
                )
            
            best_result = max(filtered_results, key=combined_score)
        
        return best_result.algorithm_name


# Example usage and demo functions
def create_demo_graph() -> nx.Graph:
    """Create a demo schema graph for testing."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(20):
        G.add_node(f"concept_{i}", type="concept", value=np.random.rand())
    
    # Add edges with random weights
    for i in range(20):
        for j in range(i+1, 20):
            if np.random.rand() < 0.3:  # 30% chance of edge
                G.add_edge(f"concept_{i}", f"concept_{j}", weight=np.random.rand())
    
    return G

def demo_psi_c_evaluator(graph: nx.Graph) -> float:
    """Demo ΨC evaluator function for testing."""
    # Simple ΨC proxy based on graph properties
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    
    if n_nodes == 0:
        return 0.0
    
    # Connectivity and structure metrics
    avg_degree = 2 * n_edges / n_nodes
    try:
        avg_clustering = nx.average_clustering(graph)
    except:
        avg_clustering = 0.0
    
    # Calculate a simple ΨC score
    psi_c = 0.5 * avg_degree / 5.0 + 0.5 * avg_clustering
    return min(1.0, max(0.0, psi_c))

def demo_coherence_evaluator(graph: nx.Graph) -> float:
    """Demo coherence evaluator function for testing."""
    # Simple coherence proxy based on graph properties
    if graph.number_of_nodes() == 0:
        return 0.0
    
    try:
        # Use transitivity as a proxy for coherence
        transitivity = nx.transitivity(graph)
        
        # Use connected components as another factor
        n_components = nx.number_connected_components(graph)
        component_factor = 1.0 / n_components
        
        coherence = 0.7 * transitivity + 0.3 * component_factor
        return min(1.0, max(0.0, coherence))
    except:
        return 0.5  # Default value on error

def demo_functional_tests() -> Dict[str, Callable]:
    """Create demo functional tests for testing."""
    def retrieval_test(graph: nx.Graph) -> float:
        """Test how well the graph preserves important nodes."""
        important_nodes = [f"concept_{i}" for i in range(5)]  # First 5 nodes are "important"
        preserved = sum(1 for node in important_nodes if node in graph.nodes())
        return preserved / len(important_nodes)
    
    def inference_test(graph: nx.Graph) -> float:
        """Test inference capability using path length as proxy."""
        # Sample random pairs and check path length
        n_samples = 10
        total_score = 0.0
        for _ in range(n_samples):
            nodes = list(graph.nodes())
            if len(nodes) < 2:
                continue
            
            source, target = np.random.choice(nodes, 2, replace=False)
            try:
                path_length = nx.shortest_path_length(graph, source=source, target=target)
                path_score = 1.0 / (1.0 + path_length)  # Shorter path is better
                total_score += path_score
            except nx.NetworkXNoPath:
                pass  # No path exists
        
        return total_score / n_samples if n_samples > 0 else 0.0
    
    return {
        "retrieval": retrieval_test,
        "inference": inference_test
    }

def run_demo():
    """Run a demo of the schema compression benchmark."""
    # Create evaluator
    evaluator = SchemaCompressionEval()
    
    # Create demo graph
    schema_graph = create_demo_graph()
    print(f"Created demo graph with {schema_graph.number_of_nodes()} nodes and {schema_graph.number_of_edges()} edges")
    
    # Run benchmark
    results = evaluator.run_benchmark(
        schema_graph=schema_graph,
        psi_c_evaluator=demo_psi_c_evaluator,
        coherence_evaluator=demo_coherence_evaluator,
        functional_tests=demo_functional_tests()
    )
    
    # Visualize results
    evaluator.visualize_results(results)
    
    # Get best algorithm
    best_algo = evaluator.get_best_algorithm()
    print(f"Best overall algorithm: {best_algo}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    run_demo() 