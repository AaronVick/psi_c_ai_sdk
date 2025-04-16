"""
Meta-Schema Compression Engine
-----------------------------

This module detects recurring meta-patterns in schema evolution and 
compresses them into abstract operators, enabling efficient schema storage 
and symbolic abstraction.

Long-term ΨC agents will accumulate redundant subgraphs in their schema.
This module:
1. Identifies isomorphic (structurally equivalent) subgraphs
2. Detects recurring patterns across schema snapshots
3. Compresses redundant structures into higher-level operators
4. Enables emergent symbolic abstraction

Mathematical basis:
    Pattern(G₁) ≈ Pattern(G₂) ⇒ Operator = compress(G₁)
"""

import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict
import datetime
import json
import os
from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
import warnings

# Setup logging
logger = logging.getLogger(__name__)


class MetaSchemaCompressor:
    """
    Detects and compresses recurring patterns in schema evolution.
    
    The compressor analyzes schema snapshots over time to identify redundant
    subgraphs and abstract them into higher-level operators, reducing memory
    footprint and enabling emergent symbolic processing.
    """
    
    def __init__(
        self,
        min_pattern_size: int = 3,
        min_pattern_frequency: int = 2,
        similarity_threshold: float = 0.85,
        max_compression_ratio: float = 0.3,
        snapshot_dir: Optional[str] = None
    ):
        """
        Initialize the meta-schema compressor.
        
        Args:
            min_pattern_size: Minimum nodes in a pattern to consider for compression
            min_pattern_frequency: Minimum occurrences required for compression
            similarity_threshold: Threshold for considering patterns equivalent
            max_compression_ratio: Maximum ratio of schema to compress at once
            snapshot_dir: Directory to save/load schema snapshots
        """
        self.min_pattern_size = min_pattern_size
        self.min_pattern_frequency = min_pattern_frequency
        self.similarity_threshold = similarity_threshold
        self.max_compression_ratio = max_compression_ratio
        
        # Set snapshot directory
        if snapshot_dir:
            self.snapshot_dir = Path(snapshot_dir)
        else:
            self.snapshot_dir = Path.home() / ".psi_c_ai_sdk" / "schema_snapshots"
        
        # Ensure directory exists
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.snapshots = []
        self.patterns = defaultdict(list)  # pattern_hash -> [occurrences]
        self.pattern_ops = {}  # pattern_hash -> compression_operator
        self.compressed_schemas = []
        
        # Metrics
        self.metrics = {
            "snapshot_count": 0,
            "patterns_detected": 0,
            "patterns_compressed": 0,
            "compression_ratio": 0.0,
            "symbolic_operators": 0
        }

    def snapshot_schema(self, agent, label: Optional[str] = None):
        """
        Capture a snapshot of the agent's current schema graph for analysis.
        
        Args:
            agent: The ΨC agent to snapshot
            label: Optional descriptive label for the snapshot
            
        Returns:
            Dictionary containing snapshot metadata
        """
        # Extract schema graph from agent
        schema_graph = self._extract_schema_graph(agent)
        
        if not schema_graph:
            logger.warning("Failed to extract schema graph from agent")
            return None
        
        # Generate timestamp and label
        timestamp = datetime.datetime.now().isoformat()
        if not label:
            label = f"schema_{len(self.snapshots) + 1}"
        
        # Create snapshot
        snapshot = {
            "timestamp": timestamp,
            "label": label,
            "graph": schema_graph,
            "node_count": schema_graph.number_of_nodes(),
            "edge_count": schema_graph.number_of_edges(),
            "metadata": self._extract_schema_metadata(agent)
        }
        
        # Store snapshot
        self.snapshots.append(snapshot)
        self.metrics["snapshot_count"] = len(self.snapshots)
        
        # Save to file
        self._save_snapshot(snapshot)
        
        logger.info(f"Captured schema snapshot '{label}' with {snapshot['node_count']} nodes and {snapshot['edge_count']} edges")
        return snapshot
        
    def _extract_schema_graph(self, agent) -> nx.Graph:
        """
        Extract the schema graph from an agent.
        
        Args:
            agent: The ΨC agent to analyze
            
        Returns:
            NetworkX graph representing the schema
        """
        try:
            # Try different agent implementations
            
            # Option 1: Direct schema graph
            if hasattr(agent, 'schema_graph'):
                return agent.schema_graph
                
            # Option 2: Schema object with graph
            elif hasattr(agent, 'schema') and hasattr(agent.schema, 'graph'):
                return agent.schema.graph
                
            # Option 3: Memory graph (as fallback)
            elif hasattr(agent, 'memory_graph'):
                logger.warning("Using memory graph as schema graph fallback")
                return agent.memory_graph
                
            # Option 4: Generate mock graph for testing
            else:
                logger.warning("Using mock schema graph - couldn't access agent schema")
                return self._generate_mock_schema(agent)
                
        except Exception as e:
            logger.error(f"Error extracting schema graph: {e}")
            return nx.Graph()
    
    def _extract_schema_metadata(self, agent) -> Dict:
        """
        Extract additional metadata about the schema.
        
        Args:
            agent: The ΨC agent to analyze
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        try:
            # Try different agent implementations
            if hasattr(agent, 'schema_metadata'):
                metadata.update(agent.schema_metadata)
                
            if hasattr(agent, 'get_schema_metrics'):
                metrics = agent.get_schema_metrics()
                metadata['metrics'] = metrics
                
            # Add basic agent info if available
            if hasattr(agent, 'id'):
                metadata['agent_id'] = agent.id
                
            if hasattr(agent, 'version'):
                metadata['agent_version'] = agent.version
                
        except Exception as e:
            logger.error(f"Error extracting schema metadata: {e}")
            
        return metadata
    
    def _save_snapshot(self, snapshot):
        """
        Save a schema snapshot to disk.
        
        Args:
            snapshot: The snapshot dictionary to save
        """
        try:
            # Create a copy without the graph (which isn't JSON serializable)
            save_snapshot = snapshot.copy()
            
            # Convert graph to serializable format
            G = snapshot['graph']
            save_snapshot['graph'] = {
                'nodes': list(G.nodes(data=True)),
                'edges': list(G.edges(data=True))
            }
            
            # Generate filename
            timestamp = snapshot['timestamp'].replace(':', '-')
            filename = f"schema_{timestamp}_{snapshot['label']}.json"
            filepath = self.snapshot_dir / filename
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_snapshot, f, indent=2, default=str)
                
            logger.info(f"Saved schema snapshot to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving schema snapshot: {e}")
            
    def _generate_mock_schema(self, agent) -> nx.Graph:
        """
        Generate a mock schema graph for testing.
        
        Args:
            agent: The agent to mock a schema for
            
        Returns:
            NetworkX graph representing a mock schema
        """
        # Create a random graph with community structure (for pattern detection testing)
        G = nx.Graph()
        
        # Add some clusters
        clusters = 4
        nodes_per_cluster = 5
        
        for c in range(clusters):
            # Create dense cluster
            for i in range(nodes_per_cluster):
                node_id = f"c{c}_n{i}"
                G.add_node(node_id, cluster=c, type="concept")
                
                # Connect to previous nodes in same cluster (dense)
                for j in range(i):
                    prev_node = f"c{c}_n{j}"
                    weight = np.random.uniform(0.7, 0.9)
                    G.add_edge(node_id, prev_node, weight=weight)
            
            # Add connections between clusters (sparse)
            if c > 0:
                # Connect each cluster to previous ones
                for i in range(2):  # Just 2 connections
                    source = f"c{c}_n{np.random.randint(0, nodes_per_cluster)}"
                    target_cluster = np.random.randint(0, c)
                    target = f"c{target_cluster}_n{np.random.randint(0, nodes_per_cluster)}"
                    weight = np.random.uniform(0.3, 0.6)
                    G.add_edge(source, target, weight=weight)
        
        return G
    
    def detect_patterns(self):
        """
        Detect recurring patterns across schema snapshots.
        
        This identifies isomorphic subgraphs and patterns that appear
        multiple times within and across schema snapshots.
        
        Returns:
            Number of patterns detected
        """
        if len(self.snapshots) < 2:
            logger.warning("Need at least 2 snapshots for pattern detection")
            return 0
            
        # Reset pattern storage
        self.patterns = defaultdict(list)
        
        # Extract all subgraphs meeting minimum size
        all_subgraphs = []
        
        for snapshot in self.snapshots:
            graph = snapshot['graph']
            snapshot_subgraphs = self._extract_subgraphs(graph)
            all_subgraphs.extend(snapshot_subgraphs)
            
        # Group similar subgraphs
        self._group_similar_subgraphs(all_subgraphs)
        
        # Filter by frequency
        self._filter_patterns_by_frequency()
        
        # Update metrics
        self.metrics["patterns_detected"] = len(self.patterns)
        
        logger.info(f"Detected {len(self.patterns)} recurring patterns across {len(self.snapshots)} snapshots")
        return len(self.patterns)
    
    def _extract_subgraphs(self, graph) -> List[Dict]:
        """
        Extract candidate subgraphs for pattern matching.
        
        Args:
            graph: Schema graph to analyze
            
        Returns:
            List of subgraph dictionaries with metadata
        """
        subgraphs = []
        
        # Method 1: Connected components
        for comp in nx.connected_components(graph):
            if len(comp) >= self.min_pattern_size:
                subgraph = graph.subgraph(comp)
                hash_val = self._hash_graph(subgraph)
                
                subgraphs.append({
                    "graph": subgraph,
                    "nodes": list(subgraph.nodes),
                    "edges": list(subgraph.edges),
                    "size": len(comp),
                    "hash": hash_val
                })
        
        # Method 2: K-core subgraphs (dense regions)
        k_cores = []
        try:
            # Find k-cores with different k values
            for k in range(2, 4):  # Try k=2, k=3
                for c in nx.connected_components(nx.k_core(graph, k=k)):
                    comp = nx.k_core(graph, k=k).subgraph(c)
                    if len(comp) >= self.min_pattern_size:
                        subgraph = graph.subgraph(comp)
                        hash_val = self._hash_graph(subgraph)
                        
                        k_cores.append({
                            "graph": subgraph,
                            "nodes": list(subgraph.nodes),
                            "edges": list(subgraph.edges),
                            "size": len(comp),
                            "hash": hash_val,
                            "k": k
                        })
        except nx.NetworkXError:
            # Some graphs might not have sufficient density for k-cores
            pass
            
        subgraphs.extend(k_cores)
        
        # Method 3: Community detection for larger graphs
        if graph.number_of_nodes() > 20:
            try:
                communities = nx.community.greedy_modularity_communities(graph)
                for i, comm in enumerate(communities):
                    if len(comm) >= self.min_pattern_size:
                        subgraph = graph.subgraph(comm)
                        hash_val = self._hash_graph(subgraph)
                        
                        subgraphs.append({
                            "graph": subgraph,
                            "nodes": list(subgraph.nodes),
                            "edges": list(subgraph.edges),
                            "size": len(comm),
                            "hash": hash_val,
                            "community_id": i
                        })
            except Exception:
                # Community detection can sometimes fail
                pass
        
        logger.debug(f"Extracted {len(subgraphs)} candidate subgraphs")
        return subgraphs
    
    def _hash_graph(self, graph) -> str:
        """
        Generate a structural hash for a graph.
        
        This uses a graph invariant method to create a hash that will be
        the same for isomorphic graphs, regardless of node labels.
        
        Args:
            graph: NetworkX graph to hash
            
        Returns:
            Hash string
        """
        try:
            # Method 1: Using graph spectrum (eigenvalues of adjacency matrix)
            adj_matrix = nx.to_numpy_array(graph)
            eigenvalues = np.linalg.eigvals(adj_matrix)
            eigenvalues.sort()
            eigenvalues = np.round(eigenvalues, 5)  # Reduce numerical errors
            
            # Method 2: Graph properties
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            degrees = sorted([d for n, d in graph.degree()])
            
            # Combine methods into a fingerprint
            fingerprint = f"{n_nodes}:{n_edges}:{degrees}:{eigenvalues}"
            
            # Hash the fingerprint
            hash_obj = hashlib.sha256(fingerprint.encode())
            return hash_obj.hexdigest()[:16]  # Truncate for readability
            
        except Exception as e:
            logger.error(f"Error hashing graph: {e}")
            # Fallback hash based on graph size
            return f"fallback_{graph.number_of_nodes()}_{graph.number_of_edges()}"
    
    def _group_similar_subgraphs(self, subgraphs):
        """
        Group similar subgraphs into pattern candidates.
        
        Args:
            subgraphs: List of subgraph dictionaries
        """
        # First, group by exact hash matches
        for subgraph in subgraphs:
            self.patterns[subgraph['hash']].append(subgraph)
            
        # Then try to merge similar patterns
        merged = True
        while merged:
            merged = False
            
            # Get list of current pattern hashes
            pattern_hashes = list(self.patterns.keys())
            
            for i in range(len(pattern_hashes)):
                if merged:
                    break  # Restart if merged
                    
                for j in range(i + 1, len(pattern_hashes)):
                    hash_i = pattern_hashes[i]
                    hash_j = pattern_hashes[j]
                    
                    # Skip if either pattern was already merged
                    if hash_i not in self.patterns or hash_j not in self.patterns:
                        continue
                        
                    # Compare representative graphs
                    graph_i = self.patterns[hash_i][0]['graph']
                    graph_j = self.patterns[hash_j][0]['graph']
                    
                    # Check if similar enough to merge
                    if self._are_graphs_similar(graph_i, graph_j):
                        # Merge j into i
                        self.patterns[hash_i].extend(self.patterns[hash_j])
                        del self.patterns[hash_j]
                        merged = True
                        logger.debug(f"Merged similar patterns {hash_i} and {hash_j}")
                        break
    
    def _are_graphs_similar(self, graph1, graph2) -> bool:
        """
        Check if two graphs are structurally similar.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            True if the graphs are similar enough to consider as the same pattern
        """
        # Quick size check
        n1, m1 = graph1.number_of_nodes(), graph1.number_of_edges()
        n2, m2 = graph2.number_of_nodes(), graph2.number_of_edges()
        
        # If sizes differ too much, return False
        if abs(n1 - n2) / max(n1, n2) > 0.2 or abs(m1 - m2) / max(m1, m2) > 0.2:
            return False
            
        # Check graph invariants
        try:
            # Compare degree distributions
            deg1 = sorted([d for n, d in graph1.degree()])
            deg2 = sorted([d for n, d in graph2.degree()])
            
            # Pad shorter list if needed
            if len(deg1) < len(deg2):
                deg1 = deg1 + [0] * (len(deg2) - len(deg1))
            elif len(deg2) < len(deg1):
                deg2 = deg2 + [0] * (len(deg1) - len(deg2))
                
            # Calculate distance
            dist = np.sum(np.abs(np.array(deg1) - np.array(deg2))) / max(sum(deg1), sum(deg2))
            
            if dist > 0.3:
                return False
                
            # Compare spectral properties
            eig1 = sorted(np.linalg.eigvals(nx.to_numpy_array(graph1)))
            eig2 = sorted(np.linalg.eigvals(nx.to_numpy_array(graph2)))
            
            # Pad shorter list
            if len(eig1) < len(eig2):
                eig1 = list(eig1) + [0] * (len(eig2) - len(eig1))
            elif len(eig2) < len(eig1):
                eig2 = list(eig2) + [0] * (len(eig1) - len(eig2))
                
            # Compare dominant eigenvalues (top 3)
            top_eig1 = np.abs(eig1[-3:])
            top_eig2 = np.abs(eig2[-3:])
            
            eig_sim = np.sum(np.abs(top_eig1 - top_eig2)) / max(np.sum(np.abs(top_eig1)), np.sum(np.abs(top_eig2)))
            
            # Compute overall similarity
            similarity = 1 - (0.5 * dist + 0.5 * eig_sim)
            
            return similarity >= self.similarity_threshold
            
        except Exception as e:
            logger.warning(f"Error comparing graphs: {e}")
            return False
    
    def _filter_patterns_by_frequency(self):
        """
        Filter patterns by minimum frequency.
        
        Removes patterns that don't occur frequently enough.
        """
        # Copy keys to avoid modification during iteration
        pattern_hashes = list(self.patterns.keys())
        
        for pattern_hash in pattern_hashes:
            occurrences = self.patterns[pattern_hash]
            
            if len(occurrences) < self.min_pattern_frequency:
                del self.patterns[pattern_hash]
                logger.debug(f"Removed infrequent pattern {pattern_hash} with {len(occurrences)} occurrences")
                
    def load_snapshots(self, directory: Optional[str] = None):
        """
        Load schema snapshots from the specified directory.
        
        Args:
            directory: Directory containing schema snapshot files
            
        Returns:
            Number of snapshots loaded
        """
        # Default to configured directory
        if directory is None:
            directory = self.snapshot_dir
        else:
            directory = Path(directory)
            
        # Reset snapshots
        self.snapshots = []
        
        # Find snapshot files
        snapshot_files = list(directory.glob("schema_*.json"))
        
        for file_path in snapshot_files:
            try:
                with open(file_path, 'r') as f:
                    snapshot = json.load(f)
                    
                # Convert serialized graph back to NetworkX
                if 'graph' in snapshot and isinstance(snapshot['graph'], dict):
                    G = nx.Graph()
                    
                    # Add nodes with attributes
                    for node_data in snapshot['graph']['nodes']:
                        node_id = node_data[0]
                        attrs = node_data[1]
                        G.add_node(node_id, **attrs)
                        
                    # Add edges with attributes
                    for edge_data in snapshot['graph']['edges']:
                        source = edge_data[0]
                        target = edge_data[1]
                        attrs = edge_data[2] if len(edge_data) > 2 else {}
                        G.add_edge(source, target, **attrs)
                        
                    snapshot['graph'] = G
                    
                    # Add to snapshots list
                    self.snapshots.append(snapshot)
                    
            except Exception as e:
                logger.error(f"Error loading snapshot {file_path}: {e}")
                
        # Update metrics
        self.metrics["snapshot_count"] = len(self.snapshots)
        
        # Sort by timestamp
        self.snapshots.sort(key=lambda x: x.get('timestamp', ''))
        
        logger.info(f"Loaded {len(self.snapshots)} schema snapshots from {directory}")
        return len(self.snapshots)

    def compress_schema(self, agent):
        """
        Apply compression to the agent's schema based on detected patterns.
        
        Args:
            agent: The ΨC agent to compress
            
        Returns:
            Compression stats dictionary
        """
        if not self.patterns:
            logger.warning("No patterns detected. Run detect_patterns first.")
            return {"compressed": 0, "ratio": 0.0}
            
        # Extract schema graph
        schema_graph = self._extract_schema_graph(agent)
        
        if not schema_graph:
            logger.warning("Failed to extract schema graph for compression")
            return {"compressed": 0, "ratio": 0.0}
            
        # Take snapshot before compression
        pre_compression = {
            "node_count": schema_graph.number_of_nodes(),
            "edge_count": schema_graph.number_of_edges()
        }
        
        # Find pattern occurrences in the current schema
        pattern_occurrences = self._find_patterns_in_schema(schema_graph)
        
        # Sort patterns by occurrence frequency * size
        pattern_value = {}
        for pattern_hash, occurrences in pattern_occurrences.items():
            pattern_size = len(self.patterns[pattern_hash][0]['nodes'])
            pattern_value[pattern_hash] = len(occurrences) * pattern_size
            
        sorted_patterns = sorted(
            pattern_value.keys(), 
            key=lambda h: pattern_value[h], 
            reverse=True
        )
        
        # Apply compression to top patterns (within max compression ratio)
        max_nodes_to_compress = int(pre_compression["node_count"] * self.max_compression_ratio)
        compressed_nodes = 0
        compressed_patterns = 0
        
        for pattern_hash in sorted_patterns:
            occurrences = pattern_occurrences[pattern_hash]
            pattern = self.patterns[pattern_hash][0]
            pattern_size = len(pattern['nodes'])
            
            # Skip if this would exceed max compression ratio
            if compressed_nodes + (pattern_size * len(occurrences)) > max_nodes_to_compress:
                continue
                
            # Create compression operator if it doesn't exist
            if pattern_hash not in self.pattern_ops:
                op_name = f"op_{len(self.pattern_ops) + 1}"
                self.pattern_ops[pattern_hash] = {
                    "id": op_name,
                    "pattern_hash": pattern_hash,
                    "node_count": pattern_size,
                    "pattern": pattern
                }
                self.metrics["symbolic_operators"] = len(self.pattern_ops)
                
            # Apply compression to each occurrence
            for occurrence in occurrences:
                self._compress_pattern_instance(schema_graph, occurrence, self.pattern_ops[pattern_hash])
                compressed_nodes += pattern_size
                
            compressed_patterns += 1
            
        # Take snapshot after compression
        post_compression = {
            "node_count": schema_graph.number_of_nodes(),
            "edge_count": schema_graph.number_of_edges()
        }
        
        # Calculate compression ratio
        if pre_compression["node_count"] > 0:
            compression_ratio = (pre_compression["node_count"] - post_compression["node_count"]) / pre_compression["node_count"]
        else:
            compression_ratio = 0.0
            
        # Update metrics
        self.metrics["patterns_compressed"] = compressed_patterns
        self.metrics["compression_ratio"] = compression_ratio
        
        # Store compressed schema
        self.compressed_schemas.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "pre_compression": pre_compression,
            "post_compression": post_compression,
            "patterns_used": compressed_patterns,
            "compression_ratio": compression_ratio,
            "graph": schema_graph
        })
        
        # Update agent's schema graph if possible
        self._update_agent_schema(agent, schema_graph)
        
        logger.info(f"Compressed schema using {compressed_patterns} patterns, achieving {compression_ratio:.2%} reduction")
        
        return {
            "pre_nodes": pre_compression["node_count"],
            "post_nodes": post_compression["node_count"],
            "compressed_patterns": compressed_patterns,
            "compression_ratio": compression_ratio
        }
    
    def _find_patterns_in_schema(self, schema_graph) -> Dict[str, List[Set[str]]]:
        """
        Find occurrences of detected patterns in the schema graph.
        
        Args:
            schema_graph: The schema graph to search
            
        Returns:
            Dictionary mapping pattern hashes to lists of node sets
        """
        occurrences = defaultdict(list)
        
        for pattern_hash, patterns in self.patterns.items():
            # Use the first instance as the pattern representative
            pattern_graph = patterns[0]['graph']
            pattern_size = len(patterns[0]['nodes'])
            
            # Skip patterns too large for efficient matching
            if pattern_size > 15:
                logger.debug(f"Skipping large pattern ({pattern_size} nodes) for efficiency")
                continue
                
            try:
                # Use VF2 algorithm for graph isomorphism
                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    schema_graph, pattern_graph,
                    node_match=lambda n1, n2: True,  # Ignore node attributes
                    edge_match=lambda e1, e2: True   # Ignore edge attributes
                )
                
                # Find all subgraph matches
                for mapping in matcher.subgraph_isomorphisms_iter():
                    # Convert dict mapping to set of nodes
                    matched_nodes = set(mapping.keys())
                    
                    # Check if this is a new occurrence (not overlapping with previous)
                    is_new = True
                    for existing in occurrences[pattern_hash]:
                        if len(matched_nodes.intersection(existing)) > 0:
                            is_new = False
                            break
                            
                    if is_new:
                        occurrences[pattern_hash].append(matched_nodes)
                        
            except Exception as e:
                logger.error(f"Error finding pattern {pattern_hash} in schema: {e}")
                
        return occurrences
    
    def _compress_pattern_instance(self, schema_graph, nodes, operator):
        """
        Compress a pattern instance in the schema graph.
        
        Args:
            schema_graph: The schema graph to modify
            nodes: Set of nodes in the pattern instance
            operator: The compression operator to apply
        """
        if len(nodes) < 2:
            return
            
        try:
            # Create a new operator node
            op_node_id = f"{operator['id']}_{len(schema_graph)}"
            
            # Add the operator node
            schema_graph.add_node(op_node_id, 
                type="compressed_operator",
                operator=operator['id'],
                pattern_hash=operator['pattern_hash'],
                compressed_size=len(nodes)
            )
            
            # Find boundary nodes (connected to the pattern but not part of it)
            boundary_edges = []
            
            for node in nodes:
                for neighbor in schema_graph.neighbors(node):
                    if neighbor not in nodes:
                        boundary_edges.append((node, neighbor))
            
            # Connect operator node to boundary nodes
            for pattern_node, external_node in boundary_edges:
                edge_data = schema_graph.get_edge_data(pattern_node, external_node, default={})
                schema_graph.add_edge(op_node_id, external_node, **edge_data)
            
            # Remove pattern nodes
            for node in nodes:
                schema_graph.remove_node(node)
                
        except Exception as e:
            logger.error(f"Error compressing pattern instance: {e}")
    
    def _update_agent_schema(self, agent, compressed_graph):
        """
        Update the agent's schema graph with the compressed version.
        
        Args:
            agent: The ΨC agent to update
            compressed_graph: The compressed schema graph
        """
        try:
            # Try different agent implementations
            
            # Option 1: Direct schema graph
            if hasattr(agent, 'schema_graph'):
                agent.schema_graph = compressed_graph
                logger.info("Updated agent's schema_graph directly")
                return True
                
            # Option 2: Schema object with graph
            elif hasattr(agent, 'schema') and hasattr(agent.schema, 'graph'):
                agent.schema.graph = compressed_graph
                logger.info("Updated agent's schema.graph")
                return True
                
            # Option 3: No direct update possible
            else:
                logger.warning("Could not update agent's schema graph directly - agent needs to re-import")
                return False
                
        except Exception as e:
            logger.error(f"Error updating agent schema: {e}")
            return False
    
    def visualize_patterns(self, output_dir: Optional[str] = None):
        """
        Visualize detected patterns and compression results.
        
        Args:
            output_dir: Directory to save visualizations. If None, will display only.
            
        Returns:
            List of generated figure paths if output_dir provided
        """
        if not self.patterns:
            logger.warning("No patterns to visualize. Run detect_patterns first.")
            return []
            
        # Set output directory
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = None
            
        figure_paths = []
        
        # Figure 1: Pattern frequency chart
        plt.figure(figsize=(10, 6))
        
        pattern_sizes = []
        pattern_counts = []
        pattern_labels = []
        
        # Sort patterns by frequency
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for i, (pattern_hash, occurrences) in enumerate(sorted_patterns[:10]):  # Top 10
            pattern_sizes.append(len(occurrences[0]['nodes']))
            pattern_counts.append(len(occurrences))
            pattern_labels.append(f"P{i+1}\n({pattern_sizes[-1]} nodes)")
            
        # Create bar chart
        plt.bar(pattern_labels, pattern_counts)
        plt.title("Top 10 Recurring Patterns by Frequency")
        plt.ylabel("Occurrences")
        plt.xlabel("Pattern (with node count)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if out_dir:
            fig_path = out_dir / "pattern_frequency.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            figure_paths.append(str(fig_path))
        else:
            plt.show()
            
        plt.close()
        
        # Figure 2: Top patterns visualization
        for i, (pattern_hash, occurrences) in enumerate(sorted_patterns[:5]):  # Top 5
            plt.figure(figsize=(8, 6))
            
            pattern_graph = occurrences[0]['graph']
            
            # Plot the pattern graph
            pos = nx.spring_layout(pattern_graph)
            nx.draw_networkx(
                pattern_graph, pos,
                node_color='lightblue',
                node_size=300,
                edge_color='gray',
                with_labels=True,
                font_size=10
            )
            
            plt.title(f"Pattern {i+1} ({len(pattern_graph.nodes())} nodes, {len(occurrences)} occurrences)")
            plt.axis('off')
            
            if out_dir:
                fig_path = out_dir / f"pattern_{i+1}.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                figure_paths.append(str(fig_path))
            else:
                plt.show()
                
            plt.close()
            
        # Figure 3: Compression results if available
        if self.compressed_schemas:
            plt.figure(figsize=(10, 6))
            
            # Extract compression data
            timestamps = []
            ratios = []
            node_counts = []
            
            for cs in self.compressed_schemas:
                timestamps.append(cs['timestamp'][-8:])  # Just time part
                ratios.append(cs['compression_ratio'] * 100)
                node_counts.append(cs['post_compression']['node_count'])
                
            # Plot compression ratio over time
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, ratios, marker='o', linestyle='-', color='green')
            plt.title("Schema Compression Results")
            plt.ylabel("Compression Ratio (%)")
            plt.grid(linestyle='--', alpha=0.7)
            
            # Plot node count over time
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, node_counts, marker='s', linestyle='-', color='blue')
            plt.ylabel("Node Count After Compression")
            plt.grid(linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if out_dir:
                fig_path = out_dir / "compression_results.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                figure_paths.append(str(fig_path))
            else:
                plt.show()
                
            plt.close()
            
        return figure_paths
    
    def visualize_compression_example(self, output_dir: Optional[str] = None):
        """
        Visualize a before/after example of schema compression.
        
        Args:
            output_dir: Directory to save visualizations. If None, will display only.
            
        Returns:
            Path to the generated figure or None
        """
        if not self.compressed_schemas:
            logger.warning("No compression examples to visualize.")
            return None
            
        # Get the latest compression example
        compression = self.compressed_schemas[-1]
        
        # Set output directory
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = None
            
        # Create the visualization
        plt.figure(figsize=(15, 8))
        
        # Get the before and after schema graphs (before is from snapshot)
        after_graph = compression['graph']
        before_snapshot = self.snapshots[-2] if len(self.snapshots) > 1 else self.snapshots[-1]
        before_graph = before_snapshot['graph']
        
        # Before compression
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(before_graph, seed=42)
        node_colors = ['lightblue'] * before_graph.number_of_nodes()
        
        nx.draw_networkx(
            before_graph, pos,
            node_color=node_colors,
            node_size=200,
            edge_color='gray',
            with_labels=False,
            font_size=8
        )
        plt.title(f"Before Compression\n({before_graph.number_of_nodes()} nodes, {before_graph.number_of_edges()} edges)")
        plt.axis('off')
        
        # After compression
        plt.subplot(1, 2, 2)
        pos = nx.spring_layout(after_graph, seed=42)
        
        # Color operator nodes differently
        node_colors = []
        for node in after_graph.nodes():
            if after_graph.nodes[node].get('type') == 'compressed_operator':
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx(
            after_graph, pos,
            node_color=node_colors,
            node_size=200,
            edge_color='gray',
            with_labels=False,
            font_size=8
        )
        
        # Highlight operator nodes with larger size
        operator_nodes = [n for n, attrs in after_graph.nodes(data=True)
                         if attrs.get('type') == 'compressed_operator']
        if operator_nodes:
            nx.draw_networkx_nodes(
                after_graph, pos,
                nodelist=operator_nodes,
                node_color='red',
                node_size=300
            )
        
        plt.title(f"After Compression\n({after_graph.number_of_nodes()} nodes, {after_graph.number_of_edges()} edges)")
        plt.axis('off')
        
        plt.suptitle(f"Schema Compression Example\nCompression Ratio: {compression['compression_ratio']:.2%}", fontsize=16)
        plt.tight_layout()
        
        if out_dir:
            fig_path = out_dir / "compression_example.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(fig_path)
        else:
            plt.show()
            plt.close()
            return None
    
    def save_patterns(self, file_path: Optional[str] = None):
        """
        Save detected patterns to a file for later use.
        
        Args:
            file_path: Path to save patterns JSON file
            
        Returns:
            Path to the saved file
        """
        if not self.patterns:
            logger.warning("No patterns to save. Run detect_patterns first.")
            return None
            
        # Generate default path if not provided
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.snapshot_dir / f"patterns_{timestamp}.json"
        else:
            file_path = Path(file_path)
            
        # Create serializable pattern data
        pattern_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pattern_count": len(self.patterns),
            "metrics": self.metrics,
            "patterns": {}
        }
        
        for pattern_hash, occurrences in self.patterns.items():
            # Use the first occurrence as representative
            pattern = occurrences[0]
            
            # Convert graph to serializable format
            G = pattern['graph']
            serialized_graph = {
                'nodes': list(G.nodes(data=True)),
                'edges': list(G.edges(data=True))
            }
            
            pattern_data["patterns"][pattern_hash] = {
                "hash": pattern_hash,
                "size": pattern['size'],
                "occurrence_count": len(occurrences),
                "graph": serialized_graph
            }
            
            # Add compression operator if exists
            if pattern_hash in self.pattern_ops:
                pattern_data["patterns"][pattern_hash]["operator"] = self.pattern_ops[pattern_hash]["id"]
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(pattern_data, f, indent=2, default=str)
                
            logger.info(f"Saved {len(self.patterns)} patterns to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
            return None
    
    def load_patterns(self, file_path: str):
        """
        Load patterns from a file.
        
        Args:
            file_path: Path to the patterns JSON file
            
        Returns:
            Number of patterns loaded
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r') as f:
                pattern_data = json.load(f)
                
            # Reset patterns
            self.patterns = defaultdict(list)
            self.pattern_ops = {}
            
            # Load pattern data
            for pattern_hash, data in pattern_data["patterns"].items():
                # Convert serialized graph back to NetworkX
                G = nx.Graph()
                
                # Add nodes with attributes
                for node_data in data["graph"]['nodes']:
                    node_id = node_data[0]
                    attrs = node_data[1]
                    G.add_node(node_id, **attrs)
                    
                # Add edges with attributes
                for edge_data in data["graph"]['edges']:
                    source = edge_data[0]
                    target = edge_data[1]
                    attrs = edge_data[2] if len(edge_data) > 2 else {}
                    G.add_edge(source, target, **attrs)
                
                # Create pattern entry
                pattern = {
                    "graph": G,
                    "nodes": list(G.nodes()),
                    "edges": list(G.edges()),
                    "size": data["size"],
                    "hash": pattern_hash
                }
                
                # Add to patterns
                self.patterns[pattern_hash].append(pattern)
                
                # Load operator if present
                if "operator" in data:
                    self.pattern_ops[pattern_hash] = {
                        "id": data["operator"],
                        "pattern_hash": pattern_hash,
                        "node_count": data["size"],
                        "pattern": pattern
                    }
            
            # Update metrics
            self.metrics["patterns_detected"] = len(self.patterns)
            self.metrics["symbolic_operators"] = len(self.pattern_ops)
            
            logger.info(f"Loaded {len(self.patterns)} patterns from {file_path}")
            return len(self.patterns)
            
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return 0
    
    def get_metrics(self):
        """
        Get current metrics.
        
        Returns:
            Dictionary of compressor metrics
        """
        return self.metrics.copy()


def run_compression(agent, min_pattern_size=3, min_frequency=2, visualize=True):
    """
    Utility function to run schema compression on an agent.
    
    Args:
        agent: ΨC agent to compress
        min_pattern_size: Minimum nodes in a pattern to consider
        min_frequency: Minimum occurrences required for compression
        visualize: Whether to generate visualizations
        
    Returns:
        Compression stats dictionary
    """
    compressor = MetaSchemaCompressor(
        min_pattern_size=min_pattern_size,
        min_pattern_frequency=min_frequency
    )
    
    # Take snapshot of current state
    compressor.snapshot_schema(agent, "pre_compression")
    
    # Generate a mock schema change if this is a test
    if not hasattr(agent, 'schema_graph') and not hasattr(agent, 'schema'):
        logger.info("This appears to be a test run with a mock agent")
        # Take another snapshot with mock changes
        compressor.snapshot_schema(agent, "mock_changes")
    
    # Load and analyze snapshots
    if compressor.metrics["snapshot_count"] < 2:
        # Try to load existing snapshots
        compressor.load_snapshots()
        
    # Detect patterns
    pattern_count = compressor.detect_patterns()
    print(f"Detected {pattern_count} recurring patterns")
    
    if pattern_count == 0:
        print("Insufficient pattern data for compression")
        return {"compressed": 0, "ratio": 0.0}
    
    # Compress schema
    results = compressor.compress_schema(agent)
    print(f"Compressed schema: {results['compression_ratio']:.2%} reduction")
    print(f"Nodes: {results['pre_nodes']} → {results['post_nodes']}")
    print(f"Patterns used: {results['compressed_patterns']}")
    
    # Visualize results
    if visualize:
        vis_dir = Path.home() / ".psi_c_ai_sdk" / "compression_results"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        compressor.visualize_patterns(vis_dir)
        example_path = compressor.visualize_compression_example(vis_dir)
        
        if example_path:
            print(f"Created visualization at: {example_path}")
    
    # Save patterns for future use
    pattern_file = compressor.save_patterns()
    if pattern_file:
        print(f"Saved patterns to: {pattern_file}")
    
    return results


if __name__ == "__main__":
    # Simple demo with mock agent
    class MockAgent:
        def __init__(self):
            self.schema_graph = None
            
    # Create a mock agent
    agent = MockAgent()
    
    # Run compression
    results = run_compression(agent, visualize=True) 