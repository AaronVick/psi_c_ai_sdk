"""
Agent Memory Entanglement Monitor
---------------------------------

This module detects tightly coupled memories that may resist deletion or create
reflexive loops in ΨC agents. By monitoring memory entanglement, we can prevent 
"unerasable trauma" or belief constructs that anchor too many reflections and 
potentially distort the agent's evolution.

Key Features:
- Calculates entanglement scores for each memory
- Tracks graph-based distance between memories
- Identifies dangerously entangled memory clusters
- Provides intervention methods (warnings, schema unbinding)

Mathematical basis:
Memory entanglement score is defined as:
    E(M_i) = ∑ Coherence(M_i, M_j) / d_G(M_i, M_j)²
    
Where:
- Coherence is the semantic/structural similarity
- d_G is the graph distance (path length) between memories
"""

import logging
import math
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import numpy as np
from collections import defaultdict
import networkx as nx
import warnings

# Setup logging
logger = logging.getLogger(__name__)


class MemoryEntanglementMonitor:
    """
    Monitors and detects entanglement between memory elements in a ΨC agent.
    
    The monitor analyzes the memory graph structure to identify memories that are
    too tightly coupled, which could lead to problematic cognitive patterns like:
    
    1. Unerasable memories that resist forgetting/updating
    2. Reflexive belief loops that trap the agent in circular reasoning
    3. Memory anchors that disproportionately influence schema evolution
    """
    
    def __init__(
        self, 
        entanglement_threshold: float = 2.5, 
        critical_threshold: float = 4.0,
        min_coherence_threshold: float = 0.4,
        max_distance_factor: int = 3
    ):
        """
        Initialize the memory entanglement monitor.
        
        Args:
            entanglement_threshold: Threshold above which to flag entangled memories
            critical_threshold: Threshold for critical entanglement requiring intervention
            min_coherence_threshold: Minimum coherence to consider in entanglement
            max_distance_factor: Maximum graph distance factor to consider related
        """
        self.entanglement_threshold = entanglement_threshold
        self.critical_threshold = critical_threshold
        self.min_coherence_threshold = min_coherence_threshold
        self.max_distance_factor = max_distance_factor
        
        # Internal state
        self.memory_graph = None
        self.entanglement_scores = {}
        self.memory_clusters = []
        self.critical_memories = set()
        self.intervention_history = []
        
        # Metrics
        self.metrics = {
            "avg_entanglement": 0.0,
            "max_entanglement": 0.0,
            "critical_count": 0,
            "clustered_memories_pct": 0.0,
            "intervention_count": 0
        }
    
    def build_memory_graph(self, agent):
        """
        Construct a graph representation of the agent's memory structure.
        
        Args:
            agent: The ΨC agent to analyze
        """
        # Extract memory graph from agent
        # This implementation is flexible to work with different agent memory representations
        
        try:
            # Try to directly access the agent's memory graph if available
            if hasattr(agent, 'memory_graph'):
                self.memory_graph = agent.memory_graph
                logger.info("Retrieved memory graph directly from agent")
                return
                
            # Otherwise build a new graph from agent memories and connections
            G = nx.Graph()
            
            # Add memories as nodes
            memories = self._get_agent_memories(agent)
            for mem_id, memory in memories.items():
                G.add_node(mem_id, data=memory)
            
            # Add connections as edges with coherence weights
            connections = self._get_memory_connections(agent, memories)
            for mem_i, mem_j, coherence in connections:
                if coherence >= self.min_coherence_threshold:
                    G.add_edge(mem_i, mem_j, weight=coherence)
            
            self.memory_graph = G
            logger.info(f"Built memory graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to build memory graph: {e}")
            # Create empty graph as fallback
            self.memory_graph = nx.Graph()
    
    def _get_agent_memories(self, agent) -> Dict[str, Any]:
        """
        Extract memories from agent. Wrapper to handle different agent implementations.
        
        Args:
            agent: The ΨC agent to analyze
            
        Returns:
            Dictionary mapping memory IDs to memory objects
        """
        try:
            # Try different attribute names based on implementation
            if hasattr(agent, 'memories'):
                return agent.memories
            elif hasattr(agent, 'memory'):
                if isinstance(agent.memory, dict):
                    return agent.memory
                else:
                    return {i: mem for i, mem in enumerate(agent.memory)}
            elif hasattr(agent, 'memory_system') and hasattr(agent.memory_system, 'memories'):
                return agent.memory_system.memories
            else:
                # Mock data for testing
                logger.warning("Using mock memory data - couldn't access agent memories")
                return self._generate_mock_memories()
        except Exception as e:
            logger.error(f"Error accessing agent memories: {e}")
            return {}
    
    def _get_memory_connections(self, agent, memories) -> List[Tuple[str, str, float]]:
        """
        Extract memory connections with coherence weights.
        
        Args:
            agent: The ΨC agent to analyze
            memories: Dictionary of memory objects
            
        Returns:
            List of tuples (memory_id1, memory_id2, coherence_score)
        """
        try:
            # Try to get connections from agent's coherence matrix if available
            if hasattr(agent, 'coherence_matrix'):
                coherence_matrix = agent.coherence_matrix
                connections = []
                
                mem_ids = list(memories.keys())
                for i in range(len(mem_ids)):
                    for j in range(i+1, len(mem_ids)):
                        if coherence_matrix[i][j] >= self.min_coherence_threshold:
                            connections.append((mem_ids[i], mem_ids[j], coherence_matrix[i][j]))
                
                return connections
            
            # Otherwise try to calculate coherence between memories
            elif hasattr(agent, 'calculate_coherence'):
                connections = []
                mem_ids = list(memories.keys())
                
                for i in range(len(mem_ids)):
                    for j in range(i+1, len(mem_ids)):
                        coherence = agent.calculate_coherence(memories[mem_ids[i]], memories[mem_ids[j]])
                        if coherence >= self.min_coherence_threshold:
                            connections.append((mem_ids[i], mem_ids[j], coherence))
                
                return connections
            
            else:
                # Generate mock connections for testing
                logger.warning("Using mock connection data - couldn't access agent coherence")
                return self._generate_mock_connections(memories)
                
        except Exception as e:
            logger.error(f"Error extracting memory connections: {e}")
            return []
    
    def calculate_entanglement_scores(self):
        """
        Calculate entanglement scores for all memories in the graph.
        
        The entanglement score quantifies how tightly a memory is coupled with 
        other memories, factoring in both coherence strength and graph distance.
        
        Memories with high entanglement scores are resistant to modification
        and have outsized influence on the agent's cognitive processes.
        """
        if not self.memory_graph:
            logger.warning("No memory graph available. Run build_memory_graph first.")
            return
        
        # Reset scores
        self.entanglement_scores = {}
        
        # Calculate shortest path lengths between all nodes
        try:
            # Use all_pairs_shortest_path_length for efficiency
            path_lengths = dict(nx.all_pairs_shortest_path_length(self.memory_graph))
        except nx.NetworkXError:
            logger.warning("Graph is not connected, calculating component-wise path lengths")
            # Handle disconnected graph
            path_lengths = {}
            for component in nx.connected_components(self.memory_graph):
                subgraph = self.memory_graph.subgraph(component)
                component_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
                path_lengths.update(component_paths)
        
        # Calculate entanglement scores
        for node_i in self.memory_graph.nodes():
            score = 0.0
            
            # Sum contribution from each connected memory
            for node_j in self.memory_graph.nodes():
                if node_i == node_j:
                    continue
                    
                # Check if nodes are in the same component
                if node_j not in path_lengths.get(node_i, {}):
                    continue
                
                # Get graph distance
                distance = path_lengths[node_i][node_j]
                
                # Skip if too far (optimization)
                if distance > self.max_distance_factor:
                    continue
                
                # Find the path with highest coherence
                coherence = self._get_path_coherence(node_i, node_j)
                
                # Calculate contribution to entanglement score
                # Higher coherence and lower distance = higher entanglement
                if distance > 0 and coherence >= self.min_coherence_threshold:
                    score += coherence / (distance ** 2)
            
            self.entanglement_scores[node_i] = score
        
        # Update metrics
        if self.entanglement_scores:
            self.metrics["avg_entanglement"] = sum(self.entanglement_scores.values()) / len(self.entanglement_scores)
            self.metrics["max_entanglement"] = max(self.entanglement_scores.values())
        
        logger.info(f"Calculated entanglement scores for {len(self.entanglement_scores)} memories")
    
    def _get_path_coherence(self, node_i, node_j) -> float:
        """
        Get the coherence between two nodes based on the strongest path.
        
        Args:
            node_i: First node ID
            node_j: Second node ID
            
        Returns:
            Coherence score
        """
        # Direct edge case
        if self.memory_graph.has_edge(node_i, node_j):
            return self.memory_graph[node_i][node_j].get('weight', 0.0)
        
        # Otherwise find strongest connecting path
        try:
            paths = list(nx.all_simple_paths(self.memory_graph, node_i, node_j, cutoff=self.max_distance_factor))
            
            if not paths:
                return 0.0
                
            # Calculate coherence for each path (product of edge weights)
            path_coherences = []
            for path in paths:
                path_coherence = 1.0
                for i in range(len(path) - 1):
                    edge_coherence = self.memory_graph[path[i]][path[i+1]].get('weight', 0.0)
                    path_coherence *= edge_coherence
                path_coherences.append(path_coherence)
            
            # Return strongest path coherence
            return max(path_coherences)
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0
    
    def detect_entangled_clusters(self):
        """
        Identify clusters of entangled memories that may form problematic loops.
        
        This identifies subgraphs of highly entangled memories that could:
        1. Form circular reasoning patterns
        2. Amplify each other's resistance to updating
        3. Create cognitive blind spots
        """
        if not self.memory_graph or not self.entanglement_scores:
            logger.warning("Entanglement scores not calculated. Run calculate_entanglement_scores first.")
            return
        
        # Reset clusters
        self.memory_clusters = []
        self.critical_memories = set()
        
        # Mark critical memories (high entanglement score)
        for mem_id, score in self.entanglement_scores.items():
            if score > self.entanglement_threshold:
                # Flag as moderate risk
                logger.info(f"Memory {mem_id} has high entanglement score: {score:.2f}")
                
            if score > self.critical_threshold:
                # Flag as critical
                self.critical_memories.add(mem_id)
                logger.warning(f"CRITICAL: Memory {mem_id} has extreme entanglement: {score:.2f}")
        
        # Update metrics
        self.metrics["critical_count"] = len(self.critical_memories)
        
        # Detect clusters of entangled memories
        # Create a subgraph of memories with high entanglement
        entangled_nodes = [node for node, score in self.entanglement_scores.items() 
                           if score > self.entanglement_threshold]
        
        if entangled_nodes:
            entangled_subgraph = self.memory_graph.subgraph(entangled_nodes)
            
            # Find connected components (clusters)
            for component in nx.connected_components(entangled_subgraph):
                if len(component) > 1:  # Only consider multi-node clusters
                    self.memory_clusters.append(list(component))
            
            # Update metrics
            total_memories = len(self.memory_graph.nodes())
            clustered_memories = sum(len(cluster) for cluster in self.memory_clusters)
            if total_memories > 0:
                self.metrics["clustered_memories_pct"] = (clustered_memories / total_memories) * 100
            
            logger.info(f"Detected {len(self.memory_clusters)} entangled memory clusters")
    
    def recommend_interventions(self) -> List[Dict]:
        """
        Recommend interventions for problematic memory entanglements.
        
        Returns:
            List of intervention recommendations with details
        """
        if not self.memory_clusters and not self.critical_memories:
            return []
        
        interventions = []
        
        # Process critical individual memories
        for mem_id in self.critical_memories:
            intervention = {
                "type": "critical_memory",
                "memory_id": mem_id,
                "entanglement_score": self.entanglement_scores[mem_id],
                "action": "unbind",
                "details": f"Memory has extreme entanglement score of {self.entanglement_scores[mem_id]:.2f}, " 
                          f"exceeding critical threshold of {self.critical_threshold}."
            }
            interventions.append(intervention)
        
        # Process problematic clusters
        for i, cluster in enumerate(self.memory_clusters):
            # Check if this is a loop
            subgraph = self.memory_graph.subgraph(cluster)
            has_cycle = len(cluster) > 2 and len(subgraph.edges()) >= len(cluster)
            
            # Calculate cluster metrics
            avg_score = sum(self.entanglement_scores[mem_id] for mem_id in cluster) / len(cluster)
            max_score = max(self.entanglement_scores[mem_id] for mem_id in cluster)
            
            # Determine action based on severity
            if max_score > self.critical_threshold:
                action = "dissolve_cluster"
            elif has_cycle:
                action = "break_cycle"
            else:
                action = "monitor"
            
            intervention = {
                "type": "memory_cluster",
                "cluster_id": i,
                "memory_ids": cluster,
                "avg_entanglement": avg_score,
                "max_entanglement": max_score,
                "has_cycle": has_cycle,
                "action": action,
                "details": f"Cluster of {len(cluster)} memories with average entanglement " 
                          f"{avg_score:.2f} and maximum {max_score:.2f}."
            }
            interventions.append(intervention)
        
        self.intervention_history.extend(interventions)
        self.metrics["intervention_count"] = len(self.intervention_history)
        
        return interventions
    
    def apply_interventions(self, agent, interventions=None):
        """
        Apply recommended interventions to the agent.
        
        Args:
            agent: The ΨC agent to modify
            interventions: List of interventions to apply, or None to use recommendations
            
        Returns:
            Number of interventions applied
        """
        if interventions is None:
            interventions = self.recommend_interventions()
        
        if not interventions:
            return 0
        
        applied_count = 0
        
        for intervention in interventions:
            try:
                if intervention["action"] == "unbind":
                    # Attempt to reduce entanglement by modifying weights
                    mem_id = intervention["memory_id"]
                    self._unbind_memory(agent, mem_id)
                    applied_count += 1
                    logger.info(f"Unbound critically entangled memory: {mem_id}")
                    
                elif intervention["action"] == "break_cycle":
                    # Break a problematic cycle by removing the weakest link
                    cluster = intervention["memory_ids"]
                    self._break_cycle(agent, cluster)
                    applied_count += 1
                    logger.info(f"Broke cycle in memory cluster: {cluster}")
                    
                elif intervention["action"] == "dissolve_cluster":
                    # Dissolve a critically entangled cluster
                    cluster = intervention["memory_ids"]
                    self._dissolve_cluster(agent, cluster)
                    applied_count += 1
                    logger.info(f"Dissolved critically entangled cluster: {cluster}")
                    
            except Exception as e:
                logger.error(f"Failed to apply intervention: {e}")
        
        return applied_count
    
    def _unbind_memory(self, agent, mem_id):
        """
        Reduce entanglement of a critical memory by modifying its connections.
        
        Args:
            agent: The ΨC agent to modify
            mem_id: ID of the memory to unbind
        """
        # Try different agent implementations
        
        # Option 1: Use agent's built-in methods if available
        if hasattr(agent, 'reduce_memory_entanglement'):
            agent.reduce_memory_entanglement(mem_id)
            return
            
        # Option 2: Modify coherence weights directly
        elif hasattr(agent, 'memory_graph') or hasattr(agent, 'coherence_matrix'):
            # Identify connections to weaken
            connections = list(self.memory_graph.edges(mem_id, data=True))
            
            # Sort by coherence (weight)
            connections.sort(key=lambda x: x[2].get('weight', 0), reverse=True)
            
            # Reduce strength of strongest connections by 50%
            for i in range(min(3, len(connections))):
                source, target, data = connections[i]
                
                # Update in our graph
                self.memory_graph[source][target]['weight'] *= 0.5
                
                # Try to update in agent's data structures
                if hasattr(agent, 'coherence_matrix'):
                    source_idx = list(self.memory_graph.nodes()).index(source)
                    target_idx = list(self.memory_graph.nodes()).index(target)
                    agent.coherence_matrix[source_idx][target_idx] *= 0.5
                    agent.coherence_matrix[target_idx][source_idx] *= 0.5
            
            return
            
        # Option 3: Fallback to warning
        else:
            warnings.warn(f"Cannot unbind memory {mem_id} - agent implementation doesn't support it")
    
    def _break_cycle(self, agent, cluster):
        """
        Break a cycle in a memory cluster by removing the weakest link.
        
        Args:
            agent: The ΨC agent to modify
            cluster: List of memory IDs in the cluster
        """
        # Find cycles in the cluster
        subgraph = self.memory_graph.subgraph(cluster)
        
        try:
            # Find simple cycles
            cycles = list(nx.simple_cycles(subgraph.to_directed()))
            
            if not cycles:
                return
                
            # Get the first cycle
            cycle = cycles[0]
            
            # Find the weakest link in the cycle
            min_weight = float('inf')
            weakest_edge = None
            
            for i in range(len(cycle)):
                node1 = cycle[i]
                node2 = cycle[(i + 1) % len(cycle)]
                
                if subgraph.has_edge(node1, node2):
                    weight = subgraph[node1][node2].get('weight', 0)
                    if weight < min_weight:
                        min_weight = weight
                        weakest_edge = (node1, node2)
            
            if weakest_edge:
                # Remove weakest edge from our graph
                if self.memory_graph.has_edge(*weakest_edge):
                    self.memory_graph.remove_edge(*weakest_edge)
                
                # Try to update agent data structures
                if hasattr(agent, 'memory_graph'):
                    if agent.memory_graph.has_edge(*weakest_edge):
                        agent.memory_graph.remove_edge(*weakest_edge)
                        
                if hasattr(agent, 'coherence_matrix'):
                    source_idx = list(self.memory_graph.nodes()).index(weakest_edge[0])
                    target_idx = list(self.memory_graph.nodes()).index(weakest_edge[1])
                    agent.coherence_matrix[source_idx][target_idx] = 0.0
                    agent.coherence_matrix[target_idx][source_idx] = 0.0
                
                logger.info(f"Broke cycle by removing edge between {weakest_edge[0]} and {weakest_edge[1]}")
                
        except nx.NetworkXNoCycle:
            logger.info("No cycle found in cluster")
            return
    
    def _dissolve_cluster(self, agent, cluster):
        """
        Dissolve a critically entangled cluster by reducing internal coherence.
        
        Args:
            agent: The ΨC agent to modify
            cluster: List of memory IDs in the cluster
        """
        # Get the subgraph for the cluster
        subgraph = self.memory_graph.subgraph(cluster)
        
        # Sort internal edges by weight
        edges = list(subgraph.edges(data=True))
        edges.sort(key=lambda x: x[2].get('weight', 0), reverse=True)
        
        # Reduce strength of all internal edges by 70%
        for source, target, data in edges:
            # Update in our graph
            self.memory_graph[source][target]['weight'] *= 0.3
            
            # Try to update in agent's data structures
            if hasattr(agent, 'coherence_matrix'):
                source_idx = list(self.memory_graph.nodes()).index(source)
                target_idx = list(self.memory_graph.nodes()).index(target)
                agent.coherence_matrix[source_idx][target_idx] *= 0.3
                agent.coherence_matrix[target_idx][source_idx] *= 0.3
        
        logger.info(f"Dissolved cluster by reducing coherence between {len(edges)} memory pairs")
    
    def analyze(self, agent):
        """
        Perform a complete entanglement analysis on an agent.
        
        Args:
            agent: The ΨC agent to analyze
            
        Returns:
            Dict containing analysis results and intervention recommendations
        """
        # Build the memory graph
        self.build_memory_graph(agent)
        
        # Calculate entanglement scores
        self.calculate_entanglement_scores()
        
        # Detect entangled clusters
        self.detect_entangled_clusters()
        
        # Generate intervention recommendations
        interventions = self.recommend_interventions()
        
        # Return analysis results
        return {
            "metrics": self.metrics,
            "critical_memories": list(self.critical_memories),
            "memory_clusters": self.memory_clusters,
            "interventions": interventions,
            "entanglement_scores": self.entanglement_scores
        }
    
    def get_metrics(self):
        """Return current metrics."""
        return self.metrics
    
    def reset(self):
        """Reset the monitor's state."""
        self.memory_graph = None
        self.entanglement_scores = {}
        self.memory_clusters = []
        self.critical_memories = set()
        self.intervention_history = []
        self.metrics = {
            "avg_entanglement": 0.0,
            "max_entanglement": 0.0,
            "critical_count": 0,
            "clustered_memories_pct": 0.0,
            "intervention_count": 0
        }
    
    def _generate_mock_memories(self, count=20) -> Dict[str, Any]:
        """Generate mock memories for testing."""
        memories = {}
        for i in range(count):
            memories[f"mem_{i}"] = {
                "id": f"mem_{i}",
                "content": f"Mock memory {i}",
                "importance": np.random.uniform(0.3, 0.9),
                "embedding": np.random.random(10)
            }
        return memories
    
    def _generate_mock_connections(self, memories) -> List[Tuple[str, str, float]]:
        """Generate mock connections for testing."""
        connections = []
        mem_ids = list(memories.keys())
        
        # Generate a mostly connected graph with some clusters
        for i in range(len(mem_ids)):
            # Connect to 30% of others with varying coherence
            for j in range(i+1, len(mem_ids)):
                if np.random.random() < 0.3:
                    coherence = np.random.uniform(0.4, 0.95)
                    connections.append((mem_ids[i], mem_ids[j], coherence))
                    
            # Ensure some tight clusters
            if i < len(mem_ids) - 3:
                for j in range(1, 4):  # Connect to next 3 nodes with high coherence
                    if i + j < len(mem_ids):
                        coherence = np.random.uniform(0.8, 0.95)
                        connections.append((mem_ids[i], mem_ids[i+j], coherence))
        
        return connections


def analyze_agent_memory_entanglement(agent, apply_fixes=False):
    """
    Utility function to analyze and optionally fix memory entanglement in an agent.
    
    Args:
        agent: ΨC agent to analyze
        apply_fixes: Whether to automatically apply recommended interventions
        
    Returns:
        Dict containing analysis results
    """
    monitor = MemoryEntanglementMonitor()
    results = monitor.analyze(agent)
    
    # Print summary to console
    critical_count = len(results["critical_memories"])
    cluster_count = len(results["memory_clusters"])
    
    print(f"\nMemory Entanglement Analysis:")
    print(f"  Average entanglement: {results['metrics']['avg_entanglement']:.3f}")
    print(f"  Maximum entanglement: {results['metrics']['max_entanglement']:.3f}")
    print(f"  Critical memories: {critical_count}")
    print(f"  Entangled clusters: {cluster_count}")
    
    # Log detailed information
    if critical_count > 0:
        print("\nCritical Memories:")
        for mem_id in results["critical_memories"]:
            score = monitor.entanglement_scores[mem_id]
            print(f"  {mem_id} - Score: {score:.3f}")
    
    if cluster_count > 0:
        print("\nEntangled Clusters:")
        for i, cluster in enumerate(results["memory_clusters"]):
            print(f"  Cluster {i}: {len(cluster)} memories")
    
    # Apply fixes if requested
    if apply_fixes and (critical_count > 0 or cluster_count > 0):
        print("\nApplying interventions...")
        applied = monitor.apply_interventions(agent)
        print(f"Applied {applied} interventions")
        
        # Re-analyze after fixes
        updated_results = monitor.analyze(agent)
        updated_critical = len(updated_results["critical_memories"])
        updated_clusters = len(updated_results["memory_clusters"])
        
        print(f"\nAfter intervention:")
        print(f"  Critical memories: {updated_critical} (was {critical_count})")
        print(f"  Entangled clusters: {updated_clusters} (was {cluster_count})")
    
    return results


if __name__ == "__main__":
    # Simple demo with mock agent
    class MockAgent:
        def __init__(self):
            self.memory_graph = None
            
    # Create a mock agent
    agent = MockAgent()
    
    # Analyze memory entanglement
    results = analyze_agent_memory_entanglement(agent, apply_fixes=True) 