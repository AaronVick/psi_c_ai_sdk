"""
Schema Belief Justification Chain

This module provides tools for reconstructing the ancestry of beliefs in the agent's 
schema, tracing how they were formed, evolved, and inherited. The justification 
chain helps maintain epistemic transparency by making the provenance of beliefs 
clear and traceable.

The module allows for visualization and analysis of belief dependency networks,
helping identify key foundational beliefs and understand the impact of belief
modifications throughout the system.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class JustificationType(Enum):
    """Types of justification relationships between beliefs."""
    OBSERVATION = "observation"          # Direct observation/experience
    INFERENCE = "inference"              # Logical inference
    TESTIMONY = "testimony"              # Learned from external source
    REFLECTION = "reflection"            # Result of reflection process
    COMBINATION = "combination"          # Combined from multiple beliefs
    REVISION = "revision"                # Revised version of previous belief
    CONTRADICTION_RESOLUTION = "contradiction_resolution"  # Resolved contradiction
    UNKNOWN = "unknown"                  # Unknown justification


@dataclass
class JustificationEvidence:
    """Evidence supporting a justification link."""
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    evidence_type: str = "memory"  # memory, observation, inference, etc.
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    source_id: Optional[str] = None  # Source of evidence (e.g., memory ID)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JustificationEvidence':
        """Create from dictionary."""
        return cls(
            evidence_id=data.get("evidence_id", str(uuid.uuid4())),
            evidence_type=data.get("evidence_type", "memory"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            source_id=data.get("source_id"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class JustificationLink:
    """Link between a belief and its justification."""
    link_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_belief_id: str = ""
    target_belief_id: str = ""
    justification_type: JustificationType = JustificationType.UNKNOWN
    timestamp: float = field(default_factory=time.time)
    strength: float = 1.0  # How strongly this justifies the belief
    evidence: List[JustificationEvidence] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "link_id": self.link_id,
            "source_belief_id": self.source_belief_id,
            "target_belief_id": self.target_belief_id,
            "justification_type": self.justification_type.value,
            "timestamp": self.timestamp,
            "strength": self.strength,
            "evidence": [e.to_dict() for e in self.evidence],
            "explanation": self.explanation,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JustificationLink':
        """Create from dictionary."""
        just_type = data.get("justification_type", "unknown")
        if isinstance(just_type, str):
            try:
                just_type = JustificationType(just_type)
            except ValueError:
                just_type = JustificationType.UNKNOWN
        
        return cls(
            link_id=data.get("link_id", str(uuid.uuid4())),
            source_belief_id=data.get("source_belief_id", ""),
            target_belief_id=data.get("target_belief_id", ""),
            justification_type=just_type,
            timestamp=data.get("timestamp", time.time()),
            strength=data.get("strength", 1.0),
            evidence=[
                JustificationEvidence.from_dict(e) 
                for e in data.get("evidence", [])
            ],
            explanation=data.get("explanation", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class Belief:
    """Representation of a belief in the system."""
    belief_id: str
    content: str
    confidence: float = 1.0
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "belief_id": self.belief_id,
            "content": self.content,
            "confidence": self.confidence,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "category": self.category,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Belief':
        """Create from dictionary."""
        return cls(
            belief_id=data.get("belief_id", ""),
            content=data.get("content", ""),
            confidence=data.get("confidence", 1.0),
            creation_time=data.get("creation_time", time.time()),
            last_updated=data.get("last_updated", time.time()),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


class JustificationChain:
    """
    Manages the belief justification chain for an agent.
    
    This class maintains a directed graph representing the justification
    relationships between beliefs, allowing for tracing and analysis of
    belief ancestry and epistemic lineage.
    """
    
    def __init__(
        self,
        schema_graph: Optional[SchemaGraph] = None,
        memory_store: Optional[MemoryStore] = None
    ):
        """
        Initialize the justification chain.
        
        Args:
            schema_graph: SchemaGraph to integrate with
            memory_store: MemoryStore to integrate with
        """
        self.schema_graph = schema_graph
        self.memory_store = memory_store
        
        # Justification graph (directed)
        self.justification_graph = nx.DiGraph()
        
        # Beliefs by ID
        self.beliefs: Dict[str, Belief] = {}
        
        # Justification links
        self.links: Dict[str, JustificationLink] = {}
        
        # Root beliefs (those without justifications)
        self._root_beliefs: Optional[Set[str]] = None
    
    def add_belief(self, belief: Belief) -> None:
        """
        Add a belief to the justification chain.
        
        Args:
            belief: Belief to add
        """
        # Add to beliefs dictionary
        self.beliefs[belief.belief_id] = belief
        
        # Add to justification graph
        if not self.justification_graph.has_node(belief.belief_id):
            self.justification_graph.add_node(
                belief.belief_id,
                content=belief.content,
                confidence=belief.confidence,
                creation_time=belief.creation_time,
                category=belief.category,
                tags=belief.tags
            )
        else:
            # Update existing node
            self.justification_graph.nodes[belief.belief_id].update({
                "content": belief.content,
                "confidence": belief.confidence,
                "last_updated": belief.last_updated,
                "category": belief.category,
                "tags": belief.tags
            })
        
        # Reset cache
        self._root_beliefs = None
    
    def add_justification(
        self,
        source_belief_id: str,
        target_belief_id: str,
        justification_type: JustificationType,
        strength: float = 1.0,
        evidence: Optional[List[JustificationEvidence]] = None,
        explanation: str = ""
    ) -> Optional[str]:
        """
        Add a justification link between beliefs.
        
        Args:
            source_belief_id: ID of the justifying belief
            target_belief_id: ID of the belief being justified
            justification_type: Type of justification relationship
            strength: Strength of the justification (0-1)
            evidence: Supporting evidence for the justification
            explanation: Explanation of the justification
            
        Returns:
            ID of the created link, or None if failed
        """
        # Verify that both beliefs exist
        if source_belief_id not in self.beliefs or target_belief_id not in self.beliefs:
            logger.error(f"Cannot add justification: one or both beliefs do not exist")
            return None
        
        # Check for self-loops
        if source_belief_id == target_belief_id:
            logger.error(f"Cannot add justification: self-loops are not allowed")
            return None
        
        # Check for cycles
        if self._would_create_cycle(source_belief_id, target_belief_id):
            logger.error(f"Cannot add justification: would create a cycle")
            return None
        
        # Create link
        link = JustificationLink(
            source_belief_id=source_belief_id,
            target_belief_id=target_belief_id,
            justification_type=justification_type,
            strength=strength,
            evidence=evidence or [],
            explanation=explanation
        )
        
        # Add to links dictionary
        self.links[link.link_id] = link
        
        # Add to justification graph
        self.justification_graph.add_edge(
            source_belief_id,
            target_belief_id,
            link_id=link.link_id,
            justification_type=justification_type.value,
            strength=strength,
            timestamp=link.timestamp
        )
        
        # Reset cache
        self._root_beliefs = None
        
        return link.link_id
    
    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """
        Check if adding an edge would create a cycle.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            True if the edge would create a cycle
        """
        # If the target can reach the source, adding source→target would create a cycle
        return nx.has_path(self.justification_graph, target_id, source_id)
    
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """
        Get a belief by ID.
        
        Args:
            belief_id: ID of the belief to get
            
        Returns:
            Belief if found, None otherwise
        """
        return self.beliefs.get(belief_id)
    
    def get_justification_link(self, link_id: str) -> Optional[JustificationLink]:
        """
        Get a justification link by ID.
        
        Args:
            link_id: ID of the link to get
            
        Returns:
            JustificationLink if found, None otherwise
        """
        return self.links.get(link_id)
    
    def get_belief_justifications(self, belief_id: str) -> List[JustificationLink]:
        """
        Get all justification links for a belief.
        
        Args:
            belief_id: ID of the belief
            
        Returns:
            List of justification links pointing to the belief
        """
        if belief_id not in self.beliefs:
            return []
        
        # Get incoming edges
        result = []
        for source_id, _, data in self.justification_graph.in_edges(belief_id, data=True):
            link_id = data.get("link_id")
            if link_id in self.links:
                result.append(self.links[link_id])
        
        return result
    
    def get_belief_consequences(self, belief_id: str) -> List[JustificationLink]:
        """
        Get all beliefs justified by this belief.
        
        Args:
            belief_id: ID of the belief
            
        Returns:
            List of justification links originating from the belief
        """
        if belief_id not in self.beliefs:
            return []
        
        # Get outgoing edges
        result = []
        for _, target_id, data in self.justification_graph.out_edges(belief_id, data=True):
            link_id = data.get("link_id")
            if link_id in self.links:
                result.append(self.links[link_id])
        
        return result
    
    def get_root_beliefs(self) -> List[Belief]:
        """
        Get all root beliefs (those without justifications).
        
        Returns:
            List of root beliefs
        """
        if self._root_beliefs is None:
            # Find nodes with in-degree 0
            self._root_beliefs = {
                node for node, in_degree in self.justification_graph.in_degree()
                if in_degree == 0
            }
        
        return [self.beliefs[belief_id] for belief_id in self._root_beliefs 
                if belief_id in self.beliefs]
    
    def get_justification_chain(self, belief_id: str, max_depth: int = -1) -> nx.DiGraph:
        """
        Get the justification chain for a belief.
        
        Args:
            belief_id: ID of the belief
            max_depth: Maximum depth to traverse (-1 for unlimited)
            
        Returns:
            DiGraph representing the justification chain
        """
        if belief_id not in self.beliefs:
            return nx.DiGraph()
        
        # Create subgraph of predecessors up to max_depth
        chain = nx.DiGraph()
        
        # Start from the belief
        nodes_to_explore = [(belief_id, 0)]
        visited = set()
        
        while nodes_to_explore:
            current_id, depth = nodes_to_explore.pop(0)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            # Add node to chain
            if current_id in self.beliefs:
                belief = self.beliefs[current_id]
                chain.add_node(
                    current_id,
                    content=belief.content,
                    confidence=belief.confidence,
                    creation_time=belief.creation_time,
                    category=belief.category,
                    tags=belief.tags
                )
            
            # Add predecessors if not at max depth
            if max_depth == -1 or depth < max_depth:
                for source_id, _, data in self.justification_graph.in_edges(current_id, data=True):
                    # Add edge to chain
                    chain.add_edge(
                        source_id,
                        current_id,
                        **data
                    )
                    
                    # Add source to exploration queue
                    nodes_to_explore.append((source_id, depth + 1))
        
        return chain
    
    def get_consequence_chain(self, belief_id: str, max_depth: int = -1) -> nx.DiGraph:
        """
        Get the chain of beliefs justified by this belief.
        
        Args:
            belief_id: ID of the belief
            max_depth: Maximum depth to traverse (-1 for unlimited)
            
        Returns:
            DiGraph representing the consequence chain
        """
        if belief_id not in self.beliefs:
            return nx.DiGraph()
        
        # Create subgraph of successors up to max_depth
        chain = nx.DiGraph()
        
        # Start from the belief
        nodes_to_explore = [(belief_id, 0)]
        visited = set()
        
        while nodes_to_explore:
            current_id, depth = nodes_to_explore.pop(0)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            # Add node to chain
            if current_id in self.beliefs:
                belief = self.beliefs[current_id]
                chain.add_node(
                    current_id,
                    content=belief.content,
                    confidence=belief.confidence,
                    creation_time=belief.creation_time,
                    category=belief.category,
                    tags=belief.tags
                )
            
            # Add successors if not at max depth
            if max_depth == -1 or depth < max_depth:
                for _, target_id, data in self.justification_graph.out_edges(current_id, data=True):
                    # Add edge to chain
                    chain.add_edge(
                        current_id,
                        target_id,
                        **data
                    )
                    
                    # Add target to exploration queue
                    nodes_to_explore.append((target_id, depth + 1))
        
        return chain
    
    def calculate_belief_robustness(self, belief_id: str) -> float:
        """
        Calculate the epistemic robustness of a belief based on its justification structure.
        
        This measures how well-supported a belief is by its justifications.
        
        Args:
            belief_id: ID of the belief
            
        Returns:
            Robustness score between 0 and 1
        """
        if belief_id not in self.beliefs:
            return 0.0
        
        # Get justification chain
        chain = self.get_justification_chain(belief_id)
        
        # Root beliefs have lower robustness
        if chain.in_degree(belief_id) == 0:
            return 0.3  # Base robustness for root beliefs
        
        # Calculate direct justification strength
        direct_strength = sum(
            data.get("strength", 0.5)
            for _, _, data in chain.in_edges(belief_id, data=True)
        )
        
        # Normalize direct strength
        direct_robustness = min(1.0, direct_strength / 2.0)
        
        # Calculate indirect justification strength
        indirect_robustness = 0.0
        
        # For each direct justifier, calculate its robustness
        for source_id, _ in chain.in_edges(belief_id):
            # Remove the current belief to avoid cycles
            temp_chain = chain.copy()
            if temp_chain.has_node(belief_id):
                temp_chain.remove_node(belief_id)
                
            # Calculate source robustness
            source_robustness = self._calculate_node_robustness(source_id, temp_chain)
            
            # Add weighted contribution
            edge_data = chain.get_edge_data(source_id, belief_id)
            strength = edge_data.get("strength", 0.5)
            indirect_robustness += source_robustness * strength
        
        # Normalize indirect robustness
        if chain.in_degree(belief_id) > 0:
            indirect_robustness /= chain.in_degree(belief_id)
        
        # Combine direct and indirect robustness
        return 0.4 * direct_robustness + 0.6 * indirect_robustness
    
    def _calculate_node_robustness(self, node_id: str, chain: nx.DiGraph) -> float:
        """
        Calculate robustness for a node in the chain.
        
        Args:
            node_id: ID of the node
            chain: DiGraph representing the chain
            
        Returns:
            Robustness score between 0 and 1
        """
        # Root nodes have base robustness
        if chain.in_degree(node_id) == 0:
            return 0.3
        
        # Calculate strength from direct justifiers
        strength = sum(
            data.get("strength", 0.5)
            for _, _, data in chain.in_edges(node_id, data=True)
        )
        
        # Normalize
        return min(1.0, strength / max(2.0, chain.in_degree(node_id)))
    
    def calculate_belief_impact(self, belief_id: str) -> float:
        """
        Calculate the impact of a belief on the overall belief system.
        
        This measures how many other beliefs depend on this belief.
        
        Args:
            belief_id: ID of the belief
            
        Returns:
            Impact score between 0 and 1
        """
        if belief_id not in self.beliefs:
            return 0.0
        
        # Get consequence chain
        chain = self.get_consequence_chain(belief_id)
        
        # No consequences means no impact
        if chain.out_degree(belief_id) == 0:
            return 0.0
        
        # Calculate direct impact
        direct_count = chain.out_degree(belief_id)
        
        # Calculate total impact (all descendants)
        total_descendants = len(nx.descendants(chain, belief_id))
        
        # Calculate weighted impact score
        # Scale by logarithm to handle large numbers of descendants
        if total_descendants == 0:
            return 0.0
            
        impact = (0.4 * direct_count / 10.0) + (0.6 * min(1.0, total_descendants / 30.0))
        
        return min(1.0, impact)
    
    def find_critical_beliefs(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most critical beliefs in the system based on their impact.
        
        Critical beliefs are those that many other beliefs depend on.
        
        Args:
            top_n: Number of critical beliefs to return
            
        Returns:
            List of (belief_id, impact_score) tuples
        """
        impact_scores = []
        
        for belief_id in self.beliefs:
            impact = self.calculate_belief_impact(belief_id)
            impact_scores.append((belief_id, impact))
        
        # Sort by impact (descending)
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        
        return impact_scores[:top_n]
    
    def find_vulnerable_beliefs(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most vulnerable beliefs in the system based on low robustness.
        
        Vulnerable beliefs are those with weak justification structures.
        
        Args:
            top_n: Number of vulnerable beliefs to return
            
        Returns:
            List of (belief_id, vulnerability_score) tuples
        """
        vulnerability_scores = []
        
        for belief_id in self.beliefs:
            robustness = self.calculate_belief_robustness(belief_id)
            impact = self.calculate_belief_impact(belief_id)
            
            # Vulnerability is higher for high-impact beliefs with low robustness
            vulnerability = (1.0 - robustness) * (0.5 + 0.5 * impact)
            
            vulnerability_scores.append((belief_id, vulnerability))
        
        # Sort by vulnerability (descending)
        vulnerability_scores.sort(key=lambda x: x[1], reverse=True)
        
        return vulnerability_scores[:top_n]
    
    def visualize_justification_chain(
        self,
        belief_id: str,
        max_depth: int = 3,
        filename: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Visualize the justification chain for a belief.
        
        Args:
            belief_id: ID of the belief
            max_depth: Maximum depth to visualize
            filename: File to save the visualization to (optional)
            show: Whether to display the visualization
            
        Returns:
            Matplotlib figure if show is True, None otherwise
        """
        if belief_id not in self.beliefs:
            logger.error(f"Cannot visualize: belief {belief_id} not found")
            return None
        
        # Get the justification chain
        chain = self.get_justification_chain(belief_id, max_depth)
        
        if len(chain) == 0:
            logger.warning(f"Empty justification chain for belief {belief_id}")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(chain, k=0.15, iterations=50)
        
        # Create node labels
        node_labels = {}
        for node in chain.nodes():
            content = chain.nodes[node].get("content", "")
            # Truncate long content
            if len(content) > 50:
                content = content[:47] + "..."
            node_labels[node] = content
        
        # Create edge labels
        edge_labels = {}
        for source, target, data in chain.edges(data=True):
            just_type = data.get("justification_type", "unknown")
            strength = data.get("strength", 0.5)
            edge_labels[(source, target)] = f"{just_type}\n({strength:.2f})"
        
        # Calculate node colors based on robustness
        node_colors = []
        for node in chain.nodes():
            robustness = self._calculate_node_robustness(node, chain)
            # Color from red (low robustness) to green (high robustness)
            node_colors.append((1.0 - robustness, robustness, 0.2))
        
        # Calculate node sizes based on impact
        node_sizes = []
        for node in chain.nodes():
            impact = len(list(nx.descendants(chain, node))) + 1
            node_sizes.append(300 + 100 * min(5, impact))
        
        # Draw the graph
        nx.draw_networkx_nodes(chain, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(chain, pos, arrowsize=20, width=2.0, edge_color='black')
        nx.draw_networkx_labels(chain, pos, labels=node_labels, font_size=10)
        nx.draw_networkx_edge_labels(chain, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Justification Chain for: {node_labels[belief_id]}")
        plt.axis('off')
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
            return plt.gcf()
        else:
            plt.close()
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the justification chain to a dictionary.
        
        Returns:
            Dictionary representation of the justification chain
        """
        return {
            "beliefs": {
                belief_id: belief.to_dict() 
                for belief_id, belief in self.beliefs.items()
            },
            "links": {
                link_id: link.to_dict() 
                for link_id, link in self.links.items()
            }
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        schema_graph: Optional[SchemaGraph] = None,
        memory_store: Optional[MemoryStore] = None
    ) -> 'JustificationChain':
        """
        Create a justification chain from a dictionary.
        
        Args:
            data: Dictionary representation of the justification chain
            schema_graph: SchemaGraph to integrate with
            memory_store: MemoryStore to integrate with
            
        Returns:
            JustificationChain instance
        """
        chain = cls(schema_graph, memory_store)
        
        # Add beliefs
        for belief_id, belief_data in data.get("beliefs", {}).items():
            chain.add_belief(Belief.from_dict(belief_data))
        
        # Add links
        for link_id, link_data in data.get("links", {}).items():
            link = JustificationLink.from_dict(link_data)
            chain.links[link_id] = link
            
            # Add edge to graph
            chain.justification_graph.add_edge(
                link.source_belief_id,
                link.target_belief_id,
                link_id=link_id,
                justification_type=link.justification_type.value,
                strength=link.strength,
                timestamp=link.timestamp
            )
        
        return chain
    
    def save_to_file(self, filepath: Union[str, Path]) -> bool:
        """
        Save the justification chain to a file.
        
        Args:
            filepath: Path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save justification chain: {e}")
            return False
    
    @classmethod
    def load_from_file(
        cls,
        filepath: Union[str, Path],
        schema_graph: Optional[SchemaGraph] = None,
        memory_store: Optional[MemoryStore] = None
    ) -> Optional['JustificationChain']:
        """
        Load a justification chain from a file.
        
        Args:
            filepath: Path to load from
            schema_graph: SchemaGraph to integrate with
            memory_store: MemoryStore to integrate with
            
        Returns:
            JustificationChain if successful, None otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data, schema_graph, memory_store)
        except Exception as e:
            logger.error(f"Failed to load justification chain: {e}")
            return None
    
    def extract_from_schema(self) -> int:
        """
        Extract belief justifications from the schema graph.
        
        Returns:
            Number of beliefs extracted
        """
        if not self.schema_graph:
            logger.warning("No schema graph provided, cannot extract beliefs")
            return 0
        
        # Count how many new beliefs are added
        new_belief_count = 0
        
        # Find belief nodes in the schema
        for node_id, data in self.schema_graph.graph.nodes(data=True):
            # Check if this is a belief node
            node_type = data.get("node_type", "")
            if node_type in ["belief", "concept"]:
                content = data.get("content", "")
                if not content:
                    continue
                
                # Create belief if not exists
                if node_id not in self.beliefs:
                    belief = Belief(
                        belief_id=node_id,
                        content=content,
                        confidence=data.get("confidence", 1.0),
                        creation_time=data.get("creation_time", time.time()),
                        last_updated=data.get("last_updated", time.time()),
                        category=node_type,
                        tags=data.get("tags", []),
                        metadata=data.get("metadata", {})
                    )
                    self.add_belief(belief)
                    new_belief_count += 1
        
        # Extract justification relationships
        for source_id, target_id, data in self.schema_graph.graph.edges(data=True):
            # Skip if either node is not in our beliefs
            if source_id not in self.beliefs or target_id not in self.beliefs:
                continue
            
            # Check if this is a justification edge
            edge_type = data.get("edge_type", "")
            if edge_type in ["supports", "justifies", "infers", "derives"]:
                # Determine justification type
                if edge_type == "supports":
                    just_type = JustificationType.TESTIMONY
                elif edge_type == "justifies":
                    just_type = JustificationType.INFERENCE
                elif edge_type == "infers":
                    just_type = JustificationType.INFERENCE
                elif edge_type == "derives":
                    just_type = JustificationType.REFLECTION
                else:
                    just_type = JustificationType.UNKNOWN
                
                # Add justification link if not would create a cycle
                if not self._would_create_cycle(source_id, target_id):
                    self.add_justification(
                        source_belief_id=source_id,
                        target_belief_id=target_id,
                        justification_type=just_type,
                        strength=data.get("weight", 0.5),
                        explanation=data.get("description", "")
                    )
        
        return new_belief_count
    
    def integrate_with_schema(self) -> int:
        """
        Integrate beliefs and justifications into the schema graph.
        
        Returns:
            Number of updates made to schema
        """
        if not self.schema_graph:
            logger.warning("No schema graph provided, cannot integrate with schema")
            return 0
        
        update_count = 0
        
        # Add beliefs to schema
        for belief_id, belief in self.beliefs.items():
            # Check if node exists
            if self.schema_graph.graph.has_node(belief_id):
                # Update existing node
                self.schema_graph.graph.nodes[belief_id].update({
                    "content": belief.content,
                    "confidence": belief.confidence,
                    "last_updated": belief.last_updated,
                    "tags": belief.tags,
                    "metadata": belief.metadata
                })
            else:
                # Create new node
                self.schema_graph.graph.add_node(
                    belief_id,
                    node_type="belief",
                    content=belief.content,
                    confidence=belief.confidence,
                    creation_time=belief.creation_time,
                    last_updated=belief.last_updated,
                    tags=belief.tags,
                    metadata=belief.metadata
                )
                
            update_count += 1
        
        # Add justification links to schema
        for link_id, link in self.links.items():
            # Get source and target nodes
            source_id = link.source_belief_id
            target_id = link.target_belief_id
            
            # Skip if either node doesn't exist in schema
            if not self.schema_graph.graph.has_node(source_id) or \
               not self.schema_graph.graph.has_node(target_id):
                continue
            
            # Determine edge type
            if link.justification_type == JustificationType.INFERENCE:
                edge_type = "infers"
            elif link.justification_type == JustificationType.TESTIMONY:
                edge_type = "supports"
            elif link.justification_type == JustificationType.REFLECTION:
                edge_type = "derives"
            else:
                edge_type = "justifies"
            
            # Add/update edge
            if self.schema_graph.graph.has_edge(source_id, target_id):
                # Update existing edge
                self.schema_graph.graph[source_id][target_id].update({
                    "edge_type": edge_type,
                    "weight": link.strength,
                    "description": link.explanation,
                    "justification_link_id": link_id
                })
            else:
                # Add new edge
                self.schema_graph.graph.add_edge(
                    source_id,
                    target_id,
                    edge_type=edge_type,
                    weight=link.strength,
                    description=link.explanation,
                    justification_link_id=link_id,
                    timestamp=link.timestamp
                )
                
            update_count += 1
        
        return update_count


def create_belief_from_memory(
    memory_content: str,
    belief_id: Optional[str] = None,
    confidence: float = 1.0,
    category: str = "memory-derived",
    tags: Optional[List[str]] = None
) -> Belief:
    """
    Create a belief from memory content.
    
    Args:
        memory_content: Content of the memory
        belief_id: Optional ID for the belief
        confidence: Confidence in the belief
        category: Category of the belief
        tags: Tags for the belief
        
    Returns:
        Created belief
    """
    return Belief(
        belief_id=belief_id or str(uuid.uuid4()),
        content=memory_content,
        confidence=confidence,
        category=category,
        tags=tags or ["memory-derived"]
    )


def create_evidence_from_memory(
    memory_content: str,
    memory_id: Optional[str] = None,
    evidence_type: str = "memory",
    confidence: float = 1.0
) -> JustificationEvidence:
    """
    Create justification evidence from memory content.
    
    Args:
        memory_content: Content of the memory
        memory_id: ID of the memory
        evidence_type: Type of evidence
        confidence: Confidence in the evidence
        
    Returns:
        Created evidence
    """
    return JustificationEvidence(
        evidence_type=evidence_type,
        content=memory_content,
        source_id=memory_id,
        confidence=confidence
    ) 