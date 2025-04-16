"""
Justification Engine for ΨC-AI SDK.

This module implements the JustificationEngine which provides explainability
features for the ΨC-AI SDK including:
1. Generating explanations for beliefs and memories
2. Visualizing ancestry chains of derived memories
3. Tracing the sources of knowledge
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
import json
import uuid
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import os

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.beliefs.revision import BeliefRevisionSystem, DecisionOutcome

logger = logging.getLogger(__name__)


class ExplanationFormat(Enum):
    """Format options for explanations."""
    TEXT = "text"               # Simple text format
    JSON = "json"               # Structured JSON format
    GRAPH = "graph"             # NetworkX graph format
    VISUALIZATION = "visual"    # Matplotlib visualization


@dataclass
class SourceReference:
    """Reference to a source of information."""
    source_id: str
    source_type: str  # e.g., "user", "document", "web", "inference"
    source_name: Optional[str] = None
    trust_level: float = 0.5  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AncestryNode:
    """Node in an ancestry chain for a memory."""
    memory_id: str
    content: str
    creation_time: float
    source_references: List[SourceReference] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    operation: Optional[str] = None  # e.g., "merge", "derive", "reflect"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefExplanation:
    """Explanation for why a belief or memory exists."""
    memory_id: str
    content: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    coherence_score: Optional[float] = None
    ancestry_chain: List[AncestryNode] = field(default_factory=list)
    source_references: List[SourceReference] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class JustificationEngine:
    """
    Engine for generating explanations and tracing the ancestry of memories and beliefs.
    
    The JustificationEngine provides explainability features for the ΨC-AI SDK,
    allowing users to understand why certain beliefs exist, how they were derived,
    and what sources contributed to them.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        belief_system: Optional[BeliefRevisionSystem] = None,
        coherence_scorer: Optional[CoherenceScorer] = None,
        ancestry_tracker_enabled: bool = True,
        source_tracker_enabled: bool = True,
        explanation_log_path: Optional[str] = None,
        visualization_dir: Optional[str] = None
    ):
        """
        Initialize the JustificationEngine.
        
        Args:
            memory_store: Memory store containing beliefs and memories
            belief_system: Optional belief revision system for decision explanations
            coherence_scorer: Optional coherence scorer for coherence explanations
            ancestry_tracker_enabled: Whether to track memory ancestry
            source_tracker_enabled: Whether to track memory sources
            explanation_log_path: Optional path to log explanations
            visualization_dir: Optional directory to save visualizations
        """
        self.memory_store = memory_store
        self.belief_system = belief_system
        self.coherence_scorer = coherence_scorer
        
        self.ancestry_tracker_enabled = ancestry_tracker_enabled
        self.source_tracker_enabled = source_tracker_enabled
        self.explanation_log_path = explanation_log_path
        self.visualization_dir = visualization_dir
        
        # Memory ancestry tracking
        self.ancestry_graph = nx.DiGraph()
        
        # Source tracking
        self.source_registry: Dict[str, SourceReference] = {}
        
        # Explanation cache
        self.explanation_cache: Dict[str, BeliefExplanation] = {}
        
        # Ensure visualization directory exists if specified
        if self.visualization_dir and not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)
            
        # Set up logging if path provided
        if explanation_log_path:
            explanation_logger = logging.getLogger("explanation")
            explanation_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(explanation_log_path)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            explanation_logger.addHandler(handler)
            
            self.explanation_logger = explanation_logger
        else:
            self.explanation_logger = None
    
    def explain_memory(
        self, 
        memory_id: str,
        format: ExplanationFormat = ExplanationFormat.TEXT,
        include_ancestry: bool = True,
        include_sources: bool = True,
        visualization_path: Optional[str] = None
    ) -> Union[str, Dict[str, Any], nx.DiGraph]:
        """
        Generate an explanation for why a memory exists and how it was derived.
        
        Args:
            memory_id: ID of the memory to explain
            format: Format of the explanation
            include_ancestry: Whether to include ancestry information
            include_sources: Whether to include source information
            visualization_path: Optional path to save visualization
            
        Returns:
            Explanation in the requested format
        """
        # Get the memory
        memory = self.memory_store.get_memory_by_id(memory_id)
        if not memory:
            return f"Memory with ID {memory_id} not found"
        
        # Check cache
        if memory_id in self.explanation_cache:
            explanation = self.explanation_cache[memory_id]
        else:
            # Build explanation
            explanation = self._build_explanation(memory)
            self.explanation_cache[memory_id] = explanation
        
        # Convert to requested format
        if format == ExplanationFormat.TEXT:
            return self._format_explanation_as_text(explanation, include_ancestry, include_sources)
        elif format == ExplanationFormat.JSON:
            return vars(explanation)
        elif format == ExplanationFormat.GRAPH:
            return self._get_ancestry_graph(memory_id)
        elif format == ExplanationFormat.VISUALIZATION:
            return self._visualize_explanation(explanation, visualization_path)
        
        return "Unknown format requested"
    
    def register_memory_operation(
        self,
        memory_id: str,
        operation: str,
        parent_ids: List[str] = None,
        source_references: List[SourceReference] = None
    ) -> None:
        """
        Register a memory operation for ancestry tracking.
        
        Args:
            memory_id: ID of the memory that was created/modified
            operation: Type of operation (e.g., "create", "merge", "reflect")
            parent_ids: Optional list of parent memory IDs
            source_references: Optional list of source references
        """
        if not self.ancestry_tracker_enabled:
            return
            
        memory = self.memory_store.get_memory_by_id(memory_id)
        if not memory:
            logger.warning(f"Cannot register operation for unknown memory: {memory_id}")
            return
            
        # Add node to ancestry graph
        if memory_id not in self.ancestry_graph:
            self.ancestry_graph.add_node(
                memory_id,
                content=memory.content,
                creation_time=memory.creation_time,
                operation=operation,
                metadata=memory.metadata
            )
            
        # Add edges from parents
        if parent_ids:
            for parent_id in parent_ids:
                if parent_id in self.ancestry_graph:
                    self.ancestry_graph.add_edge(parent_id, memory_id)
                    
        # Add source references
        if source_references and self.source_tracker_enabled:
            for ref in source_references:
                self.register_source_reference(memory_id, ref)
    
    def register_source_reference(self, memory_id: str, source_ref: SourceReference) -> None:
        """
        Register a source reference for a memory.
        
        Args:
            memory_id: ID of the memory
            source_ref: Source reference to register
        """
        if not self.source_tracker_enabled:
            return
            
        # Add to source registry if not already there
        if source_ref.source_id not in self.source_registry:
            self.source_registry[source_ref.source_id] = source_ref
            
        # Add reference to memory's metadata
        memory = self.memory_store.get_memory_by_id(memory_id)
        if memory:
            if "sources" not in memory.metadata:
                memory.metadata["sources"] = []
            memory.metadata["sources"].append(source_ref.source_id)
            
            # Update the memory
            self.memory_store.update_memory(memory)
    
    def get_ancestry_chain(self, memory_id: str, max_depth: int = 10) -> List[AncestryNode]:
        """
        Get the ancestry chain for a memory.
        
        Args:
            memory_id: ID of the memory
            max_depth: Maximum depth of ancestry to retrieve
            
        Returns:
            List of ancestry nodes, ordered from oldest to newest
        """
        if not self.ancestry_tracker_enabled or memory_id not in self.ancestry_graph:
            return []
            
        # Get all ancestors using BFS
        ancestors = []
        visited = set()
        queue = [(memory_id, 0)]  # (node, depth)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            
            # Get the memory
            memory = self.memory_store.get_memory_by_id(current_id)
            if not memory:
                continue
                
            # Create ancestry node
            node_data = self.ancestry_graph.nodes[current_id]
            
            # Get parent IDs
            parent_ids = [parent for parent, _ in self.ancestry_graph.in_edges(current_id)]
            
            # Get source references
            source_refs = []
            if "sources" in memory.metadata:
                for source_id in memory.metadata["sources"]:
                    if source_id in self.source_registry:
                        source_refs.append(self.source_registry[source_id])
            
            node = AncestryNode(
                memory_id=current_id,
                content=memory.content,
                creation_time=memory.creation_time,
                parent_ids=parent_ids,
                operation=node_data.get("operation"),
                source_references=source_refs,
                metadata=memory.metadata
            )
            
            ancestors.append(node)
            
            # Add parents to queue
            for parent in parent_ids:
                queue.append((parent, depth + 1))
                
        # Sort by creation time (oldest first)
        ancestors.sort(key=lambda n: n.creation_time)
        
        return ancestors
    
    def visualize_ancestry(
        self, 
        memory_id: str,
        max_depth: int = 5,
        include_content: bool = True,
        show_operations: bool = True,
        show_sources: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the ancestry chain for a memory.
        
        Args:
            memory_id: ID of the memory
            max_depth: Maximum depth of ancestry to visualize
            include_content: Whether to include memory content in nodes
            show_operations: Whether to show operation types on edges
            show_sources: Whether to show sources
            save_path: Optional path to save the visualization
        """
        if not self.ancestry_tracker_enabled or memory_id not in self.ancestry_graph:
            logger.warning(f"No ancestry data available for memory: {memory_id}")
            return
            
        # Extract subgraph for this memory and its ancestors
        ancestors = set()
        queue = [memory_id]
        depth_map = {memory_id: 0}
        
        while queue:
            current = queue.pop(0)
            ancestors.add(current)
            
            if depth_map[current] < max_depth:
                for parent, _ in self.ancestry_graph.in_edges(current):
                    if parent not in depth_map:
                        queue.append(parent)
                        depth_map[parent] = depth_map[current] + 1
        
        subgraph = self.ancestry_graph.subgraph(ancestors)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Use hierarchical layout
        pos = nx.drawing.nx_agraph.graphviz_layout(subgraph, prog='dot', args='-Grankdir=BT')
        
        # Prepare node labels
        node_labels = {}
        for node in subgraph.nodes:
            memory = self.memory_store.get_memory_by_id(node)
            if not memory:
                continue
                
            if include_content:
                # Truncate content for display
                content = memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
                label = f"{node[:6]}...\n{content}"
            else:
                label = f"{node[:10]}..."
                
            node_labels[node] = label
            
        # Prepare edge labels
        edge_labels = {}
        if show_operations:
            for source, target in subgraph.edges:
                target_data = subgraph.nodes[target]
                if "operation" in target_data and target_data["operation"]:
                    edge_labels[(source, target)] = target_data["operation"]
        
        # Draw the graph
        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            labels=node_labels,
            node_color="lightblue",
            node_size=2000,
            font_size=8,
            arrows=True
        )
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(
            subgraph,
            pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Add title
        memory = self.memory_store.get_memory_by_id(memory_id)
        title = f"Ancestry for: {memory.content[:50]}..." if memory else f"Ancestry for: {memory_id}"
        plt.title(title)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return save_path
        elif self.visualization_dir:
            filename = f"ancestry_{memory_id[:10]}_{int(time.time())}.png"
            path = os.path.join(self.visualization_dir, filename)
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        else:
            plt.show()
            plt.close()
            return None
    
    def _build_explanation(self, memory: Memory) -> BeliefExplanation:
        """Build a comprehensive explanation for a memory."""
        # Get basic information
        memory_id = memory.id
        content = memory.content
        
        # Get confidence from memory metadata or estimate it
        confidence = memory.metadata.get("confidence", 0.5)
        
        # Get supporting and contradicting evidence
        supporting_evidence = []
        contradicting_evidence = []
        
        # If we have a coherence scorer, get related memories
        if self.coherence_scorer:
            # Get coherence score
            coherence_score = self.coherence_scorer.calculate_memory_coherence(memory)
            
            # Get supporting memories (high coherence)
            supporting_memories = self.coherence_scorer.find_supporting_memories(
                memory, threshold=0.7, limit=5
            )
            supporting_evidence = [mem.content for mem in supporting_memories]
            
            # Get contradicting memories
            if self.belief_system and self.belief_system.contradiction_detector:
                contradictions = self.belief_system.contradiction_detector.find_contradictions(
                    [memory], self.memory_store.get_all_memories()
                )
                contradicting_evidence = [mem.content for mem, _ in contradictions]
        else:
            coherence_score = None
        
        # Get ancestry chain if tracking enabled
        if self.ancestry_tracker_enabled:
            ancestry_chain = self.get_ancestry_chain(memory_id)
        else:
            ancestry_chain = []
            
        # Get source references
        source_references = []
        if self.source_tracker_enabled and "sources" in memory.metadata:
            for source_id in memory.metadata["sources"]:
                if source_id in self.source_registry:
                    source_references.append(self.source_registry[source_id])
                    
        # Get decision history if available
        decisions = []
        if self.belief_system:
            # Check decision log if available
            decision_history = self.belief_system.get_decisions_for_memory(memory_id)
            decisions = [decision.to_dict() for decision in decision_history]
        
        # Create explanation
        explanation = BeliefExplanation(
            memory_id=memory_id,
            content=content,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            coherence_score=coherence_score,
            ancestry_chain=ancestry_chain,
            source_references=source_references,
            decisions=decisions,
            metadata=memory.metadata
        )
        
        return explanation
    
    def _format_explanation_as_text(
        self, 
        explanation: BeliefExplanation, 
        include_ancestry: bool = True,
        include_sources: bool = True
    ) -> str:
        """Format an explanation as readable text."""
        lines = []
        
        lines.append(f"MEMORY: {explanation.content}")
        lines.append(f"ID: {explanation.memory_id}")
        lines.append(f"Confidence: {explanation.confidence:.2f}")
        
        if explanation.coherence_score is not None:
            lines.append(f"Coherence: {explanation.coherence_score:.2f}")
            
        if explanation.supporting_evidence:
            lines.append("\nSUPPORTING EVIDENCE:")
            for i, evidence in enumerate(explanation.supporting_evidence, 1):
                lines.append(f"  {i}. {evidence}")
                
        if explanation.contradicting_evidence:
            lines.append("\nCONTRADICTING EVIDENCE:")
            for i, evidence in enumerate(explanation.contradicting_evidence, 1):
                lines.append(f"  {i}. {evidence}")
                
        if include_sources and explanation.source_references:
            lines.append("\nSOURCES:")
            for i, source in enumerate(explanation.source_references, 1):
                lines.append(f"  {i}. {source.source_name or source.source_id} "
                           f"(type: {source.source_type}, trust: {source.trust_level:.2f})")
                
        if include_ancestry and explanation.ancestry_chain:
            lines.append("\nANCESTRY:")
            for i, node in enumerate(explanation.ancestry_chain, 1):
                timestamp = datetime.fromtimestamp(node.creation_time).strftime('%Y-%m-%d %H:%M:%S')
                op_text = f" [{node.operation}]" if node.operation else ""
                lines.append(f"  {i}. {timestamp}{op_text}: {node.content[:50]}...")
                
        if explanation.decisions:
            lines.append("\nDECISION HISTORY:")
            for i, decision in enumerate(explanation.decisions, 1):
                reason = decision.get("reason", "Unknown")
                timestamp = datetime.fromtimestamp(decision.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')
                outcome = decision.get("outcome", "Unknown")
                lines.append(f"  {i}. {timestamp} - {outcome} (Reason: {reason})")
                
        return "\n".join(lines)
    
    def _get_ancestry_graph(self, memory_id: str) -> nx.DiGraph:
        """Get the ancestry graph for a memory."""
        if not self.ancestry_tracker_enabled or memory_id not in self.ancestry_graph:
            return nx.DiGraph()
            
        # Find all ancestors
        ancestors = set()
        queue = [memory_id]
        
        while queue:
            current = queue.pop(0)
            ancestors.add(current)
            
            for parent, _ in self.ancestry_graph.in_edges(current):
                if parent not in ancestors:
                    queue.append(parent)
        
        # Extract subgraph
        return self.ancestry_graph.subgraph(ancestors)
    
    def _visualize_explanation(
        self, 
        explanation: BeliefExplanation, 
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Visualize an explanation and return the path to the image file."""
        return self.visualize_ancestry(
            explanation.memory_id,
            save_path=save_path
        ) 