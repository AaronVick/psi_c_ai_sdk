"""
Conflict Resolution Module for Multi-Agent Systems in ΨC-AI SDK

This module provides tools for detecting and resolving conflicts between agents' beliefs,
using various resolution strategies based on trust, evidence, and consensus.
"""

import logging
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set, Any

import networkx as nx

from psi_c_ai_sdk.schema.schema import SchemaGraph, NodeType
from psi_c_ai_sdk.belief.belief_revision import BeliefRevisionSystem
from psi_c_ai_sdk.memory.memory import Memory


logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts that can occur between agents."""
    DIRECT_CONTRADICTION = auto()  # Directly opposing beliefs
    INDIRECT_CONTRADICTION = auto()  # Beliefs that indirectly contradict through inference
    VALUE_MISMATCH = auto()  # Different values for the same attribute
    INCOMPLETE_INFORMATION = auto()  # One agent has information the other doesn't


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts between agents."""
    TRUST_BASED = auto()  # Use trust levels to decide
    EVIDENCE_BASED = auto()  # Use evidence/citations to decide
    RECENCY_BASED = auto()  # More recent information wins
    MAJORITY_CONSENSUS = auto()  # Majority vote among agents
    NEGOTIATION = auto()  # Interactive negotiation process
    CONSERVATIVE = auto()  # Keep both beliefs but mark as potentially conflicting


class ConflictRecord:
    """Record of a detected conflict between agents."""
    
    def __init__(
        self,
        conflict_id: str,
        conflict_type: ConflictType,
        agent_beliefs: Dict[str, Any],
        detection_time: datetime,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a conflict record.
        
        Args:
            conflict_id: Unique identifier for the conflict
            conflict_type: Type of conflict
            agent_beliefs: Dictionary mapping agent IDs to their beliefs
            detection_time: When the conflict was detected
            context: Additional contextual information about the conflict
        """
        self.conflict_id = conflict_id
        self.conflict_type = conflict_type
        self.agent_beliefs = agent_beliefs
        self.detection_time = detection_time
        self.context = context or {}
        self.resolution: Optional[Dict[str, Any]] = None
        self.resolution_time: Optional[datetime] = None
        self.resolution_strategy: Optional[ResolutionStrategy] = None
        
    def set_resolution(
        self,
        resolution: Dict[str, Any],
        strategy: ResolutionStrategy
    ) -> None:
        """
        Set the resolution for this conflict.
        
        Args:
            resolution: The resolution details
            strategy: Strategy used for resolution
        """
        self.resolution = resolution
        self.resolution_time = datetime.now()
        self.resolution_strategy = strategy
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the conflict record to a dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.name,
            "agent_beliefs": self.agent_beliefs,
            "detection_time": self.detection_time.isoformat(),
            "context": self.context,
            "resolution": self.resolution,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "resolution_strategy": self.resolution_strategy.name if self.resolution_strategy else None
        }


class ConflictDetector:
    """Detects conflicts between agent beliefs and knowledge."""
    
    def __init__(self, agent_schemas: Dict[str, SchemaGraph]):
        """
        Initialize the conflict detector.
        
        Args:
            agent_schemas: Dictionary mapping agent IDs to their schema graphs
        """
        self.agent_schemas = agent_schemas
        self.known_conflicts: List[ConflictRecord] = []
        
    def find_direct_contradictions(self) -> List[ConflictRecord]:
        """
        Find direct contradictions between agent beliefs.
        
        Returns:
            List of conflict records for direct contradictions
        """
        contradictions = []
        
        # Get all belief nodes from each agent
        agent_beliefs = {}
        for agent_id, schema in self.agent_schemas.items():
            beliefs = {}
            for node_id, node_data in schema.graph.nodes(data=True):
                if node_data.get("node_type") == NodeType.BELIEF:
                    beliefs[node_id] = node_data
            agent_beliefs[agent_id] = beliefs
        
        # Compare belief content across agents
        belief_content_map = {}
        for agent_id, beliefs in agent_beliefs.items():
            for belief_id, belief_data in beliefs.items():
                content = belief_data.get("content", "")
                if not content:
                    continue
                    
                # Check for direct contradictions by looking at opposing statements
                for other_agent_id, other_beliefs in agent_beliefs.items():
                    if agent_id == other_agent_id:
                        continue
                        
                    for other_belief_id, other_belief_data in other_beliefs.items():
                        other_content = other_belief_data.get("content", "")
                        if not other_content:
                            continue
                            
                        # Check for potential contradictions
                        if self._are_contradictory(content, other_content):
                            conflict_id = f"direct:{agent_id}:{belief_id}:{other_agent_id}:{other_belief_id}"
                            if conflict_id in [c.conflict_id for c in contradictions]:
                                continue
                                
                            contradiction = ConflictRecord(
                                conflict_id=conflict_id,
                                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                                agent_beliefs={
                                    agent_id: {
                                        "belief_id": belief_id,
                                        "content": content,
                                        "confidence": belief_data.get("metadata", {}).get("confidence", 0.5)
                                    },
                                    other_agent_id: {
                                        "belief_id": other_belief_id,
                                        "content": other_content,
                                        "confidence": other_belief_data.get("metadata", {}).get("confidence", 0.5)
                                    }
                                },
                                detection_time=datetime.now(),
                                context={
                                    "detection_method": "direct_content_comparison"
                                }
                            )
                            contradictions.append(contradiction)
        
        # Update known conflicts
        self.known_conflicts.extend(contradictions)
        return contradictions
    
    def _are_contradictory(self, content1: str, content2: str) -> bool:
        """
        Determine if two content strings are contradictory.
        
        This is a simplified implementation. In a real system, this would use
        NLU capabilities to detect semantic contradictions.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            True if the contents contradict each other
        """
        # This is a very simplistic approach that just checks for negations
        # A real implementation would use more sophisticated NLU
        
        # Check for direct negations
        if content1.startswith("not ") and content1[4:] in content2:
            return True
        if content2.startswith("not ") and content2[4:] in content1:
            return True
            
        # Check for "is" vs "is not" patterns
        if " is " in content1 and content1.replace(" is ", " is not ") in content2:
            return True
        if " is " in content2 and content2.replace(" is ", " is not ") in content1:
            return True
            
        # Check for "can" vs "cannot" patterns
        if " can " in content1 and content1.replace(" can ", " cannot ") in content2:
            return True
        if " can " in content2 and content2.replace(" can ", " cannot ") in content1:
            return True
            
        # Many more sophisticated rules would be needed for a real system
        return False
    
    def find_value_mismatches(self) -> List[ConflictRecord]:
        """
        Find mismatches in values across agent beliefs.
        
        Returns:
            List of conflict records for value mismatches
        """
        mismatches = []
        # Implementation would search for cases where agents assign
        # different values to the same attributes/properties
        
        # This is a simplified implementation
        return mismatches


class ConflictResolver:
    """Resolves conflicts between agent beliefs using various strategies."""
    
    def __init__(
        self,
        agent_schemas: Dict[str, SchemaGraph],
        agent_trust_scores: Optional[Dict[str, float]] = None,
        default_strategy: ResolutionStrategy = ResolutionStrategy.TRUST_BASED
    ):
        """
        Initialize the conflict resolver.
        
        Args:
            agent_schemas: Dictionary mapping agent IDs to their schema graphs
            agent_trust_scores: Optional trust scores for each agent
            default_strategy: Default strategy to use for resolution
        """
        self.agent_schemas = agent_schemas
        self.agent_trust_scores = agent_trust_scores or {
            agent_id: 0.5 for agent_id in agent_schemas
        }
        self.default_strategy = default_strategy
        self.resolution_history: List[ConflictRecord] = []
        
    def resolve_conflict(
        self,
        conflict: ConflictRecord,
        strategy: Optional[ResolutionStrategy] = None
    ) -> ConflictRecord:
        """
        Resolve a conflict using the specified strategy.
        
        Args:
            conflict: The conflict to resolve
            strategy: Strategy to use (defaults to the instance default)
            
        Returns:
            Updated conflict record with resolution
        """
        strategy = strategy or self.default_strategy
        
        if strategy == ResolutionStrategy.TRUST_BASED:
            resolution = self._resolve_by_trust(conflict)
        elif strategy == ResolutionStrategy.EVIDENCE_BASED:
            resolution = self._resolve_by_evidence(conflict)
        elif strategy == ResolutionStrategy.RECENCY_BASED:
            resolution = self._resolve_by_recency(conflict)
        elif strategy == ResolutionStrategy.MAJORITY_CONSENSUS:
            resolution = self._resolve_by_consensus(conflict)
        elif strategy == ResolutionStrategy.CONSERVATIVE:
            resolution = self._resolve_conservatively(conflict)
        else:
            # Default to conservative approach if strategy not implemented
            resolution = self._resolve_conservatively(conflict)
            
        # Update the conflict with resolution
        conflict.set_resolution(resolution, strategy)
        self.resolution_history.append(conflict)
        
        return conflict
    
    def _resolve_by_trust(self, conflict: ConflictRecord) -> Dict[str, Any]:
        """
        Resolve conflict based on agent trust scores.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution details
        """
        # Get the agents involved
        agent_beliefs = conflict.agent_beliefs
        agent_ids = list(agent_beliefs.keys())
        
        if len(agent_ids) < 2:
            return {
                "resolution_type": "inconclusive",
                "reason": "Not enough agents involved in conflict",
                "accepted_belief": None
            }
            
        # Calculate weighted scores based on trust and confidence
        scores = {}
        for agent_id, belief_info in agent_beliefs.items():
            trust_score = self.agent_trust_scores.get(agent_id, 0.5)
            confidence = belief_info.get("confidence", 0.5)
            scores[agent_id] = trust_score * confidence
            
        # Find the agent with the highest score
        winner_agent_id = max(scores, key=scores.get)
        winning_belief = agent_beliefs[winner_agent_id]
        
        return {
            "resolution_type": "trust_based",
            "winner": winner_agent_id,
            "accepted_belief": winning_belief,
            "agent_scores": scores,
            "reasoning": f"Agent {winner_agent_id} had the highest trust-weighted score of {scores[winner_agent_id]}"
        }
    
    def _resolve_by_evidence(self, conflict: ConflictRecord) -> Dict[str, Any]:
        """
        Resolve conflict based on evidence provided by agents.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution details
        """
        # This is a placeholder for a more sophisticated implementation
        # that would examine evidence cited by each agent
        return {
            "resolution_type": "inconclusive",
            "reason": "Evidence-based resolution not fully implemented",
            "accepted_belief": None
        }
        
    def _resolve_by_recency(self, conflict: ConflictRecord) -> Dict[str, Any]:
        """
        Resolve conflict based on recency of information.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution details
        """
        # This is a placeholder for a more sophisticated implementation
        # that would examine timestamps of beliefs
        return {
            "resolution_type": "inconclusive",
            "reason": "Recency-based resolution not fully implemented",
            "accepted_belief": None
        }
        
    def _resolve_by_consensus(self, conflict: ConflictRecord) -> Dict[str, Any]:
        """
        Resolve conflict based on majority consensus among agents.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution details
        """
        # This is a placeholder for a more sophisticated implementation
        # that would group similar beliefs and count them
        return {
            "resolution_type": "inconclusive",
            "reason": "Consensus-based resolution not fully implemented",
            "accepted_belief": None
        }
        
    def _resolve_conservatively(self, conflict: ConflictRecord) -> Dict[str, Any]:
        """
        Resolve conflict conservatively by keeping all beliefs but marking as conflicting.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution details
        """
        return {
            "resolution_type": "conservative",
            "reason": "Keeping all beliefs but marking them as potentially conflicting",
            "accepted_beliefs": list(conflict.agent_beliefs.values()),
            "warning": "These beliefs contradict each other and should be treated with caution"
        }
        
    def apply_resolution(self, conflict: ConflictRecord) -> None:
        """
        Apply the resolution to the agent schemas.
        
        This modifies the agent schemas to reflect the resolution decision.
        
        Args:
            conflict: The resolved conflict to apply
        """
        if not conflict.resolution:
            logger.warning(f"Cannot apply resolution for conflict {conflict.conflict_id} - no resolution found")
            return
            
        resolution_type = conflict.resolution.get("resolution_type")
        
        if resolution_type == "trust_based":
            winner = conflict.resolution.get("winner")
            winning_belief = conflict.resolution.get("accepted_belief")
            
            if not winner or not winning_belief:
                return
                
            # Update the schemas to reflect the winning belief
            for agent_id, schema in self.agent_schemas.items():
                if agent_id == winner:
                    # Winner's belief remains unchanged
                    continue
                    
                # For other agents, update their beliefs
                agent_belief_info = conflict.agent_beliefs.get(agent_id)
                if not agent_belief_info:
                    continue
                    
                belief_id = agent_belief_info.get("belief_id")
                if not belief_id or belief_id not in schema.graph:
                    continue
                    
                # Update the node attributes to reflect the winning belief
                schema.graph.nodes[belief_id]["content"] = winning_belief.get("content")
                
                # Add a note about the revision
                if "metadata" not in schema.graph.nodes[belief_id]:
                    schema.graph.nodes[belief_id]["metadata"] = {}
                    
                schema.graph.nodes[belief_id]["metadata"]["revised"] = True
                schema.graph.nodes[belief_id]["metadata"]["revision_source"] = winner
                schema.graph.nodes[belief_id]["metadata"]["original_content"] = agent_belief_info.get("content")
                schema.graph.nodes[belief_id]["metadata"]["revision_time"] = datetime.now().isoformat()
                
        elif resolution_type == "conservative":
            # For conservative resolution, mark all beliefs as potentially conflicting
            for agent_id, agent_belief_info in conflict.agent_beliefs.items():
                schema = self.agent_schemas.get(agent_id)
                if not schema:
                    continue
                    
                belief_id = agent_belief_info.get("belief_id")
                if not belief_id or belief_id not in schema.graph:
                    continue
                    
                # Mark the belief as potentially conflicting
                if "metadata" not in schema.graph.nodes[belief_id]:
                    schema.graph.nodes[belief_id]["metadata"] = {}
                    
                schema.graph.nodes[belief_id]["metadata"]["potentially_conflicting"] = True
                schema.graph.nodes[belief_id]["metadata"]["conflict_id"] = conflict.conflict_id
        
        # Other resolution types would have their own implementation


class MultiAgentCoordinator:
    """
    Coordinates beliefs and knowledge across multiple agents,
    managing conflicts and synchronizing shared knowledge.
    """
    
    def __init__(self, agent_schemas: Dict[str, SchemaGraph]):
        """
        Initialize the multi-agent coordinator.
        
        Args:
            agent_schemas: Dictionary mapping agent IDs to their schema graphs
        """
        self.agent_schemas = agent_schemas
        self.conflict_detector = ConflictDetector(agent_schemas)
        self.conflict_resolver = ConflictResolver(agent_schemas)
        self.shared_beliefs: Dict[str, Any] = {}
        
    def detect_and_resolve_conflicts(self) -> List[ConflictRecord]:
        """
        Detect and resolve all conflicts between agents.
        
        Returns:
            List of resolved conflict records
        """
        # Detect direct contradictions
        conflicts = self.conflict_detector.find_direct_contradictions()
        
        # Detect value mismatches
        conflicts.extend(self.conflict_detector.find_value_mismatches())
        
        # Resolve each conflict
        resolved_conflicts = []
        for conflict in conflicts:
            resolved = self.conflict_resolver.resolve_conflict(conflict)
            self.conflict_resolver.apply_resolution(resolved)
            resolved_conflicts.append(resolved)
            
        return resolved_conflicts
    
    def synchronize_knowledge(self) -> Dict[str, Any]:
        """
        Synchronize shared knowledge across all agents.
        
        Returns:
            Dictionary of statistics about the synchronization
        """
        # Find concepts that exist in multiple agents
        shared_concepts = self._find_shared_concepts()
        
        # Update each agent's schema with shared concepts
        updates_per_agent = {}
        for agent_id, schema in self.agent_schemas.items():
            updates = 0
            for concept_name, concept_info in shared_concepts.items():
                if agent_id not in concept_info["agent_ids"]:
                    # Agent doesn't have this concept, add it
                    new_node_id = f"{concept_name}_{agent_id}"
                    schema.add_node(
                        new_node_id,
                        label=concept_name,
                        node_type=NodeType.CONCEPT,
                        metadata={
                            "shared": True,
                            "source_agents": concept_info["agent_ids"]
                        }
                    )
                    updates += 1
                    
                    # Add edges to related concepts
                    for related_concept, relation in concept_info.get("relations", {}).items():
                        if related_concept in schema.graph:
                            schema.add_edge(
                                new_node_id, 
                                related_concept,
                                label=relation
                            )
            
            updates_per_agent[agent_id] = updates
            
        return {
            "shared_concepts_count": len(shared_concepts),
            "updates_per_agent": updates_per_agent
        }
    
    def _find_shared_concepts(self) -> Dict[str, Any]:
        """
        Find concepts that are shared across multiple agents.
        
        Returns:
            Dictionary mapping concept names to information about the concept
        """
        concept_agents = {}
        concept_relations = {}
        
        # Find which agents have which concepts
        for agent_id, schema in self.agent_schemas.items():
            for node_id, node_data in schema.graph.nodes(data=True):
                if node_data.get("node_type") == NodeType.CONCEPT:
                    concept_name = node_data.get("label", node_id)
                    
                    if concept_name not in concept_agents:
                        concept_agents[concept_name] = []
                        concept_relations[concept_name] = {}
                        
                    concept_agents[concept_name].append(agent_id)
                    
                    # Collect relations for this concept
                    for _, target, edge_data in schema.graph.out_edges(node_id, data=True):
                        target_data = schema.graph.nodes[target]
                        if target_data.get("node_type") == NodeType.CONCEPT:
                            target_name = target_data.get("label", target)
                            relation = edge_data.get("label", "related_to")
                            concept_relations[concept_name][target_name] = relation
        
        # Filter to only concepts that appear in multiple agents
        shared_concepts = {}
        for concept_name, agent_ids in concept_agents.items():
            if len(agent_ids) > 1:
                shared_concepts[concept_name] = {
                    "agent_ids": agent_ids,
                    "relations": concept_relations[concept_name]
                }
                
        return shared_concepts
    
    def find_collaborative_opportunities(self) -> List[Dict[str, Any]]:
        """
        Find opportunities for agent collaboration based on complementary knowledge.
        
        Returns:
            List of collaboration opportunities
        """
        opportunities = []
        
        # This is a simplified implementation that could be expanded
        # to identify areas where agents have complementary expertise
        
        # Look for concepts that one agent has detailed knowledge about
        # but other agents reference without much detail
        all_concepts = set()
        agent_concepts = {agent_id: set() for agent_id in self.agent_schemas}
        
        for agent_id, schema in self.agent_schemas.items():
            for node_id, node_data in schema.graph.nodes(data=True):
                if node_data.get("node_type") == NodeType.CONCEPT:
                    concept_name = node_data.get("label", node_id)
                    all_concepts.add(concept_name)
                    agent_concepts[agent_id].add(concept_name)
        
        # For each concept, check which agents have detailed knowledge
        for concept in all_concepts:
            experts = []
            referencer_only = []
            
            for agent_id, schema in self.agent_schemas.items():
                expertise_level = self._assess_concept_expertise(schema, concept)
                
                if expertise_level > 0.7:  # Arbitrary threshold
                    experts.append(agent_id)
                elif expertise_level > 0:
                    referencer_only.append(agent_id)
            
            if experts and referencer_only:
                opportunities.append({
                    "concept": concept,
                    "experts": experts,
                    "potential_learners": referencer_only,
                    "opportunity_type": "knowledge_sharing"
                })
        
        return opportunities
    
    def _assess_concept_expertise(self, schema: SchemaGraph, concept_name: str) -> float:
        """
        Assess an agent's expertise level for a given concept.
        
        Args:
            schema: The agent's schema graph
            concept_name: Name of the concept to assess
            
        Returns:
            Expertise score between 0.0 and 1.0
        """
        # This is a simplified assessment that could be expanded
        concept_node = None
        
        # Find the concept node
        for node_id, node_data in schema.graph.nodes(data=True):
            if (node_data.get("node_type") == NodeType.CONCEPT and 
                node_data.get("label") == concept_name):
                concept_node = node_id
                break
        
        if not concept_node:
            return 0.0
        
        # Count related beliefs and concepts
        related_beliefs = 0
        related_concepts = 0
        
        for source, target, _ in schema.graph.out_edges(concept_node):
            target_data = schema.graph.nodes[target]
            if target_data.get("node_type") == NodeType.BELIEF:
                related_beliefs += 1
            elif target_data.get("node_type") == NodeType.CONCEPT:
                related_concepts += 1
        
        # Simple scoring function
        score = 0.1  # Base score for having the concept
        score += min(related_beliefs * 0.2, 0.5)  # Up to 0.5 for beliefs
        score += min(related_concepts * 0.1, 0.4)  # Up to 0.4 for related concepts
        
        return min(score, 1.0)  # Cap at 1.0 