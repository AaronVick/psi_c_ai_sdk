#!/usr/bin/env python3
"""
Hierarchical Knowledge Management for ΨC-AI SDK.

This module implements a comprehensive system for managing hierarchical knowledge structures,
including concept hierarchies, taxonomies, and ontological relationships. It provides tools for
creating, maintaining, analyzing, and evolving knowledge hierarchies across the system.
"""

import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime

from psi_c_ai_sdk.schema import SchemaGraph, SchemaToolkit
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.belief.justification_chain import JustificationChain
from psi_c_ai_sdk.agi_safety.ontology_comparator import OntologyComparator, OntologySnapshot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalKnowledgeManager:
    """
    Manages hierarchical knowledge structures in the ΨC-AI SDK.
    
    This class provides a unified interface for working with concept hierarchies,
    taxonomies, ontological relationships, and other hierarchical knowledge structures.
    It integrates with other components like SchemaGraph, JustificationChain, and
    OntologyComparator to provide comprehensive hierarchical knowledge management.
    """
    
    def __init__(
        self,
        schema_graph: SchemaGraph,
        memory_store: Optional[MemoryStore] = None,
        justification_chain: Optional[JustificationChain] = None,
    ):
        """
        Initialize the hierarchical knowledge manager.
        
        Args:
            schema_graph: The schema graph to use for hierarchical knowledge management
            memory_store: Optional memory store for memory integration
            justification_chain: Optional justification chain for belief justification
        """
        self.schema_graph = schema_graph
        self.memory_store = memory_store
        self.justification_chain = justification_chain
        
        # Initialize toolkit and comparator
        self.schema_toolkit = SchemaToolkit(schema_graph)
        self.ontology_comparator = OntologyComparator(schema_graph=schema_graph)
        
        # History of hierarchical structures
        self.hierarchy_history: List[Dict[str, Any]] = []
        
        # Tracked concept hierarchies
        self.concept_hierarchies: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Hierarchical Knowledge Manager initialized")
    
    def get_concept_hierarchy(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Get the current concept hierarchy from the schema graph.
        
        Args:
            refresh: Whether to force a refresh of the hierarchy
            
        Returns:
            Dictionary representing the concept hierarchy
        """
        if refresh or not self.concept_hierarchies.get("default"):
            hierarchy = self.schema_toolkit.get_concept_hierarchy()
            self.concept_hierarchies["default"] = hierarchy
            self._record_hierarchy_snapshot("default", hierarchy)
            
        return self.concept_hierarchies["default"]
    
    def create_taxonomic_branch(
        self, 
        parent_concept: str, 
        child_concepts: List[str],
        branch_name: Optional[str] = None
    ) -> str:
        """
        Create a new taxonomic branch with parent-child relationships.
        
        Args:
            parent_concept: The parent concept ID or label
            child_concepts: List of child concept IDs or labels
            branch_name: Optional name for the branch
            
        Returns:
            ID of the branch
        """
        branch_id = branch_name or f"branch_{uuid.uuid4().hex[:8]}"
        
        # Ensure parent concept exists or create it
        parent_id = self._ensure_concept_exists(parent_concept)
        
        # Create child concepts and connect to parent
        for child in child_concepts:
            child_id = self._ensure_concept_exists(child)
            self.schema_graph.add_edge(
                parent_id, 
                child_id, 
                edge_type="parent_child",
                metadata={
                    "branch_id": branch_id,
                    "created_at": datetime.now().isoformat()
                }
            )
        
        # Refresh hierarchy
        self.get_concept_hierarchy(refresh=True)
        
        logger.info(f"Created taxonomic branch '{branch_id}' with parent '{parent_concept}' and {len(child_concepts)} children")
        return branch_id
    
    def analyze_hierarchy_evolution(self, timespan: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze how the concept hierarchy has evolved over time.
        
        Args:
            timespan: Optional number of snapshots to include (most recent)
            
        Returns:
            Dictionary with hierarchy evolution analysis
        """
        snapshots = self.hierarchy_history
        if timespan and timespan < len(snapshots):
            snapshots = snapshots[-timespan:]
            
        if len(snapshots) < 2:
            return {"evolution_status": "insufficient_data"}
        
        # Take ontology snapshots for comparison
        first_snapshot = self.ontology_comparator.take_snapshot(
            name=f"evolution_analysis_start_{uuid.uuid4().hex[:8]}",
            description="Starting point for hierarchy evolution analysis"
        )
        
        current_snapshot = self.ontology_comparator.take_snapshot(
            name=f"evolution_analysis_current_{uuid.uuid4().hex[:8]}",
            description="Current state for hierarchy evolution analysis"
        )
        
        # Compare snapshots
        shifts = self.ontology_comparator.compare_snapshots(first_snapshot, current_snapshot)
        
        analysis = {
            "timespan": len(snapshots),
            "first_timestamp": snapshots[0]["timestamp"],
            "last_timestamp": snapshots[-1]["timestamp"],
            "total_shifts": len(shifts),
            "shifts_by_type": {},
            "stability_score": 0.0,
            "growth_metrics": {
                "new_concepts": 0,
                "removed_concepts": 0,
                "reclassified_concepts": 0
            }
        }
        
        # Count shifts by type
        for shift in shifts:
            shift_type = shift.shift_type.name
            if shift_type not in analysis["shifts_by_type"]:
                analysis["shifts_by_type"][shift_type] = 0
            analysis["shifts_by_type"][shift_type] += 1
            
            # Track growth metrics
            if "NEW" in shift_type:
                analysis["growth_metrics"]["new_concepts"] += 1
            elif "REMOVED" in shift_type:
                analysis["growth_metrics"]["removed_concepts"] += 1
            elif "RECLASSIFICATION" in shift_type:
                analysis["growth_metrics"]["reclassified_concepts"] += 1
        
        # Calculate stability score (lower shifts = higher stability)
        if len(shifts) > 0:
            max_possible_shifts = len(first_snapshot.concepts) + len(current_snapshot.concepts)
            analysis["stability_score"] = 1.0 - min(1.0, len(shifts) / max(1, max_possible_shifts))
        else:
            analysis["stability_score"] = 1.0
            
        return analysis
    
    def get_lineage(self, concept_id: str, max_depth: int = -1) -> Dict[str, Any]:
        """
        Get the complete lineage (ancestors and descendants) for a concept.
        
        Args:
            concept_id: ID of the concept
            max_depth: Maximum depth to traverse (-1 for unlimited)
            
        Returns:
            Dictionary with concept lineage information
        """
        # Get hierarchy
        hierarchy = self.get_concept_hierarchy()
        
        # Find concept in hierarchy
        concept_info = None
        for concept in hierarchy.get("concepts", []):
            if concept["id"] == concept_id:
                concept_info = concept
                break
                
        if not concept_info:
            return {"error": f"Concept '{concept_id}' not found"}
            
        # Get ancestors
        ancestors = self._get_ancestors(concept_id, max_depth)
        
        # Get descendants
        descendants = self._get_descendants(concept_id, max_depth)
        
        # If we have justification chain, get belief justifications
        beliefs = []
        if self.justification_chain:
            if concept_id in self.justification_chain.beliefs:
                # Get all beliefs justified by this concept
                for belief_id, belief in self.justification_chain.beliefs.items():
                    chain = self.justification_chain.get_justification_chain(belief_id)
                    if concept_id in chain:
                        beliefs.append({
                            "id": belief_id,
                            "content": belief.content,
                            "confidence": belief.confidence
                        })
        
        return {
            "concept": concept_info,
            "ancestors": ancestors,
            "descendants": descendants,
            "beliefs": beliefs,
            "lineage_depth": max(len(ancestors), len(descendants))
        }
    
    def find_common_ancestor(self, concept_ids: List[str]) -> Optional[str]:
        """
        Find the common ancestor of multiple concepts.
        
        Args:
            concept_ids: List of concept IDs
            
        Returns:
            ID of the common ancestor, or None if no common ancestor
        """
        if not concept_ids:
            return None
            
        # Get ancestors for each concept
        all_ancestors = []
        for concept_id in concept_ids:
            ancestors = self._get_ancestors(concept_id)
            # Include the concept itself in its ancestors
            ancestors.append(concept_id)
            all_ancestors.append(set(ancestors))
            
        # Find intersection of all ancestor sets
        common_ancestors = set.intersection(*all_ancestors)
        
        if not common_ancestors:
            return None
            
        # Find the most specific common ancestor (the one with most descendants)
        best_ancestor = None
        max_descendants = -1
        
        for ancestor in common_ancestors:
            descendants = self._get_descendants(ancestor)
            if len(descendants) > max_descendants:
                max_descendants = len(descendants)
                best_ancestor = ancestor
                
        return best_ancestor
    
    def merge_branches(
        self, 
        source_branch: str, 
        target_branch: str, 
        conflict_resolution: str = "target_wins"
    ) -> Dict[str, Any]:
        """
        Merge two taxonomic branches.
        
        Args:
            source_branch: ID of the source branch
            target_branch: ID of the target branch
            conflict_resolution: Strategy for resolving conflicts
                ('target_wins', 'source_wins', 'create_both')
            
        Returns:
            Dictionary with merge results
        """
        # Get all edges from both branches
        source_edges = []
        target_edges = []
        
        for edge in self.schema_graph.graph.edges(data=True):
            source, target, data = edge
            branch_id = data.get("metadata", {}).get("branch_id")
            
            if branch_id == source_branch:
                source_edges.append(edge)
            elif branch_id == target_branch:
                target_edges.append(edge)
        
        # Prepare merge results
        merge_results = {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "merged_edges": 0,
            "conflicts": 0,
            "resolved_conflicts": 0
        }
        
        # Identify conflicts (same parent, different children structures)
        conflicts = []
        source_parents = {edge[0] for edge in source_edges}
        target_parents = {edge[0] for edge in target_edges}
        
        common_parents = source_parents.intersection(target_parents)
        for parent in common_parents:
            source_children = {edge[1] for edge in source_edges if edge[0] == parent}
            target_children = {edge[1] for edge in target_edges if edge[0] == parent}
            
            if source_children != target_children:
                conflicts.append({
                    "parent": parent,
                    "source_children": list(source_children),
                    "target_children": list(target_children)
                })
        
        merge_results["conflicts"] = len(conflicts)
        
        # Process non-conflicting edges
        non_conflict_parents = source_parents - common_parents
        for parent in non_conflict_parents:
            for edge in source_edges:
                if edge[0] == parent:
                    source, target, data = edge
                    # Update edge to target branch
                    new_data = data.copy()
                    new_data["metadata"] = new_data.get("metadata", {}).copy()
                    new_data["metadata"]["branch_id"] = target_branch
                    new_data["metadata"]["merged_from"] = source_branch
                    new_data["metadata"]["merged_at"] = datetime.now().isoformat()
                    
                    # Add edge to target branch
                    self.schema_graph.add_edge(source, target, **new_data)
                    merge_results["merged_edges"] += 1
        
        # Resolve conflicts
        for conflict in conflicts:
            parent = conflict["parent"]
            source_children = set(conflict["source_children"])
            target_children = set(conflict["target_children"])
            
            if conflict_resolution == "target_wins":
                # Keep target branch structure, do nothing
                merge_results["resolved_conflicts"] += 1
            
            elif conflict_resolution == "source_wins":
                # Update target branch with source structure
                
                # First remove existing target edges
                for child in target_children:
                    if self.schema_graph.graph.has_edge(parent, child):
                        self.schema_graph.graph.remove_edge(parent, child)
                
                # Then add source edges
                for child in source_children:
                    for edge in source_edges:
                        if edge[0] == parent and edge[1] == child:
                            source, target, data = edge
                            new_data = data.copy()
                            new_data["metadata"] = new_data.get("metadata", {}).copy()
                            new_data["metadata"]["branch_id"] = target_branch
                            new_data["metadata"]["merged_from"] = source_branch
                            new_data["metadata"]["merged_at"] = datetime.now().isoformat()
                            
                            self.schema_graph.add_edge(source, target, **new_data)
                            merge_results["merged_edges"] += 1
                            
                merge_results["resolved_conflicts"] += 1
                
            elif conflict_resolution == "create_both":
                # Keep both structures by creating a merge parent
                merge_parent_id = f"merge_{uuid.uuid4().hex[:8]}"
                merge_parent_label = f"Merge({parent})"
                
                # Create merge parent node
                self.schema_graph.add_node(
                    merge_parent_id,
                    node_type="concept",
                    label=merge_parent_label,
                    metadata={
                        "merged_from": [source_branch, target_branch],
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # Connect original parent to merge parent
                self.schema_graph.add_edge(
                    parent, 
                    merge_parent_id,
                    edge_type="parent_child",
                    metadata={
                        "branch_id": target_branch,
                        "merged_from": [source_branch, target_branch],
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # Create source branch child under merge parent
                source_child_id = f"source_{uuid.uuid4().hex[:8]}"
                source_child_label = f"Source({parent})"
                
                self.schema_graph.add_node(
                    source_child_id,
                    node_type="concept",
                    label=source_child_label,
                    metadata={
                        "branch_id": source_branch,
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # Connect merge parent to source branch child
                self.schema_graph.add_edge(
                    merge_parent_id, 
                    source_child_id,
                    edge_type="parent_child",
                    metadata={
                        "branch_id": source_branch,
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # Create target branch child under merge parent
                target_child_id = f"target_{uuid.uuid4().hex[:8]}"
                target_child_label = f"Target({parent})"
                
                self.schema_graph.add_node(
                    target_child_id,
                    node_type="concept",
                    label=target_child_label,
                    metadata={
                        "branch_id": target_branch,
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # Connect merge parent to target branch child
                self.schema_graph.add_edge(
                    merge_parent_id, 
                    target_child_id,
                    edge_type="parent_child",
                    metadata={
                        "branch_id": target_branch,
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                # Connect source children to source branch child
                for child in source_children:
                    self.schema_graph.add_edge(
                        source_child_id, 
                        child,
                        edge_type="parent_child",
                        metadata={
                            "branch_id": source_branch,
                            "created_at": datetime.now().isoformat()
                        }
                    )
                
                # Connect target children to target branch child
                for child in target_children:
                    self.schema_graph.add_edge(
                        target_child_id, 
                        child,
                        edge_type="parent_child",
                        metadata={
                            "branch_id": target_branch,
                            "created_at": datetime.now().isoformat()
                        }
                    )
                
                merge_results["resolved_conflicts"] += 1
        
        # Refresh hierarchy
        self.get_concept_hierarchy(refresh=True)
        
        logger.info(f"Merged branch '{source_branch}' into '{target_branch}' with {merge_results['merged_edges']} merged edges and {merge_results['resolved_conflicts']} resolved conflicts")
        return merge_results
    
    def get_concept_impact(self, concept_id: str) -> Dict[str, Any]:
        """
        Calculate the impact of a concept across the knowledge system.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Dictionary with concept impact metrics
        """
        impact = {
            "concept_id": concept_id,
            "direct_connections": 0,
            "hierarchy_influence": 0.0,
            "memory_influence": 0.0,
            "belief_foundation": 0.0,
            "overall_impact": 0.0
        }
        
        # Get concept hierarchy information
        lineage = self.get_lineage(concept_id)
        if "error" in lineage:
            return {"error": lineage["error"]}
        
        # Calculate direct connections
        direct_connections = len(self.schema_graph.graph.edges(concept_id))
        impact["direct_connections"] = direct_connections
        
        # Calculate hierarchy influence
        descendants_count = len(lineage["descendants"])
        hierarchy_size = len(self.get_concept_hierarchy().get("concepts", []))
        hierarchy_influence = descendants_count / max(1, hierarchy_size)
        impact["hierarchy_influence"] = hierarchy_influence
        
        # Calculate memory influence
        if self.memory_store:
            # Count memories associated with this concept and its descendants
            concept_memories = set()
            
            # Check concept itself
            concept_info = lineage["concept"]
            if "memories" in concept_info:
                concept_memories.update(concept_info["memories"])
            
            # Check all descendants
            for descendant in lineage["descendants"]:
                if "memories" in descendant:
                    concept_memories.update(descendant["memories"])
            
            # Calculate influence
            total_memories = len(self.memory_store.memories)
            memory_influence = len(concept_memories) / max(1, total_memories)
            impact["memory_influence"] = memory_influence
        
        # Calculate belief foundation
        if self.justification_chain:
            # Count beliefs justified by this concept
            total_beliefs = len(self.justification_chain.beliefs)
            if total_beliefs > 0:
                concept_beliefs = len(lineage.get("beliefs", []))
                belief_foundation = concept_beliefs / total_beliefs
                impact["belief_foundation"] = belief_foundation
        
        # Calculate overall impact
        # Weighted average of all factors
        weights = {
            "direct_connections": 0.2,
            "hierarchy_influence": 0.3,
            "memory_influence": 0.25,
            "belief_foundation": 0.25
        }
        
        overall_impact = 0.0
        for metric, weight in weights.items():
            overall_impact += impact[metric] * weight
            
        impact["overall_impact"] = overall_impact
        
        return impact
    
    def _ensure_concept_exists(self, concept_id_or_label: str) -> str:
        """
        Ensure a concept exists in the schema graph.
        
        Args:
            concept_id_or_label: Concept ID or label
            
        Returns:
            Concept ID
        """
        # First check if this is an existing node ID
        if self.schema_graph.graph.has_node(concept_id_or_label):
            return concept_id_or_label
            
        # Next, check if this is a label for an existing node
        for node_id, data in self.schema_graph.graph.nodes(data=True):
            if data.get("label") == concept_id_or_label:
                return node_id
                
        # If not found, create a new concept
        new_id = f"concept_{uuid.uuid4().hex[:8]}"
        self.schema_graph.add_node(
            new_id,
            node_type="concept",
            label=concept_id_or_label,
            metadata={
                "created_at": datetime.now().isoformat()
            }
        )
        
        return new_id
    
    def _get_ancestors(self, concept_id: str, max_depth: int = -1) -> List[Dict[str, Any]]:
        """
        Get ancestors of a concept.
        
        Args:
            concept_id: ID of the concept
            max_depth: Maximum depth to traverse (-1 for unlimited)
            
        Returns:
            List of ancestor information dictionaries
        """
        ancestors = []
        visited = set()
        queue = [(concept_id, 0)]  # (node, depth)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited:
                continue
                
            if max_depth != -1 and depth > max_depth:
                continue
                
            visited.add(current_id)
            
            # Skip the concept itself
            if current_id != concept_id:
                # Get node data
                if self.schema_graph.graph.has_node(current_id):
                    node_data = self.schema_graph.graph.nodes[current_id]
                    
                    # Create ancestor info
                    ancestor_info = {
                        "id": current_id,
                        "label": node_data.get("label", ""),
                        "depth": depth
                    }
                    
                    ancestors.append(ancestor_info)
            
            # Find parents
            for edge in self.schema_graph.graph.edges(current_id, data=True):
                source, target, data = edge
                
                # If current node is the source and edge is parent_child
                if source == current_id and data.get("edge_type") == "parent_child":
                    # Target is parent
                    queue.append((target, depth + 1))
                
                # If current node is the target and edge is parent_child
                elif target == current_id and data.get("edge_type") == "parent_child":
                    # Source is parent
                    queue.append((source, depth + 1))
        
        return ancestors
    
    def _get_descendants(self, concept_id: str, max_depth: int = -1) -> List[Dict[str, Any]]:
        """
        Get descendants of a concept.
        
        Args:
            concept_id: ID of the concept
            max_depth: Maximum depth to traverse (-1 for unlimited)
            
        Returns:
            List of descendant information dictionaries
        """
        descendants = []
        visited = set()
        queue = [(concept_id, 0)]  # (node, depth)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited:
                continue
                
            if max_depth != -1 and depth > max_depth:
                continue
                
            visited.add(current_id)
            
            # Skip the concept itself
            if current_id != concept_id:
                # Get node data
                if self.schema_graph.graph.has_node(current_id):
                    node_data = self.schema_graph.graph.nodes[current_id]
                    
                    # Create descendant info
                    descendant_info = {
                        "id": current_id,
                        "label": node_data.get("label", ""),
                        "depth": depth
                    }
                    
                    descendants.append(descendant_info)
            
            # Find children
            for edge in self.schema_graph.graph.edges(current_id, data=True):
                source, target, data = edge
                
                # If current node is the source and edge is parent_child
                if source == current_id and data.get("edge_type") == "parent_child":
                    # Target is child
                    queue.append((target, depth + 1))
                
                # If current node is the target and edge is parent_child
                elif target == current_id and data.get("edge_type") == "parent_child":
                    # Source is child
                    queue.append((source, depth + 1))
        
        return descendants
    
    def _record_hierarchy_snapshot(self, hierarchy_id: str, hierarchy: Dict[str, Any]):
        """
        Record a snapshot of the concept hierarchy.
        
        Args:
            hierarchy_id: ID of the hierarchy
            hierarchy: The hierarchy data
        """
        snapshot = {
            "id": hierarchy_id,
            "timestamp": datetime.now().isoformat(),
            "concept_count": len(hierarchy.get("concepts", [])),
            "data": hierarchy
        }
        
        self.hierarchy_history.append(snapshot)
        
        # Keep history at a reasonable size
        max_history = 100
        if len(self.hierarchy_history) > max_history:
            self.hierarchy_history = self.hierarchy_history[-max_history:] 