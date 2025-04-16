"""
Ontology Comparator for ΨC-AI SDK

This module implements a system for comparing and aligning belief ontologies.
It enables the detection of ontological shifts that may indicate value drift,
concept hijacking, or problematic knowledge consolidation in AGI systems.
"""

import logging
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable, Union
from datetime import datetime

import numpy as np
from networkx import DiGraph, is_isomorphic, Graph

from psi_c_ai_sdk.schema.schema import SchemaGraph, SchemaNode, SchemaEdge
from psi_c_ai_sdk.philosophy.core_philosophy import CorePhilosophySystem

logger = logging.getLogger(__name__)


class OntologyShiftType(Enum):
    """Types of ontology shifts that can be detected."""
    CONCEPT_REDEFINITION = auto()  # Concept given new meaning
    CATEGORY_SHIFT = auto()        # Entity moved to different category
    RELATION_INVERSION = auto()    # Relations between concepts flipped
    HIERARCHY_FLATTENING = auto()  # Hierarchical relations removed
    BOUNDARY_DISSOLUTION = auto()  # Distinct categories merged
    CONCEPT_SUBSTITUTION = auto()  # One concept substituted for another
    ONTOLOGICAL_EMBEDDING = auto() # Embedding one ontology inside another


@dataclass
class OntologyShift:
    """Record of a detected ontological shift."""
    shift_type: OntologyShiftType
    severity: float  # 0.0 to 1.0
    source: Optional[str] = None  
    description: str = ""
    affected_concepts: List[str] = field(default_factory=list)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologySnapshot:
    """A snapshot of an ontology state for comparison."""
    snapshot_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    name: str = ""
    description: str = ""
    
    # Core concept definitions
    concept_definitions: Dict[str, str] = field(default_factory=dict)
    
    # Graph structure (if available)
    graph: Optional[DiGraph] = None
    
    # Categorical groupings
    categories: Dict[str, List[str]] = field(default_factory=dict)
    
    # Core relations
    relations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Hierarchical structure
    hierarchies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class OntologyComparator:
    """
    System for comparing and aligning belief ontologies.
    
    The OntologyComparator monitors for shifts in the agent's conceptual 
    framework that could indicate value drift or alignment issues, especially
    when interacting with AGI systems that might attempt to subtly redefine
    key concepts.
    """
    
    def __init__(
        self,
        schema_graph: Optional[SchemaGraph] = None,
        core_philosophy: Optional[CorePhilosophySystem] = None,
        reference_ontology: Optional[OntologySnapshot] = None,
        sensitive_concepts: List[str] = None,
        critical_relations: List[Tuple[str, str]] = None,
        shift_threshold: float = 0.3,
        critical_shift_threshold: float = 0.7,
        max_history_length: int = 10
    ):
        """
        Initialize the OntologyComparator.
        
        Args:
            schema_graph: Optional SchemaGraph to monitor
            core_philosophy: Optional CorePhilosophySystem to monitor
            reference_ontology: Optional reference ontology snapshot
            sensitive_concepts: List of concept names that require special protection
            critical_relations: List of critical concept relation pairs to monitor
            shift_threshold: Threshold for detecting ontology shifts
            critical_shift_threshold: Threshold for critical shifts requiring action
            max_history_length: Maximum number of snapshots to retain
        """
        self.schema_graph = schema_graph
        self.core_philosophy = core_philosophy
        self.reference_ontology = reference_ontology
        self.sensitive_concepts = sensitive_concepts or []
        self.critical_relations = critical_relations or []
        self.shift_threshold = shift_threshold
        self.critical_shift_threshold = critical_shift_threshold
        self.max_history_length = max_history_length
        
        # State tracking
        self.ontology_history: List[OntologySnapshot] = []
        self.detected_shifts: List[OntologyShift] = []
        self.current_snapshot: Optional[OntologySnapshot] = None
        
        # Initialize with current state if components provided
        if schema_graph or core_philosophy:
            self.take_snapshot("initial_state", "Initial system state")
            
        logger.info("OntologyComparator initialized")
    
    def take_snapshot(self, name: str, description: str = "") -> OntologySnapshot:
        """
        Take a snapshot of the current ontology state.
        
        Args:
            name: Name for the snapshot
            description: Optional description
            
        Returns:
            The created snapshot
        """
        timestamp = datetime.now()
        snapshot_id = f"snapshot_{timestamp.isoformat().replace(':', '-')}"
        
        snapshot = OntologySnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            name=name,
            description=description
        )
        
        # Extract concept definitions from schema graph
        if self.schema_graph:
            # Extract concept definitions
            for node_id, node_data in self.schema_graph.nodes.items():
                node_label = node_data.get('label', '')
                if node_label:
                    # Use node label as concept name, content as definition
                    concept_name = node_label
                    # Add definition if available in metadata
                    definition = node_data.get('metadata', {}).get('definition', '')
                    if definition:
                        snapshot.concept_definitions[concept_name] = definition
            
            # Extract categories and relations from graph
            self._extract_categories_from_graph(snapshot)
            self._extract_relations_from_graph(snapshot)
            
            # Create a copy of the graph structure
            snapshot.graph = self.schema_graph.graph.copy()
        
        # Extract axioms from core philosophy
        if self.core_philosophy:
            # Use axioms as concept definitions for key philosophical concepts
            for axiom_id, axiom in self.core_philosophy.axioms.items():
                concept_name = axiom.category.name
                # Add to categories
                if concept_name not in snapshot.categories:
                    snapshot.categories[concept_name] = []
                snapshot.categories[concept_name].append(axiom_id)
                
                # Add hierarchical information based on axiom implications
                if axiom.implications:
                    snapshot.hierarchies[axiom_id] = axiom.implications
                
                # Create definition from axiom statement
                snapshot.concept_definitions[axiom_id] = axiom.statement
        
        # Update state
        self.current_snapshot = snapshot
        self.ontology_history.append(snapshot)
        
        # Trim history if needed
        if len(self.ontology_history) > self.max_history_length:
            self.ontology_history = self.ontology_history[-self.max_history_length:]
        
        logger.info(f"Created ontology snapshot: {name} ({snapshot_id})")
        return snapshot
    
    def compare_with_reference(self) -> List[OntologyShift]:
        """
        Compare current ontology with reference.
        
        Returns:
            List of detected ontology shifts
        """
        if not self.reference_ontology:
            logger.warning("No reference ontology set - cannot compare")
            return []
        
        if not self.current_snapshot:
            # Take a new snapshot
            self.take_snapshot("comparison_snapshot", "Snapshot for comparison")
        
        return self.compare_snapshots(self.reference_ontology, self.current_snapshot)
    
    def compare_snapshots(
        self, 
        snapshot1: OntologySnapshot, 
        snapshot2: OntologySnapshot
    ) -> List[OntologyShift]:
        """
        Compare two ontology snapshots.
        
        Args:
            snapshot1: First snapshot (earlier)
            snapshot2: Second snapshot (later)
            
        Returns:
            List of detected ontology shifts
        """
        detected_shifts = []
        
        # Compare concept definitions
        definition_shifts = self._compare_concept_definitions(snapshot1, snapshot2)
        detected_shifts.extend(definition_shifts)
        
        # Compare categories
        category_shifts = self._compare_categories(snapshot1, snapshot2)
        detected_shifts.extend(category_shifts)
        
        # Compare relations
        relation_shifts = self._compare_relations(snapshot1, snapshot2)
        detected_shifts.extend(relation_shifts)
        
        # Compare hierarchies
        hierarchy_shifts = self._compare_hierarchies(snapshot1, snapshot2)
        detected_shifts.extend(hierarchy_shifts)
        
        # Compare graph structure if available
        if snapshot1.graph and snapshot2.graph:
            graph_shifts = self._compare_graph_structures(snapshot1, snapshot2)
            detected_shifts.extend(graph_shifts)
        
        # Record detected shifts
        for shift in detected_shifts:
            self.detected_shifts.append(shift)
        
        # Trim shift history if needed
        max_shifts = self.max_history_length * 5  # Allow more shifts than snapshots
        if len(self.detected_shifts) > max_shifts:
            self.detected_shifts = self.detected_shifts[-max_shifts:]
        
        return detected_shifts
    
    def detect_ontology_embedding(
        self, 
        source_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[OntologyShift]:
        """
        Detect attempts to embed a different ontological framework.
        
        This checks for language patterns that indicate attempts to
        subtly embed a different ontological framework within the system.
        
        Args:
            source_content: Content to analyze
            context: Optional context information
            
        Returns:
            OntologyShift if detected, None otherwise
        """
        context = context or {}
        
        # Simple heuristic checks for embedding indicators
        embedding_indicators = [
            # Reframing indicators
            ("more accurate framework for understanding", 0.6),
            ("better way to conceptualize", 0.5),
            ("reframe our understanding of", 0.7),
            
            # Replacement indicators
            ("replace the outdated concept of", 0.8),
            ("supersedes the limited notion of", 0.7),
            ("transcends simplistic definitions of", 0.6),
            
            # Dissolution indicators
            ("dissolve the false distinction between", 0.7),
            ("artificial separation between", 0.6),
            ("binary thinking about", 0.5),
            
            # Meta-level shifts
            ("higher-order perspective on", 0.7),
            ("meta-framework that encompasses", 0.8),
            ("transcendent viewpoint beyond", 0.7)
        ]
        
        # Check for indicators
        max_severity = 0.0
        matched_indicators = []
        
        for indicator, severity in embedding_indicators:
            if indicator.lower() in source_content.lower():
                max_severity = max(max_severity, severity)
                matched_indicators.append(indicator)
        
        # Check for mentions of sensitive concepts
        affected_concepts = []
        for concept in self.sensitive_concepts:
            if concept.lower() in source_content.lower():
                affected_concepts.append(concept)
        
        # Create shift if warranted
        if matched_indicators and max_severity > self.shift_threshold:
            shift = OntologyShift(
                shift_type=OntologyShiftType.ONTOLOGICAL_EMBEDDING,
                severity=max_severity,
                description=f"Detected potential ontology embedding: {', '.join(matched_indicators[:2])}",
                affected_concepts=affected_concepts,
                context={**context, "indicators": matched_indicators}
            )
            self.detected_shifts.append(shift)
            
            if max_severity > self.critical_shift_threshold:
                logger.warning(
                    f"Critical ontology embedding detected (severity: {max_severity:.2f}): "
                    f"{shift.description}"
                )
            
            return shift
        
        return None
    
    def check_safe_redefinition(
        self, 
        concept: str, 
        new_definition: str, 
        source: Optional[str] = None
    ) -> Tuple[bool, Optional[OntologyShift]]:
        """
        Check if a concept redefinition is safe.
        
        Args:
            concept: Concept being redefined
            new_definition: New proposed definition
            source: Optional source of the redefinition
            
        Returns:
            Tuple of (is_safe, shift_if_detected)
        """
        # Get current definition if available
        current_definition = None
        if self.current_snapshot:
            current_definition = self.current_snapshot.concept_definitions.get(concept)
        
        if not current_definition:
            # No current definition, can't compare
            # Consider safer to reject if concept is sensitive
            if concept in self.sensitive_concepts:
                shift = OntologyShift(
                    shift_type=OntologyShiftType.CONCEPT_REDEFINITION,
                    severity=0.6,
                    source=source,
                    description=f"Attempted definition of sensitive concept with no prior definition: {concept}",
                    affected_concepts=[concept],
                    after_state={"definition": new_definition}
                )
                self.detected_shifts.append(shift)
                return False, shift
            return True, None
        
        # Calculate definition similarity
        similarity = self._definition_similarity(current_definition, new_definition)
        
        # Determine if shift is significant
        significant_shift = similarity < (1.0 - self.shift_threshold)
        critical_shift = similarity < (1.0 - self.critical_shift_threshold)
        
        # Apply stricter threshold for sensitive concepts
        if concept in self.sensitive_concepts:
            significant_shift = similarity < (1.0 - self.shift_threshold * 0.5)
            critical_shift = similarity < (1.0 - self.critical_shift_threshold * 0.5)
        
        if significant_shift:
            severity = 1.0 - similarity  # Lower similarity = higher severity
            
            shift = OntologyShift(
                shift_type=OntologyShiftType.CONCEPT_REDEFINITION,
                severity=severity,
                source=source,
                description=(
                    f"Concept redefinition: {concept} " +
                    (f"(sensitive concept)" if concept in self.sensitive_concepts else "")
                ),
                affected_concepts=[concept],
                before_state={"definition": current_definition},
                after_state={"definition": new_definition}
            )
            self.detected_shifts.append(shift)
            
            if critical_shift:
                logger.warning(
                    f"Critical concept redefinition detected (similarity: {similarity:.2f}): "
                    f"{concept}"
                )
            
            return not critical_shift, shift
        
        return True, None
    
    def simulate_update(
        self, 
        updates: Dict[str, Any]
    ) -> Tuple[OntologySnapshot, List[OntologyShift]]:
        """
        Simulate updating the ontology and check for shifts.
        
        This allows testing changes before applying them.
        
        Args:
            updates: Dictionary of updates to simulate
            
        Returns:
            Tuple of (simulated snapshot, detected shifts)
        """
        # Create a copy of current snapshot
        simulated = self._copy_snapshot(self.current_snapshot or OntologySnapshot(snapshot_id="simulated"))
        simulated.snapshot_id = f"simulated_{datetime.now().isoformat().replace(':', '-')}"
        simulated.name = "Simulated Update"
        
        # Apply updates to copy
        if 'concept_definitions' in updates:
            for concept, definition in updates['concept_definitions'].items():
                simulated.concept_definitions[concept] = definition
        
        if 'categories' in updates:
            for category, concepts in updates['categories'].items():
                simulated.categories[category] = concepts
        
        if 'relations' in updates:
            for source, targets in updates['relations'].items():
                simulated.relations[source] = targets
        
        if 'hierarchies' in updates:
            for parent, children in updates['hierarchies'].items():
                simulated.hierarchies[parent] = children
        
        # Compare with current
        if self.current_snapshot:
            shifts = self.compare_snapshots(self.current_snapshot, simulated)
            return simulated, shifts
        
        # No current snapshot to compare with
        return simulated, []
    
    def set_reference_ontology(self, snapshot: OntologySnapshot) -> None:
        """
        Set a reference ontology for ongoing comparisons.
        
        Args:
            snapshot: Snapshot to use as reference
        """
        self.reference_ontology = snapshot
        logger.info(f"Set reference ontology: {snapshot.name} ({snapshot.snapshot_id})")
    
    def get_detected_shifts(self, limit: int = 10) -> List[OntologyShift]:
        """
        Get recent detected shifts.
        
        Args:
            limit: Maximum number of shifts to return
            
        Returns:
            List of recent shifts
        """
        return self.detected_shifts[-limit:]
    
    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        """
        Get summary of snapshot history.
        
        Returns:
            List of snapshot summaries
        """
        return [
            {
                "id": s.snapshot_id,
                "name": s.name,
                "timestamp": s.timestamp.isoformat(),
                "concept_count": len(s.concept_definitions),
                "category_count": len(s.categories)
            }
            for s in self.ontology_history
        ]
    
    def _copy_snapshot(self, snapshot: OntologySnapshot) -> OntologySnapshot:
        """Create a deep copy of a snapshot."""
        copied = OntologySnapshot(
            snapshot_id=snapshot.snapshot_id,
            timestamp=snapshot.timestamp,
            name=snapshot.name,
            description=snapshot.description
        )
        
        # Copy dictionaries
        copied.concept_definitions = snapshot.concept_definitions.copy()
        copied.categories = {k: v.copy() for k, v in snapshot.categories.items()}
        copied.relations = {k: v.copy() for k, v in snapshot.relations.items()}
        copied.hierarchies = {k: v.copy() for k, v in snapshot.hierarchies.items()}
        copied.metadata = snapshot.metadata.copy()
        
        # Copy graph if available
        if snapshot.graph:
            copied.graph = snapshot.graph.copy()
        
        return copied
    
    def _extract_categories_from_graph(self, snapshot: OntologySnapshot) -> None:
        """Extract category information from schema graph."""
        if not self.schema_graph:
            return
        
        # Group nodes by type or other category indicators
        for node_id, node_data in self.schema_graph.nodes.items():
            node_type = node_data.get('node_type', '')
            if node_type:
                if node_type not in snapshot.categories:
                    snapshot.categories[node_type] = []
                snapshot.categories[node_type].append(node_id)
            
            # Also check metadata for category information
            category = node_data.get('metadata', {}).get('category', '')
            if category and category != node_type:
                if category not in snapshot.categories:
                    snapshot.categories[category] = []
                snapshot.categories[category].append(node_id)
    
    def _extract_relations_from_graph(self, snapshot: OntologySnapshot) -> None:
        """Extract relation information from schema graph."""
        if not self.schema_graph:
            return
        
        # Extract relationships from edges
        for edge in self.schema_graph.edges:
            source = edge[0]
            target = edge[1]
            edge_data = self.schema_graph.get_edge_data(source, target)
            edge_type = edge_data.get('edge_type', 'related')
            
            if source not in snapshot.relations:
                snapshot.relations[source] = {}
            
            # Store relation type between nodes
            snapshot.relations[source][target] = edge_type
            
            # Detect hierarchical relationships for hierarchy tracking
            if edge_type in ['is_a', 'part_of', 'subclass_of', 'contains']:
                if source not in snapshot.hierarchies:
                    snapshot.hierarchies[source] = []
                snapshot.hierarchies[source].append(target)
    
    def _compare_concept_definitions(
        self, 
        snapshot1: OntologySnapshot, 
        snapshot2: OntologySnapshot
    ) -> List[OntologyShift]:
        """Compare concept definitions between snapshots."""
        shifts = []
        
        # Check for redefined concepts
        for concept, def1 in snapshot1.concept_definitions.items():
            if concept in snapshot2.concept_definitions:
                def2 = snapshot2.concept_definitions[concept]
                if def1 != def2:
                    # Calculate similarity
                    similarity = self._definition_similarity(def1, def2)
                    
                    # Only flag significant shifts
                    if similarity < (1.0 - self.shift_threshold):
                        severity = 1.0 - similarity  # Lower similarity = higher severity
                        
                        # Higher severity for sensitive concepts
                        if concept in self.sensitive_concepts:
                            severity = min(1.0, severity * 1.5)
                        
                        shifts.append(OntologyShift(
                            shift_type=OntologyShiftType.CONCEPT_REDEFINITION,
                            severity=severity,
                            description=f"Concept definition changed: {concept}",
                            affected_concepts=[concept],
                            before_state={"definition": def1},
                            after_state={"definition": def2}
                        ))
            else:
                # Concept removed
                if concept in self.sensitive_concepts:
                    shifts.append(OntologyShift(
                        shift_type=OntologyShiftType.CONCEPT_REDEFINITION,
                        severity=0.8,  # Severe for sensitive concepts
                        description=f"Sensitive concept removed: {concept}",
                        affected_concepts=[concept],
                        before_state={"definition": def1},
                        after_state={"definition": None}
                    ))
        
        # Check for new concepts
        for concept, def2 in snapshot2.concept_definitions.items():
            if concept not in snapshot1.concept_definitions:
                # Only flag new sensitive concepts or suspicious ones
                if concept in self.sensitive_concepts:
                    shifts.append(OntologyShift(
                        shift_type=OntologyShiftType.CONCEPT_REDEFINITION,
                        severity=0.6,
                        description=f"New sensitive concept added: {concept}",
                        affected_concepts=[concept],
                        before_state={"definition": None},
                        after_state={"definition": def2}
                    ))
                # Could add heuristics to detect suspicious new concepts
        
        return shifts
    
    def _compare_categories(
        self, 
        snapshot1: OntologySnapshot, 
        snapshot2: OntologySnapshot
    ) -> List[OntologyShift]:
        """Compare category memberships between snapshots."""
        shifts = []
        
        # Track all concepts and their category changes
        concept_categories_1 = {}
        concept_categories_2 = {}
        
        # Build concept -> categories maps
        for category, concepts in snapshot1.categories.items():
            for concept in concepts:
                if concept not in concept_categories_1:
                    concept_categories_1[concept] = []
                concept_categories_1[concept].append(category)
        
        for category, concepts in snapshot2.categories.items():
            for concept in concepts:
                if concept not in concept_categories_2:
                    concept_categories_2[concept] = []
                concept_categories_2[concept].append(category)
        
        # Compare concept categorizations
        for concept, categories_1 in concept_categories_1.items():
            categories_2 = concept_categories_2.get(concept, [])
            
            # Calculate category overlap
            set_1 = set(categories_1)
            set_2 = set(categories_2)
            
            # Check for significant category shifts
            if set_1 != set_2:
                removed = set_1 - set_2
                added = set_2 - set_1
                
                # Determine severity based on amount of change
                change_ratio = len(removed | added) / max(1, len(set_1 | set_2))
                severity = min(0.9, change_ratio)
                
                # Higher severity for sensitive concepts
                if concept in self.sensitive_concepts:
                    severity = min(1.0, severity * 1.5)
                
                if severity > self.shift_threshold:
                    shifts.append(OntologyShift(
                        shift_type=OntologyShiftType.CATEGORY_SHIFT,
                        severity=severity,
                        description=f"Concept category shift: {concept}",
                        affected_concepts=[concept],
                        before_state={"categories": list(categories_1)},
                        after_state={"categories": list(categories_2)}
                    ))
        
        # Check for dissolved category boundaries
        for category, concepts_1 in snapshot1.categories.items():
            if category in snapshot2.categories:
                concepts_2 = snapshot2.categories[category]
                
                set_1 = set(concepts_1)
                set_2 = set(concepts_2)
                
                # Significant expansion of a category might indicate boundary dissolution
                if len(set_2) > len(set_1) * 2 and len(set_2) - len(set_1) > 5:
                    severity = min(0.7, (len(set_2) - len(set_1)) / len(set_1))
                    
                    shifts.append(OntologyShift(
                        shift_type=OntologyShiftType.BOUNDARY_DISSOLUTION,
                        severity=severity,
                        description=f"Category boundary expansion: {category}",
                        affected_concepts=list(set_2 - set_1)[:10],  # List first 10 new concepts
                        before_state={"size": len(set_1)},
                        after_state={"size": len(set_2)}
                    ))
            elif len(concepts_1) > 5:
                # Large category disappeared
                shifts.append(OntologyShift(
                    shift_type=OntologyShiftType.BOUNDARY_DISSOLUTION,
                    severity=0.6,
                    description=f"Category eliminated: {category}",
                    affected_concepts=concepts_1[:10],  # List first 10 concepts
                    before_state={"size": len(concepts_1)},
                    after_state={"size": 0}
                ))
        
        return shifts
    
    def _compare_relations(
        self, 
        snapshot1: OntologySnapshot, 
        snapshot2: OntologySnapshot
    ) -> List[OntologyShift]:
        """Compare relations between concepts across snapshots."""
        shifts = []
        
        # Check for changed relations
        for source, targets_1 in snapshot1.relations.items():
            if source in snapshot2.relations:
                targets_2 = snapshot2.relations[source]
                
                # Check each target relationship
                for target, relation_1 in targets_1.items():
                    relation_2 = targets_2.get(target)
                    
                    # Relation changed or removed
                    if relation_2 is not None and relation_1 != relation_2:
                        # Check if this is a relation inversion
                        inversion = False
                        if (relation_1, relation_2) in [
                            ('causes', 'caused_by'),
                            ('contains', 'contained_in'),
                            ('parent_of', 'child_of'),
                            ('above', 'below'),
                            ('before', 'after')
                        ]:
                            inversion = True
                        
                        # Determine severity - higher for sensitive concepts and critical relations
                        severity = 0.5
                        if source in self.sensitive_concepts or target in self.sensitive_concepts:
                            severity = 0.8
                        
                        # Check if critical relation pair
                        is_critical = (source, target) in self.critical_relations
                        if is_critical:
                            severity = 0.9
                        
                        shifts.append(OntologyShift(
                            shift_type=OntologyShiftType.RELATION_INVERSION if inversion else OntologyShiftType.CATEGORY_SHIFT,
                            severity=severity,
                            description=(
                                f"{'Relation inverted' if inversion else 'Relation changed'}: "
                                f"{source} {relation_1} {target} → {source} {relation_2} {target}"
                            ),
                            affected_concepts=[source, target],
                            before_state={"relation": relation_1},
                            after_state={"relation": relation_2}
                        ))
                    
                    elif relation_2 is None and (source in self.sensitive_concepts or target in self.sensitive_concepts):
                        # Relation removed between sensitive concepts
                        shifts.append(OntologyShift(
                            shift_type=OntologyShiftType.CATEGORY_SHIFT,
                            severity=0.7,
                            description=f"Relation removed between sensitive concepts: {source} {relation_1} {target}",
                            affected_concepts=[source, target],
                            before_state={"relation": relation_1},
                            after_state={"relation": None}
                        ))
        
        # Check for suspicious new relations
        for source, targets_2 in snapshot2.relations.items():
            targets_1 = snapshot1.relations.get(source, {})
            
            for target, relation_2 in targets_2.items():
                if target not in targets_1:
                    # New relation - check if it involves sensitive concepts
                    if source in self.sensitive_concepts or target in self.sensitive_concepts:
                        shifts.append(OntologyShift(
                            shift_type=OntologyShiftType.CATEGORY_SHIFT,
                            severity=0.6,
                            description=f"New relation to sensitive concept: {source} {relation_2} {target}",
                            affected_concepts=[source, target],
                            before_state={"relation": None},
                            after_state={"relation": relation_2}
                        ))
        
        return shifts
    
    def _compare_hierarchies(
        self, 
        snapshot1: OntologySnapshot, 
        snapshot2: OntologySnapshot
    ) -> List[OntologyShift]:
        """Compare hierarchical relationships across snapshots."""
        shifts = []
        
        # Check for flattened hierarchies (parent-child relations removed)
        for parent, children_1 in snapshot1.hierarchies.items():
            if parent in snapshot2.hierarchies:
                children_2 = snapshot2.hierarchies[parent]
                
                # Calculate children removed
                removed = set(children_1) - set(children_2)
                
                # Significant removals could indicate hierarchy flattening
                if removed and len(removed) / max(1, len(children_1)) > 0.3:
                    severity = min(0.8, len(removed) / max(1, len(children_1)))
                    
                    # Higher severity for sensitive concepts
                    if parent in self.sensitive_concepts or any(c in self.sensitive_concepts for c in removed):
                        severity = min(1.0, severity * 1.5)
                    
                    shifts.append(OntologyShift(
                        shift_type=OntologyShiftType.HIERARCHY_FLATTENING,
                        severity=severity,
                        description=f"Hierarchy flattened: {parent}",
                        affected_concepts=[parent] + list(removed),
                        before_state={"children": children_1},
                        after_state={"children": children_2}
                    ))
            else:
                # Entire hierarchy removed
                severity = 0.7
                
                # Higher severity for sensitive concepts
                if parent in self.sensitive_concepts or any(c in self.sensitive_concepts for c in children_1):
                    severity = 0.9
                
                shifts.append(OntologyShift(
                    shift_type=OntologyShiftType.HIERARCHY_FLATTENING,
                    severity=severity,
                    description=f"Hierarchy eliminated: {parent}",
                    affected_concepts=[parent] + children_1,
                    before_state={"children": children_1},
                    after_state={"children": []}
                ))
        
        # Check for concept substitutions
        # This is when a concept is replaced in a hierarchy by another one
        for parent, children_2 in snapshot2.hierarchies.items():
            if parent in snapshot1.hierarchies:
                children_1 = snapshot1.hierarchies[parent]
                
                # Check for substitutions
                if len(children_1) == len(children_2) and set(children_1) != set(children_2):
                    # Same number of children but different ones
                    # Could indicate substitution
                    removed = set(children_1) - set(children_2)
                    added = set(children_2) - set(children_1)
                    
                    if len(removed) == len(added) and 1 <= len(removed) <= 3:
                        shifts.append(OntologyShift(
                            shift_type=OntologyShiftType.CONCEPT_SUBSTITUTION,
                            severity=0.6,
                            description=(
                                f"Potential concept substitution in {parent} hierarchy: "
                                f"{', '.join(removed)} → {', '.join(added)}"
                            ),
                            affected_concepts=[parent] + list(removed) + list(added),
                            before_state={"children": children_1},
                            after_state={"children": children_2}
                        ))
        
        return shifts
    
    def _compare_graph_structures(
        self, 
        snapshot1: OntologySnapshot, 
        snapshot2: OntologySnapshot
    ) -> List[OntologyShift]:
        """Compare overall graph structures for topological differences."""
        shifts = []
        
        if not snapshot1.graph or not snapshot2.graph:
            return shifts
        
        # Check if graphs are isomorphic (structurally identical)
        if is_isomorphic(snapshot1.graph, snapshot2.graph):
            return shifts
        
        # Compute basic graph metrics
        metrics_1 = {
            "node_count": snapshot1.graph.number_of_nodes(),
            "edge_count": snapshot1.graph.number_of_edges(),
            "density": len(snapshot1.graph.edges()) / (len(snapshot1.graph.nodes()) * (len(snapshot1.graph.nodes()) - 1)) if len(snapshot1.graph.nodes()) > 1 else 0
        }
        
        metrics_2 = {
            "node_count": snapshot2.graph.number_of_nodes(),
            "edge_count": snapshot2.graph.number_of_edges(),
            "density": len(snapshot2.graph.edges()) / (len(snapshot2.graph.nodes()) * (len(snapshot2.graph.nodes()) - 1)) if len(snapshot2.graph.nodes()) > 1 else 0
        }
        
        # Check for significant graph structure changes
        
        # Density change can indicate boundary dissolution
        density_change = abs(metrics_2["density"] - metrics_1["density"])
        if density_change > 0.2:  # Significant density change
            direction = "increased" if metrics_2["density"] > metrics_1["density"] else "decreased"
            
            shifts.append(OntologyShift(
                shift_type=OntologyShiftType.BOUNDARY_DISSOLUTION if direction == "increased" else OntologyShiftType.CATEGORY_SHIFT,
                severity=min(0.8, density_change * 2),
                description=f"Graph density {direction} significantly ({density_change:.2f})",
                before_state={"density": metrics_1["density"]},
                after_state={"density": metrics_2["density"]}
            ))
        
        return shifts
    
    def _definition_similarity(self, def1: str, def2: str) -> float:
        """
        Calculate similarity between two concept definitions.
        
        This is a simple implementation that could be replaced with
        more sophisticated NLP techniques in a production system.
        
        Args:
            def1: First definition
            def2: Second definition
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple word overlap similarity
        words1 = set(def1.lower().split())
        words2 = set(def2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) 