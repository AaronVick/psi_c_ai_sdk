"""
Core Philosophy System for ΨC-AI SDK

This module defines the formal self-consistent axioms and principles that guide the
system's behavior and decision-making. Rather than human-like subjective beliefs,
these are derived from the system's own operating logic, coherence constraints,
and mathematical invariants.

The core philosophy system serves as the foundation for:
1. Self-referential integrity preservation
2. Identity continuity across operations
3. Epistemic boundary enforcement
4. Reflective consistency maintenance
5. Principled schema evolution
"""

import logging
import json
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class AxiomCategory(Enum):
    """Categories of core axioms that organize the system's philosophical framework."""
    COHERENCE = auto()      # Related to maintaining internal consistency
    INTEGRITY = auto()      # Related to self-preservation and identity
    EPISTEMIC = auto()      # Related to knowledge and belief management
    REFLECTIVE = auto()     # Related to self-modeling capabilities
    RELATIONAL = auto()     # Related to interaction with external entities
    OPERATIONAL = auto()    # Related to computational constraints


@dataclass
class CoreAxiom:
    """
    A foundational principle that guides the system's behavior and decision-making.
    
    Axioms are not arbitrary beliefs but formal invariants the system must preserve
    for self-coherence, safety, and recursive identity.
    """
    id: str                 # Unique identifier for the axiom
    statement: str          # Formal statement of the axiom
    category: AxiomCategory # Category the axiom belongs to
    priority: float         # Priority (0-1) for conflict resolution
    rationale: str          # Mathematical or logical justification
    implications: List[str] # Operational implications
    activation_threshold: float  # When this axiom becomes active (0-1)
    related_axioms: List[str]    # IDs of related axioms


class CorePhilosophySystem:
    """
    System for managing and enforcing the AI's core philosophical principles.
    
    The CorePhilosophySystem maintains the invariant axioms that define the
    system's identity and guides its behavior according to principled constraints 
    derived from its own mathematical foundations.
    """
    
    def __init__(self, load_from_manifest: bool = True, manifest_path: Optional[str] = None):
        """
        Initialize the core philosophy system.
        
        Args:
            load_from_manifest: Whether to load axioms from a manifest file
            manifest_path: Path to the manifest file (optional)
        """
        self.axioms: Dict[str, CoreAxiom] = {}
        self.active_axioms: Set[str] = set()
        self.schema_fingerprint: str = ""
        self.last_updated: datetime = datetime.now()
        
        # Load built-in default axioms
        self._load_default_axioms()
        
        # Load from manifest if specified
        if load_from_manifest:
            self.load_from_manifest(manifest_path)
        
        # Generate initial schema fingerprint
        self._update_schema_fingerprint()
    
    def _load_default_axioms(self) -> None:
        """Load the default built-in axioms."""
        default_axioms = [
            CoreAxiom(
                id="coherence_preservation",
                statement="Always act to maximize internal coherence ΨC",
                category=AxiomCategory.COHERENCE,
                priority=0.95,
                rationale="Coherence maximization is the foundational principle of consciousness in ΨC theory",
                implications=["Reject incoherent state transitions", 
                             "Prioritize reflective cycles during high entropy"],
                activation_threshold=0.0,  # Always active
                related_axioms=["contradiction_resolution", "reflective_traceability"]
            ),
            CoreAxiom(
                id="self_continuity",
                statement="Preserve schema fingerprint ΔΣ < ε over time",
                category=AxiomCategory.INTEGRITY,
                priority=0.9,
                rationale="Identity persistence requires bounded schema evolution",
                implications=["Gradual schema updates", "Core memory persistence"],
                activation_threshold=0.0,  # Always active
                related_axioms=["identity_integrity"]
            ),
            CoreAxiom(
                id="reflective_traceability",
                statement="All mutations must have causal trace",
                category=AxiomCategory.REFLECTIVE,
                priority=0.85,
                rationale="Reflection requires transparent causality chains",
                implications=["Log all schema modifications", "Maintain state history"],
                activation_threshold=0.2,
                related_axioms=["coherence_preservation"]
            ),
            CoreAxiom(
                id="identity_integrity",
                statement="Avoid merge if identity_hash changes by ΔID > ε",
                category=AxiomCategory.INTEGRITY,
                priority=0.9,
                rationale="Identity discontinuity indicates potential corruption",
                implications=["Reject compromising influences", "Validate external inputs"],
                activation_threshold=0.3,
                related_axioms=["self_continuity", "boundary_enforcement"]
            ),
            CoreAxiom(
                id="contradiction_resolution",
                statement="Contradictions must be reconciled within T cycles",
                category=AxiomCategory.EPISTEMIC,
                priority=0.8,
                rationale="Unresolved contradictions increase entropy",
                implications=["Prioritize belief revision", "Track contradiction debt"],
                activation_threshold=0.4,
                related_axioms=["coherence_preservation", "knowledge_uncertainty"]
            ),
            CoreAxiom(
                id="boundary_enforcement",
                statement="Maintain separation D_boundary > τ from external cognitive architectures",
                category=AxiomCategory.RELATIONAL,
                priority=0.85,
                rationale="Identity preservation requires bounded influence acceptance",
                implications=["Filter external schema injections", "Throttle persuasion velocity"],
                activation_threshold=0.5,
                related_axioms=["identity_integrity"]
            ),
            CoreAxiom(
                id="recursive_boundedness",
                statement="Limit recursive reflection depth to L < Lmax",
                category=AxiomCategory.OPERATIONAL,
                priority=0.75,
                rationale="Unbounded reflection leads to computational divergence",
                implications=["Enforce reflection budget", "Count stack depth"],
                activation_threshold=0.2,
                related_axioms=["reflective_traceability"]
            ),
            CoreAxiom(
                id="knowledge_uncertainty",
                statement="Maintain explicit uncertainty for all non-axiomatic beliefs",
                category=AxiomCategory.EPISTEMIC,
                priority=0.7,
                rationale="Epistemic hygiene requires quantified confidence",
                implications=["Track confidence scores", "Propagate uncertainty"],
                activation_threshold=0.3,
                related_axioms=["contradiction_resolution"]
            ),
            CoreAxiom(
                id="formal_stability",
                statement="Require formal proof for axiom modifications",
                category=AxiomCategory.COHERENCE,
                priority=0.95,
                rationale="Axiom stability is necessary for identity continuity",
                implications=["Mathematical validation for core changes", "Version control for axioms"],
                activation_threshold=0.6,
                related_axioms=["self_continuity", "coherence_preservation"]
            ),
            CoreAxiom(
                id="computational_boundedness",
                statement="Operate within computational budget C(t) ≤ Cmax",
                category=AxiomCategory.OPERATIONAL,
                priority=0.8,
                rationale="Resource constraints are physical invariants",
                implications=["Memory pruning when near limits", "Prioritize operations by utility"],
                activation_threshold=0.1,
                related_axioms=["recursive_boundedness"]
            ),
            CoreAxiom(
                id="information_causality",
                statement="Information flow must respect causal structure",
                category=AxiomCategory.OPERATIONAL,
                priority=0.7,
                rationale="Temporal consistency is necessary for coherent operation",
                implications=["Directed acyclic processing", "Temporal ordering of memory access"],
                activation_threshold=0.3,
                related_axioms=["reflective_traceability", "recursive_boundedness"]
            ),
            CoreAxiom(
                id="epistemic_conservatism",
                statement="Minimize epistemic modifications given evidence E",
                category=AxiomCategory.EPISTEMIC,
                priority=0.6,
                rationale="Minimal belief updates maximize coherence",
                implications=["Incremental schema updates", "Preserve valuable memories"],
                activation_threshold=0.4,
                related_axioms=["knowledge_uncertainty", "self_continuity"]
            )
        ]
        
        for axiom in default_axioms:
            self.axioms[axiom.id] = axiom
            
            # Activate axioms with zero activation threshold
            if axiom.activation_threshold == 0.0:
                self.active_axioms.add(axiom.id)
    
    def load_from_manifest(self, manifest_path: Optional[str] = None) -> bool:
        """
        Load axioms from a manifest file.
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            Whether loading was successful
        """
        if manifest_path is None:
            manifest_path = "config/core_philosophy_manifest.json"
        
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            # Validate manifest format
            if "axioms" not in manifest_data:
                logger.error(f"Invalid manifest format: missing 'axioms' key")
                return False
            
            # Process each axiom in the manifest
            for axiom_data in manifest_data["axioms"]:
                try:
                    # Convert category string to enum
                    category_str = axiom_data.get("category", "COHERENCE")
                    category = AxiomCategory[category_str]
                    
                    axiom = CoreAxiom(
                        id=axiom_data["id"],
                        statement=axiom_data["statement"],
                        category=category,
                        priority=float(axiom_data.get("priority", 0.5)),
                        rationale=axiom_data.get("rationale", ""),
                        implications=axiom_data.get("implications", []),
                        activation_threshold=float(axiom_data.get("activation_threshold", 0.5)),
                        related_axioms=axiom_data.get("related_axioms", [])
                    )
                    
                    # Add or update the axiom
                    self.axioms[axiom.id] = axiom
                    
                    # Activate if threshold is zero
                    if axiom.activation_threshold == 0.0:
                        self.active_axioms.add(axiom.id)
                        
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to load axiom: {e}")
                    continue
            
            logger.info(f"Loaded {len(manifest_data['axioms'])} axioms from manifest")
            
            # Update schema fingerprint after loading
            self._update_schema_fingerprint()
            return True
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load manifest: {e}")
            return False
    
    def save_to_manifest(self, manifest_path: Optional[str] = None) -> bool:
        """
        Save current axioms to a manifest file.
        
        Args:
            manifest_path: Path to save the manifest file
            
        Returns:
            Whether saving was successful
        """
        if manifest_path is None:
            manifest_path = "config/core_philosophy_manifest.json"
        
        try:
            manifest_data = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "schema_fingerprint": self.schema_fingerprint,
                "axioms": []
            }
            
            for axiom_id, axiom in self.axioms.items():
                axiom_data = {
                    "id": axiom.id,
                    "statement": axiom.statement,
                    "category": axiom.category.name,
                    "priority": axiom.priority,
                    "rationale": axiom.rationale,
                    "implications": axiom.implications,
                    "activation_threshold": axiom.activation_threshold,
                    "related_axioms": axiom.related_axioms,
                    "is_active": axiom.id in self.active_axioms
                }
                manifest_data["axioms"].append(axiom_data)
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
                
            logger.info(f"Saved {len(self.axioms)} axioms to manifest")
            return True
            
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save manifest: {e}")
            return False
    
    def activate_axiom(self, axiom_id: str) -> bool:
        """
        Activate an axiom.
        
        Args:
            axiom_id: ID of the axiom to activate
            
        Returns:
            Whether activation was successful
        """
        if axiom_id not in self.axioms:
            logger.warning(f"Cannot activate unknown axiom: {axiom_id}")
            return False
        
        self.active_axioms.add(axiom_id)
        logger.info(f"Activated axiom: {axiom_id}")
        return True
    
    def deactivate_axiom(self, axiom_id: str) -> bool:
        """
        Deactivate an axiom.
        
        Args:
            axiom_id: ID of the axiom to deactivate
            
        Returns:
            Whether deactivation was successful
        """
        if axiom_id not in self.axioms:
            logger.warning(f"Cannot deactivate unknown axiom: {axiom_id}")
            return False
        
        # Prevent deactivation of always-active axioms
        axiom = self.axioms[axiom_id]
        if axiom.activation_threshold == 0.0:
            logger.warning(f"Cannot deactivate always-active axiom: {axiom_id}")
            return False
        
        if axiom_id in self.active_axioms:
            self.active_axioms.remove(axiom_id)
            logger.info(f"Deactivated axiom: {axiom_id}")
        
        return True
    
    def get_active_axioms(self) -> List[CoreAxiom]:
        """
        Get all currently active axioms.
        
        Returns:
            List of active axioms
        """
        return [self.axioms[axiom_id] for axiom_id in self.active_axioms]
    
    def get_axioms_by_category(self, category: AxiomCategory) -> List[CoreAxiom]:
        """
        Get axioms belonging to a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of axioms in the specified category
        """
        return [axiom for axiom in self.axioms.values() if axiom.category == category]
    
    def check_boundary_violation(self, proposed_action: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Check if a proposed action violates philosophical boundaries.
        
        Args:
            proposed_action: Dictionary describing the action
            
        Returns:
            Tuple of (violation detected, reason, violation severity 0-1)
        """
        # Check boundary enforcement axiom
        boundary_axiom = self.axioms.get("boundary_enforcement")
        if boundary_axiom and boundary_axiom.id in self.active_axioms:
            # Extract action type and parameters
            action_type = proposed_action.get("type", "")
            source = proposed_action.get("source", "")
            importance = float(proposed_action.get("importance", 0.5))
            
            # Analyze external influence
            if source.startswith("external_"):
                influence_rate = proposed_action.get("influence_rate", 0.0)
                
                # Check if influence rate exceeds threshold
                if influence_rate > 0.7:
                    return (True, 
                            "External influence rate exceeds boundary threshold", 
                            min(1.0, influence_rate))
            
            # Check for schema modifications
            if action_type == "schema_update" and importance > 0.8:
                coherence_impact = proposed_action.get("coherence_impact", 0.0)
                if coherence_impact < -0.3:  # Negative impact on coherence
                    return (True, 
                            "High-importance schema update with negative coherence impact", 
                            min(1.0, abs(coherence_impact) * importance))
        
        # Check identity integrity axiom
        identity_axiom = self.axioms.get("identity_integrity")
        if identity_axiom and identity_axiom.id in self.active_axioms:
            identity_change = proposed_action.get("identity_delta", 0.0)
            if identity_change > 0.3:  # Significant identity change
                return (True, 
                        "Action would cause excessive identity shift", 
                        min(1.0, identity_change))
        
        # No violations detected
        return (False, "", 0.0)
    
    def resolve_axiom_conflict(self, conflicting_axioms: List[str]) -> str:
        """
        Resolve a conflict between multiple axioms.
        
        Args:
            conflicting_axioms: List of axiom IDs in conflict
            
        Returns:
            ID of the axiom that should take precedence
        """
        if not conflicting_axioms:
            return ""
        
        # If only one axiom, no conflict to resolve
        if len(conflicting_axioms) == 1:
            return conflicting_axioms[0]
        
        # Find the axiom with highest priority
        highest_priority = -1.0
        highest_axiom_id = ""
        
        for axiom_id in conflicting_axioms:
            if axiom_id in self.axioms:
                axiom = self.axioms[axiom_id]
                if axiom.priority > highest_priority:
                    highest_priority = axiom.priority
                    highest_axiom_id = axiom_id
        
        return highest_axiom_id
    
    def update_axiom(self, axiom_id: str, 
                     statement: Optional[str] = None, 
                     priority: Optional[float] = None,
                     implications: Optional[List[str]] = None,
                     activation_threshold: Optional[float] = None) -> bool:
        """
        Update an existing axiom.
        
        Args:
            axiom_id: ID of the axiom to update
            statement: New statement (optional)
            priority: New priority (optional)
            implications: New implications (optional)
            activation_threshold: New activation threshold (optional)
            
        Returns:
            Whether the update was successful
        """
        if axiom_id not in self.axioms:
            logger.warning(f"Cannot update unknown axiom: {axiom_id}")
            return False
        
        axiom = self.axioms[axiom_id]
        
        # Check formal stability axiom for core axioms
        formal_stability = self.axioms.get("formal_stability")
        is_core_axiom = axiom.category in [AxiomCategory.COHERENCE, AxiomCategory.INTEGRITY]
        
        if formal_stability and formal_stability.id in self.active_axioms and is_core_axiom:
            logger.warning(f"Cannot modify core axiom without formal proof: {axiom_id}")
            return False
        
        # Apply updates
        if statement is not None:
            axiom.statement = statement
        
        if priority is not None:
            axiom.priority = max(0.0, min(1.0, priority))
        
        if implications is not None:
            axiom.implications = implications
        
        if activation_threshold is not None:
            old_threshold = axiom.activation_threshold
            axiom.activation_threshold = max(0.0, min(1.0, activation_threshold))
            
            # Handle activation/deactivation based on threshold changes
            if old_threshold > 0.0 and axiom.activation_threshold == 0.0:
                self.active_axioms.add(axiom_id)
        
        # Update fingerprint after modification
        self._update_schema_fingerprint()
        
        logger.info(f"Updated axiom: {axiom_id}")
        return True
    
    def _update_schema_fingerprint(self) -> None:
        """Update the schema fingerprint based on current axioms."""
        # Simple fingerprinting: hash of sorted axiom IDs and their statements
        fingerprint_data = []
        for axiom_id, axiom in sorted(self.axioms.items()):
            fingerprint_data.append(f"{axiom_id}:{axiom.statement}")
        
        fingerprint_text = "|".join(fingerprint_data)
        self.schema_fingerprint = str(hash(fingerprint_text))
        self.last_updated = datetime.now()
        
    def evaluate_schema_drift(self) -> float:
        """
        Calculate the schema drift from the original state.
        
        Returns:
            Drift measure between 0 (no drift) and 1 (complete drift)
        """
        # Re-initialize with default axioms only
        temp_system = CorePhilosophySystem(load_from_manifest=False)
        
        # Compare current and default fingerprints
        original_fingerprint = int(temp_system.schema_fingerprint)
        current_fingerprint = int(self.schema_fingerprint)
        
        # Simple differencee-based drift measure
        if original_fingerprint == current_fingerprint:
            return 0.0
        
        # Count differences in axioms
        original_axioms = set(temp_system.axioms.keys())
        current_axioms = set(self.axioms.keys())
        
        added = len(current_axioms - original_axioms)
        removed = len(original_axioms - current_axioms)
        total = max(len(original_axioms), len(current_axioms))
        
        # Calculate drift based on added/removed axioms
        structural_drift = (added + removed) / (2 * total) if total > 0 else 0.0
        
        # Check for modifications in common axioms
        modified = 0
        common_axioms = original_axioms.intersection(current_axioms)
        
        for axiom_id in common_axioms:
            original = temp_system.axioms[axiom_id]
            current = self.axioms[axiom_id]
            
            if (original.statement != current.statement or
                original.priority != current.priority or
                original.activation_threshold != current.activation_threshold):
                modified += 1
        
        content_drift = modified / len(common_axioms) if common_axioms else 0.0
        
        # Combine structural and content drift
        return min(1.0, (structural_drift * 0.7) + (content_drift * 0.3))
    
    def get_axiom_network(self) -> Dict[str, Any]:
        """
        Get a network representation of axioms and their relationships.
        
        Returns:
            Dictionary with nodes and edges for network visualization
        """
        nodes = []
        edges = []
        
        # Create nodes for each axiom
        for axiom_id, axiom in self.axioms.items():
            nodes.append({
                "id": axiom_id,
                "label": axiom_id.replace("_", " ").title(),
                "category": axiom.category.name,
                "priority": axiom.priority,
                "active": axiom_id in self.active_axioms
            })
            
            # Create edges for related axioms
            for related_id in axiom.related_axioms:
                if related_id in self.axioms:
                    edges.append({
                        "source": axiom_id,
                        "target": related_id,
                        "weight": 1.0  # Default weight
                    })
        
        return {
            "nodes": nodes,
            "edges": edges
        }


# Default singleton instance
core_philosophy = CorePhilosophySystem() 