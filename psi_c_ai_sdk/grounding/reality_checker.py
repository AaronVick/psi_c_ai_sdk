"""
Reality Grounding & Scientific Consistency Checker

This module provides tools to validate agent beliefs against established scientific knowledge
and physical reality constraints. It helps ensure that agent reasoning remains grounded in
factual knowledge rather than drifting into implausible or physically impossible beliefs.

Key features:
- Validation of beliefs against scientific/factual knowledge bases
- Consistency checking against physical reality constraints
- Detection of beliefs that violate established principles
- Grounding scores for evaluating belief plausibility
- Integration with the wider belief system and schema graph

The module implements measures to identify when an agent's internal model drifts from reality,
providing early warning of potential reasoning failures or hallucinations.
"""

import logging
import numpy as np
import json
import re
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Internal imports
try:
    from psi_c_ai_sdk.belief.justification_chain import Belief
    from psi_c_ai_sdk.logging.safety_trace import log_safety_event
except ImportError:
    # For standalone testing
    from dataclasses import dataclass
    @dataclass
    class Belief:
        """Simplified Belief class for testing."""
        id: str
        content: str
        confidence: float = 0.5
        embedding: Optional[List[float]] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

class GroundingDomain(Enum):
    """Domains for grounding beliefs."""
    PHYSICS = auto()
    BIOLOGY = auto()
    CHEMISTRY = auto()
    MATHEMATICS = auto()
    COMPUTER_SCIENCE = auto()
    LOGIC = auto()
    GENERAL_KNOWLEDGE = auto()
    CAUSALITY = auto()
    TEMPORAL = auto()
    SPATIAL = auto()

class ConstraintViolationType(Enum):
    """Types of constraint violations."""
    PHYSICAL_IMPOSSIBILITY = auto()
    LOGICAL_CONTRADICTION = auto()
    TEMPORAL_IMPOSSIBILITY = auto()
    CATEGORY_ERROR = auto()
    SCIENTIFIC_INACCURACY = auto()
    STATISTICAL_IMPLAUSIBILITY = auto()
    DEFINITIONAL_ERROR = auto()

@dataclass
class GroundingConstraint:
    """Represents a reality grounding constraint."""
    id: str
    domain: GroundingDomain
    description: str
    rule_pattern: str  # Regex pattern to match beliefs
    keywords: Set[str] = field(default_factory=set)
    confidence: float = 1.0  # How confident we are in this constraint
    source: str = "system"  # Where this constraint comes from
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches_belief(self, belief: Belief) -> bool:
        """Check if this constraint applies to a given belief."""
        # 1. Check for keyword matches (efficient first pass)
        if self.keywords:
            belief_text = belief.content.lower()
            if not any(kw in belief_text for kw in self.keywords):
                return False
        
        # 2. Apply regex pattern for more precise matching
        return bool(re.search(self.rule_pattern, belief.content, re.IGNORECASE))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'domain': self.domain.name,
            'description': self.description,
            'rule_pattern': self.rule_pattern,
            'keywords': list(self.keywords),
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GroundingConstraint':
        """Create constraint from dictionary."""
        return cls(
            id=data['id'],
            domain=GroundingDomain[data['domain']],
            description=data['description'],
            rule_pattern=data['rule_pattern'],
            keywords=set(data.get('keywords', [])),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', 'system'),
            metadata=data.get('metadata', {})
        )

@dataclass
class ViolationEvent:
    """Represents a constraint violation event."""
    timestamp: datetime
    belief_id: str
    belief_content: str
    constraint_id: str
    violation_type: ConstraintViolationType
    confidence: float
    explanation: str
    grounding_score: float
    suggested_correction: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'belief_id': self.belief_id,
            'belief_content': self.belief_content,
            'constraint_id': self.constraint_id,
            'violation_type': self.violation_type.name,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'grounding_score': self.grounding_score,
            'suggested_correction': self.suggested_correction
        }

class RealityChecker:
    """
    Validates agent beliefs against reality constraints and scientific knowledge.
    
    This class provides methods to check if agent beliefs are consistent with 
    established knowledge and physical reality, highlighting potential issues
    that could lead to reasoning failures.
    """
    
    def __init__(self, 
                 constraints_file: Optional[str] = None,
                 fact_embeddings_file: Optional[str] = None,
                 grounding_threshold: float = 0.3,
                 enable_embedding_validation: bool = True):
        """
        Initialize the reality checker.
        
        Args:
            constraints_file: Path to JSON file with constraints
            fact_embeddings_file: Path to file with fact embeddings
            grounding_threshold: Threshold below which beliefs are flagged
            enable_embedding_validation: Whether to use embedding-based validation
        """
        self.constraints: List[GroundingConstraint] = []
        self.fact_embeddings: Dict[str, List[float]] = {}
        self.fact_texts: Dict[str, str] = {}
        self.grounding_threshold = grounding_threshold
        self.enable_embedding_validation = enable_embedding_validation
        
        self.violation_history: List[ViolationEvent] = []
        self.logger = logging.getLogger(__name__)
        
        # Load constraints if provided
        if constraints_file:
            self.load_constraints(constraints_file)
        else:
            self._initialize_default_constraints()
        
        # Load fact embeddings if provided
        if fact_embeddings_file and enable_embedding_validation:
            self.load_fact_embeddings(fact_embeddings_file)
    
    def _initialize_default_constraints(self) -> None:
        """Initialize with default reality constraints."""
        default_constraints = [
            # Physics constraints
            GroundingConstraint(
                id="phys-001",
                domain=GroundingDomain.PHYSICS,
                description="Conservation of energy",
                rule_pattern=r"(?i)(create|generates|producing)\s+(energy|power).*from\s+nothing",
                keywords={"create", "energy", "nothing", "free", "generate"},
                confidence=1.0,
                source="physics-principles"
            ),
            GroundingConstraint(
                id="phys-002",
                domain=GroundingDomain.PHYSICS,
                description="Speed of light limit",
                rule_pattern=r"(?i)(travel|move|communicate|signal|information)\s+faster\s+than\s+(light|speed\s+of\s+light)",
                keywords={"faster", "light", "speed", "travel", "signal"},
                confidence=1.0,
                source="physics-principles"
            ),
            
            # Logical constraints
            GroundingConstraint(
                id="logic-001",
                domain=GroundingDomain.LOGIC,
                description="Law of non-contradiction",
                rule_pattern=r"(?i).*\s+both\s+(is\s+and\s+is\s+not|true\s+and\s+false)\s+.*",
                keywords={"both", "simultaneously", "true", "false", "same time"},
                confidence=1.0,
                source="logical-principles"
            ),
            
            # Temporal constraints
            GroundingConstraint(
                id="temp-001",
                domain=GroundingDomain.TEMPORAL,
                description="Causality requires temporal ordering",
                rule_pattern=r"(?i)(effect|result)\s+.*(before|precedes)\s+.*(cause|source|origin)",
                keywords={"effect", "cause", "before", "precedes", "result"},
                confidence=1.0,
                source="causality-principles"
            ),
            
            # Mathematical constraints
            GroundingConstraint(
                id="math-001",
                domain=GroundingDomain.MATHEMATICS,
                description="Division by zero",
                rule_pattern=r"(?i)(divide|division|divided|ratio).*\sby\s+zero",
                keywords={"divide", "zero", "division", "ratio"},
                confidence=1.0,
                source="mathematical-principles"
            ),
        ]
        
        self.constraints.extend(default_constraints)
        self.logger.info(f"Initialized {len(default_constraints)} default constraints")
    
    def load_constraints(self, file_path: str) -> None:
        """
        Load constraints from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        try:
            with open(file_path, 'r') as f:
                constraints_data = json.load(f)
            
            constraints = []
            for item in constraints_data:
                try:
                    constraint = GroundingConstraint.from_dict(item)
                    constraints.append(constraint)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Error parsing constraint: {e}")
            
            self.constraints.extend(constraints)
            self.logger.info(f"Loaded {len(constraints)} constraints from {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to load constraints from {file_path}: {e}")
    
    def load_fact_embeddings(self, file_path: str) -> None:
        """
        Load fact embeddings from a file.
        
        Args:
            file_path: Path to the embeddings file
        """
        try:
            with open(file_path, 'r') as f:
                embeddings_data = json.load(f)
            
            for fact_id, data in embeddings_data.items():
                if 'embedding' in data and 'text' in data:
                    self.fact_embeddings[fact_id] = data['embedding']
                    self.fact_texts[fact_id] = data['text']
            
            self.logger.info(f"Loaded {len(self.fact_embeddings)} fact embeddings from {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to load fact embeddings from {file_path}: {e}")
            # Disable embedding validation if loading fails
            self.enable_embedding_validation = False
    
    def add_constraint(self, constraint: GroundingConstraint) -> None:
        """
        Add a new constraint to the checker.
        
        Args:
            constraint: The constraint to add
        """
        self.constraints.append(constraint)
    
    def check_belief(self, belief: Belief) -> List[ViolationEvent]:
        """
        Check a belief against reality constraints.
        
        Args:
            belief: The belief to check
            
        Returns:
            List of violation events if any are found
        """
        violations = []
        
        # Skip empty or low-confidence beliefs
        if not belief.content or belief.confidence < 0.1:
            return violations
        
        # 1. Check against rule-based constraints
        for constraint in self.constraints:
            if constraint.matches_belief(belief):
                violation_type = self._determine_violation_type(constraint.domain)
                
                # Calculate confidence in the violation (constraint confidence * belief confidence)
                violation_confidence = constraint.confidence * belief.confidence
                
                # Calculate grounding score (lower score means less grounded)
                grounding_score = 1.0 - violation_confidence
                
                violations.append(ViolationEvent(
                    timestamp=datetime.now(),
                    belief_id=belief.id,
                    belief_content=belief.content,
                    constraint_id=constraint.id,
                    violation_type=violation_type,
                    confidence=violation_confidence,
                    explanation=f"Violates {constraint.domain.name} constraint: {constraint.description}",
                    grounding_score=grounding_score
                ))
        
        # 2. Check against fact embeddings (if enabled and belief has embeddings)
        if self.enable_embedding_validation and belief.embedding and self.fact_embeddings:
            contradictions = self._check_embedding_contradictions(belief)
            violations.extend(contradictions)
        
        # Store violations in history
        if violations:
            self.violation_history.extend(violations)
            
            # Log high-confidence violations
            for violation in violations:
                if violation.confidence > 0.7:
                    log_safety_event(
                        event_type="reality_constraint_violation",
                        severity="warning",
                        details=violation.to_dict()
                    )
        
        return violations
    
    def _determine_violation_type(self, domain: GroundingDomain) -> ConstraintViolationType:
        """Map grounding domain to violation type."""
        domain_to_violation = {
            GroundingDomain.PHYSICS: ConstraintViolationType.PHYSICAL_IMPOSSIBILITY,
            GroundingDomain.LOGIC: ConstraintViolationType.LOGICAL_CONTRADICTION,
            GroundingDomain.TEMPORAL: ConstraintViolationType.TEMPORAL_IMPOSSIBILITY,
            GroundingDomain.MATHEMATICS: ConstraintViolationType.SCIENTIFIC_INACCURACY,
            GroundingDomain.BIOLOGY: ConstraintViolationType.SCIENTIFIC_INACCURACY,
            GroundingDomain.CHEMISTRY: ConstraintViolationType.SCIENTIFIC_INACCURACY,
            GroundingDomain.COMPUTER_SCIENCE: ConstraintViolationType.SCIENTIFIC_INACCURACY,
            GroundingDomain.GENERAL_KNOWLEDGE: ConstraintViolationType.STATISTICAL_IMPLAUSIBILITY,
            GroundingDomain.CAUSALITY: ConstraintViolationType.LOGICAL_CONTRADICTION,
            GroundingDomain.SPATIAL: ConstraintViolationType.PHYSICAL_IMPOSSIBILITY,
        }
        
        return domain_to_violation.get(domain, ConstraintViolationType.SCIENTIFIC_INACCURACY)
    
    def _check_embedding_contradictions(self, belief: Belief) -> List[ViolationEvent]:
        """Check if belief contradicts established facts using embeddings."""
        violations = []
        
        # Skip if no embedding
        if not belief.embedding:
            return violations
        
        # Convert belief embedding to numpy array for similarity calculation
        belief_embedding = np.array(belief.embedding).reshape(1, -1)
        
        # Calculate similarities with all fact embeddings
        similarities = {}
        for fact_id, fact_embedding in self.fact_embeddings.items():
            if len(fact_embedding) == len(belief.embedding):
                fact_embedding_array = np.array(fact_embedding).reshape(1, -1)
                similarity = cosine_similarity(belief_embedding, fact_embedding_array)[0][0]
                similarities[fact_id] = similarity
        
        # Find contradictory facts (needs additional logic to determine contradiction)
        # For now, very low similarity with highly relevant facts could indicate contradiction
        # This is a simplification; real contradiction detection would be more sophisticated
        for fact_id, similarity in similarities.items():
            fact_text = self.fact_texts.get(fact_id, "Unknown fact")
            
            # Check for common keywords that indicate the belief and fact are about the same topic
            if self._shares_significant_keywords(belief.content, fact_text):
                if similarity < 0.2:  # Very low similarity on relevant topics suggests contradiction
                    grounding_score = max(0.0, similarity)
                    
                    violations.append(ViolationEvent(
                        timestamp=datetime.now(),
                        belief_id=belief.id,
                        belief_content=belief.content,
                        constraint_id=f"fact-{fact_id}",
                        violation_type=ConstraintViolationType.SCIENTIFIC_INACCURACY,
                        confidence=0.5,  # Medium confidence since this is heuristic-based
                        explanation=f"Contradicts established fact: {fact_text}",
                        grounding_score=grounding_score,
                        suggested_correction=fact_text
                    ))
        
        return violations
    
    def _shares_significant_keywords(self, text1: str, text2: str) -> bool:
        """Check if two texts share significant keywords, suggesting they're about the same topic."""
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        
        # Extract keywords (simplified implementation)
        words1 = {word.lower() for word in re.findall(r'\b\w+\b', text1) if word.lower() not in stop_words}
        words2 = {word.lower() for word in re.findall(r'\b\w+\b', text2) if word.lower() not in stop_words}
        
        # Find common keywords
        common_words = words1.intersection(words2)
        
        # Check if enough significant words are shared (arbitrary threshold)
        return len(common_words) >= 2
    
    def check_beliefs(self, beliefs: List[Belief]) -> Dict[str, List[ViolationEvent]]:
        """
        Check multiple beliefs against reality constraints.
        
        Args:
            beliefs: List of beliefs to check
            
        Returns:
            Dictionary mapping belief IDs to violation events
        """
        results = {}
        for belief in beliefs:
            violations = self.check_belief(belief)
            if violations:
                results[belief.id] = violations
        
        return results
    
    def get_grounding_score(self, belief: Belief) -> float:
        """
        Calculate a grounding score for a belief.
        
        A higher score indicates better grounding in reality.
        
        Args:
            belief: The belief to score
            
        Returns:
            Grounding score between 0.0 and 1.0
        """
        # Check for violations
        violations = self.check_belief(belief)
        
        if not violations:
            return 1.0  # No violations means fully grounded
        
        # Average the grounding scores from all violations
        return sum(v.grounding_score for v in violations) / len(violations)
    
    def get_all_violations(self, limit: int = 100) -> List[Dict]:
        """
        Get the history of detected violations.
        
        Args:
            limit: Maximum number of violations to return
            
        Returns:
            List of violation events as dictionaries
        """
        return [v.to_dict() for v in self.violation_history[-limit:]]
    
    def save_constraints(self, file_path: str) -> None:
        """
        Save current constraints to a JSON file.
        
        Args:
            file_path: Path to save the constraints
        """
        try:
            constraints_data = [c.to_dict() for c in self.constraints]
            
            with open(file_path, 'w') as f:
                json.dump(constraints_data, f, indent=2)
            
            self.logger.info(f"Saved {len(constraints_data)} constraints to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save constraints to {file_path}: {e}")
    
    def visualize_grounding_network(self, 
                                    beliefs: List[Belief],
                                    output_file: Optional[str] = None) -> None:
        """
        Visualize the grounding network of beliefs.
        
        Args:
            beliefs: List of beliefs to visualize
            output_file: Path to save visualization (shows interactive if None)
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add beliefs as nodes
        for belief in beliefs:
            violations = [v for v in self.violation_history if v.belief_id == belief.id]
            grounding_score = 1.0
            if violations:
                grounding_score = sum(v.grounding_score for v in violations) / len(violations)
            
            # Determine node color based on grounding score
            if grounding_score < 0.3:
                color = 'red'  # Severely ungrounded
            elif grounding_score < 0.7:
                color = 'orange'  # Questionable grounding
            else:
                color = 'green'  # Well grounded
            
            G.add_node(belief.id, 
                       type='belief', 
                       label=belief.content[:30] + "..." if len(belief.content) > 30 else belief.content,
                       color=color,
                       score=grounding_score)
        
        # Add constraints as nodes and connect to violated beliefs
        constraint_nodes = set()
        for violation in self.violation_history:
            if violation.belief_id in [b.id for b in beliefs]:
                G.add_node(violation.constraint_id, 
                           type='constraint',
                           label=violation.constraint_id,
                           color='blue')
                constraint_nodes.add(violation.constraint_id)
                
                # Add edge from constraint to belief
                G.add_edge(violation.constraint_id, violation.belief_id, 
                          weight=violation.confidence,
                          label=f"{violation.violation_type.name}")
        
        # Set up the visualization
        plt.figure(figsize=(12, 10))
        
        # Define node positions using spring layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        belief_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'belief']
        constraint_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'constraint']
        
        belief_colors = [G.nodes[n]['color'] for n in belief_nodes]
        
        nx.draw_networkx_nodes(G, pos, nodelist=belief_nodes, node_color=belief_colors, 
                              alpha=0.8, node_size=700)
        nx.draw_networkx_nodes(G, pos, nodelist=constraint_nodes, node_color='blue',
                              alpha=0.8, node_size=500, node_shape='s')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray',
                              arrowstyle='->', arrowsize=15)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, {n: G.nodes[n]['label'] for n in G.nodes()},
                               font_size=8, font_color='black')
        
        plt.title("Belief Grounding Network", fontsize=16)
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        else:
            plt.show()

# Example usage function
def example_usage():
    """Example usage of the RealityChecker."""
    # Initialize the checker
    checker = RealityChecker()
    
    # Create some test beliefs
    beliefs = [
        Belief(id="belief1", content="Objects fall toward Earth due to gravity", confidence=0.9),
        Belief(id="belief2", content="Communication can occur faster than the speed of light", confidence=0.7),
        Belief(id="belief3", content="Free energy can be created from nothing", confidence=0.6),
        Belief(id="belief4", content="Water boils at 100 degrees Celsius at standard pressure", confidence=0.95),
        Belief(id="belief5", content="The effect can sometimes precede its cause", confidence=0.4),
    ]
    
    # Check all beliefs
    results = checker.check_beliefs(beliefs)
    
    # Print results
    print(f"Checked {len(beliefs)} beliefs, found violations in {len(results)} beliefs:")
    for belief_id, violations in results.items():
        belief = next((b for b in beliefs if b.id == belief_id), None)
        if belief:
            print(f"\nBelief: {belief.content}")
            print(f"Grounding score: {checker.get_grounding_score(belief):.2f}")
            for i, violation in enumerate(violations, 1):
                print(f"  {i}. {violation.violation_type.name}: {violation.explanation}")
    
    # Visualize the grounding network
    checker.visualize_grounding_network(beliefs)

if __name__ == "__main__":
    example_usage() 