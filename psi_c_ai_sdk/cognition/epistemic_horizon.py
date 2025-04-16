"""
Personal Epistemic Horizon Model

This module tracks the agent's confidence in what it knows, doubts, and can't yet resolve,
and exports this as a boundary visual or summary narrative.

Metrics include:
- Confidence per belief: Confidence(M_i) = 1 - H_i
- Quarantine zone: Beliefs flagged due to repeated contradiction or uncertainty
- Boundary Score: ℰ_t = (|stable| - |unstable|) / |total beliefs|

Example narrative output:
"I hold 248 stable beliefs, 31 unresolved, and 4 beliefs in epistemic quarantine."
"""

import math
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class BeliefNode:
    """Represents a single belief node in the epistemic horizon model."""
    
    id: str
    content: str
    confidence: float = 1.0  # 1.0 = complete confidence, 0.0 = no confidence
    entropy: float = 0.0  # Higher values indicate more uncertainty
    contradictions: int = 0  # Count of times this belief has been contradicted
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_updated: float = field(default_factory=lambda: datetime.now().timestamp())
    category: str = "uncategorized"
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    related_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_confidence(self, new_confidence: float) -> None:
        """Update the confidence score."""
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.entropy = 1.0 - self.confidence
        self.last_updated = datetime.now().timestamp()
    
    def add_contradiction(self) -> None:
        """Record a contradiction against this belief."""
        self.contradictions += 1
        # Confidence typically decreases with contradictions
        self.update_confidence(self.confidence * 0.9)  # Reduce by 10%
    
    def add_related(self, belief_id: str) -> None:
        """Add a related belief ID."""
        self.related_ids.add(belief_id)
    
    def quarantine(self, reason: str) -> None:
        """Place this belief in quarantine."""
        self.quarantined = True
        self.quarantine_reason = reason
    
    def unquarantine(self) -> None:
        """Remove this belief from quarantine."""
        self.quarantined = False
        self.quarantine_reason = None
    
    def stability_score(self) -> float:
        """
        Calculate stability score based on confidence and contradiction history.
        
        Returns:
            Stability score from -1.0 (highly unstable) to 1.0 (highly stable)
        """
        contradiction_factor = math.exp(-0.5 * self.contradictions)
        return (2 * self.confidence - 1) * contradiction_factor
    
    def is_stable(self, threshold: float = 0.6) -> bool:
        """Check if belief is stable."""
        return self.stability_score() >= threshold
    
    def is_unstable(self, threshold: float = -0.2) -> bool:
        """Check if belief is unstable."""
        return self.stability_score() <= threshold
    
    def belief_state(self, stable_threshold: float = 0.6, unstable_threshold: float = -0.2) -> str:
        """Get the belief state category."""
        if self.quarantined:
            return "quarantined"
        
        stability = self.stability_score()
        
        if stability >= stable_threshold:
            return "stable"
        elif stability <= unstable_threshold:
            return "unstable"
        else:
            return "unresolved"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "contradictions": self.contradictions,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "category": self.category,
            "tags": self.tags,
            "source": self.source,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "related_ids": list(self.related_ids),
            "metadata": self.metadata,
            "stability_score": self.stability_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeliefNode':
        """Create from dictionary representation."""
        node = cls(
            id=data["id"],
            content=data["content"],
            confidence=data.get("confidence", 1.0),
            entropy=data.get("entropy", 0.0),
            contradictions=data.get("contradictions", 0),
            created_at=data.get("created_at", datetime.now().timestamp()),
            last_updated=data.get("last_updated", datetime.now().timestamp()),
            category=data.get("category", "uncategorized"),
            tags=data.get("tags", []),
            source=data.get("source"),
            quarantined=data.get("quarantined", False),
            quarantine_reason=data.get("quarantine_reason"),
            related_ids=set(data.get("related_ids", [])),
            metadata=data.get("metadata", {})
        )
        return node


class EpistemicHorizon:
    """Tracks and manages the agent's epistemic state."""
    
    def __init__(self, 
                 stable_threshold: float = 0.6,
                 unstable_threshold: float = -0.2,
                 quarantine_threshold: int = 3):
        """
        Initialize the epistemic horizon model.
        
        Args:
            stable_threshold: Threshold for considering a belief stable
            unstable_threshold: Threshold for considering a belief unstable
            quarantine_threshold: Number of contradictions before quarantine
        """
        self.beliefs: Dict[str, BeliefNode] = {}
        self.stable_threshold = stable_threshold
        self.unstable_threshold = unstable_threshold
        self.quarantine_threshold = quarantine_threshold
        self.history: List[Dict[str, Any]] = []
    
    def add_belief(self, id: str, content: str, confidence: float = 1.0,
                 category: str = "uncategorized", tags: Optional[List[str]] = None,
                 source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new belief to the model.
        
        Args:
            id: Unique identifier for the belief
            content: The belief content
            confidence: Initial confidence score
            category: Category of the belief
            tags: List of tags associated with the belief
            source: Source of the belief
            metadata: Additional metadata
            
        Returns:
            The belief ID
        """
        if id in self.beliefs:
            # If belief already exists, update it
            belief = self.beliefs[id]
            belief.content = content
            belief.update_confidence(confidence)
            belief.category = category
            belief.tags = tags or belief.tags
            belief.source = source or belief.source
            belief.metadata.update(metadata or {})
        else:
            # Create a new belief
            belief = BeliefNode(
                id=id,
                content=content,
                confidence=confidence,
                entropy=1.0 - confidence,
                category=category,
                tags=tags or [],
                source=source,
                metadata=metadata or {}
            )
            self.beliefs[id] = belief
        
        # Record the event
        self._record_event("add_belief", belief)
        
        return id
    
    def update_belief(self, id: str, content: Optional[str] = None,
                    confidence: Optional[float] = None,
                    category: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing belief.
        
        Args:
            id: Belief ID
            content: New content (if changing)
            confidence: New confidence score (if changing)
            category: New category (if changing)
            tags: New tags (if changing)
            metadata: Additional metadata to merge (if provided)
            
        Returns:
            True if the belief was updated, False if not found
        """
        if id not in self.beliefs:
            return False
        
        belief = self.beliefs[id]
        
        if content is not None:
            belief.content = content
        
        if confidence is not None:
            belief.update_confidence(confidence)
        
        if category is not None:
            belief.category = category
        
        if tags is not None:
            belief.tags = tags
        
        if metadata is not None:
            belief.metadata.update(metadata)
        
        belief.last_updated = datetime.now().timestamp()
        
        # Record the event
        self._record_event("update_belief", belief)
        
        return True
    
    def record_contradiction(self, belief_id: str, contradicting_id: Optional[str] = None,
                           evidence: Optional[str] = None) -> bool:
        """
        Record a contradiction against a belief.
        
        Args:
            belief_id: The ID of the belief being contradicted
            contradicting_id: The ID of the contradicting belief (if any)
            evidence: Description of the contradicting evidence
            
        Returns:
            True if recorded, False if belief not found
        """
        if belief_id not in self.beliefs:
            return False
        
        belief = self.beliefs[belief_id]
        belief.add_contradiction()
        
        # If this exceeds the quarantine threshold, quarantine the belief
        if belief.contradictions >= self.quarantine_threshold and not belief.quarantined:
            reason = f"Exceeded contradiction threshold ({self.quarantine_threshold})"
            belief.quarantine(reason)
        
        # Record relationship if contradicting belief exists
        if contradicting_id and contradicting_id in self.beliefs:
            contradicting_belief = self.beliefs[contradicting_id]
            belief.add_related(contradicting_id)
            contradicting_belief.add_related(belief_id)
        
        # Record the event
        event_data = {
            "belief_id": belief_id,
            "contradicting_id": contradicting_id,
            "evidence": evidence
        }
        self._record_event("contradiction", belief, event_data)
        
        return True
    
    def quarantine_belief(self, belief_id: str, reason: str) -> bool:
        """
        Explicitly quarantine a belief.
        
        Args:
            belief_id: The belief ID
            reason: Reason for quarantine
            
        Returns:
            True if quarantined, False if belief not found
        """
        if belief_id not in self.beliefs:
            return False
        
        belief = self.beliefs[belief_id]
        belief.quarantine(reason)
        
        # Record the event
        self._record_event("quarantine", belief, {"reason": reason})
        
        return True
    
    def unquarantine_belief(self, belief_id: str) -> bool:
        """
        Remove a belief from quarantine.
        
        Args:
            belief_id: The belief ID
            
        Returns:
            True if unquarantined, False if belief not found
        """
        if belief_id not in self.beliefs:
            return False
        
        belief = self.beliefs[belief_id]
        belief.unquarantine()
        
        # Record the event
        self._record_event("unquarantine", belief)
        
        return True
    
    def relate_beliefs(self, belief_id1: str, belief_id2: str) -> bool:
        """
        Establish a relationship between two beliefs.
        
        Args:
            belief_id1: First belief ID
            belief_id2: Second belief ID
            
        Returns:
            True if related, False if either belief not found
        """
        if belief_id1 not in self.beliefs or belief_id2 not in self.beliefs:
            return False
        
        belief1 = self.beliefs[belief_id1]
        belief2 = self.beliefs[belief_id2]
        
        belief1.add_related(belief_id2)
        belief2.add_related(belief_id1)
        
        # Record the event
        self._record_event("relate", belief1, {"related_to": belief_id2})
        
        return True
    
    def get_belief(self, belief_id: str) -> Optional[BeliefNode]:
        """
        Get a belief by ID.
        
        Args:
            belief_id: The belief ID
            
        Returns:
            The belief node or None if not found
        """
        return self.beliefs.get(belief_id)
    
    def get_related_beliefs(self, belief_id: str) -> List[BeliefNode]:
        """
        Get all beliefs related to the specified belief.
        
        Args:
            belief_id: The belief ID
            
        Returns:
            List of related belief nodes
        """
        if belief_id not in self.beliefs:
            return []
        
        related_ids = self.beliefs[belief_id].related_ids
        return [self.beliefs[rid] for rid in related_ids if rid in self.beliefs]
    
    def get_beliefs_by_category(self, category: str) -> List[BeliefNode]:
        """
        Get all beliefs in a category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of belief nodes in the category
        """
        return [b for b in self.beliefs.values() if b.category == category]
    
    def get_beliefs_by_tag(self, tag: str) -> List[BeliefNode]:
        """
        Get all beliefs with a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of belief nodes with the tag
        """
        return [b for b in self.beliefs.values() if tag in b.tags]
    
    def get_stable_beliefs(self) -> List[BeliefNode]:
        """
        Get all stable beliefs.
        
        Returns:
            List of stable belief nodes
        """
        return [b for b in self.beliefs.values() 
                if not b.quarantined and b.is_stable(self.stable_threshold)]
    
    def get_unstable_beliefs(self) -> List[BeliefNode]:
        """
        Get all unstable beliefs.
        
        Returns:
            List of unstable belief nodes
        """
        return [b for b in self.beliefs.values() 
                if not b.quarantined and b.is_unstable(self.unstable_threshold)]
    
    def get_unresolved_beliefs(self) -> List[BeliefNode]:
        """
        Get all unresolved beliefs (neither stable nor unstable).
        
        Returns:
            List of unresolved belief nodes
        """
        return [b for b in self.beliefs.values() 
                if not b.quarantined and 
                not b.is_stable(self.stable_threshold) and 
                not b.is_unstable(self.unstable_threshold)]
    
    def get_quarantined_beliefs(self) -> List[BeliefNode]:
        """
        Get all quarantined beliefs.
        
        Returns:
            List of quarantined belief nodes
        """
        return [b for b in self.beliefs.values() if b.quarantined]
    
    def boundary_score(self) -> float:
        """
        Calculate the epistemic boundary score.
        
        Returns:
            Boundary score from -1.0 to 1.0
        """
        total = len(self.beliefs)
        
        if total == 0:
            return 0.0
        
        stable = len(self.get_stable_beliefs())
        unstable = len(self.get_unstable_beliefs())
        
        return (stable - unstable) / total
    
    def generate_summary(self) -> str:
        """
        Generate a narrative summary of the epistemic state.
        
        Returns:
            Summary string
        """
        stable = len(self.get_stable_beliefs())
        unstable = len(self.get_unstable_beliefs())
        unresolved = len(self.get_unresolved_beliefs())
        quarantined = len(self.get_quarantined_beliefs())
        total = len(self.beliefs)
        
        score = self.boundary_score()
        confidence_qualifier = "high" if score > 0.5 else "moderate" if score > 0 else "low"
        
        summary = f"I hold {stable} stable beliefs, {unresolved} unresolved, "
        summary += f"{unstable} unstable, and {quarantined} beliefs in epistemic quarantine.\n"
        
        if quarantined > 0:
            summary += "Quarantined beliefs include: "
            quarantine_examples = [b.content[:50] + "..." for b in self.get_quarantined_beliefs()[:3]]
            summary += ", ".join(quarantine_examples)
            if quarantined > 3:
                summary += f", and {quarantined - 3} more"
            summary += ".\n"
        
        summary += f"My overall epistemic confidence is {confidence_qualifier} "
        summary += f"(boundary score: {score:.2f})."
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the epistemic state.
        
        Returns:
            Dictionary of statistics
        """
        beliefs = list(self.beliefs.values())
        
        if not beliefs:
            return {
                "total_beliefs": 0,
                "boundary_score": 0.0
            }
        
        # Count beliefs by state
        stable = len(self.get_stable_beliefs())
        unstable = len(self.get_unstable_beliefs())
        unresolved = len(self.get_unresolved_beliefs())
        quarantined = len(self.get_quarantined_beliefs())
        total = len(beliefs)
        
        # Calculate average metrics
        avg_confidence = sum(b.confidence for b in beliefs) / total
        avg_entropy = sum(b.entropy for b in beliefs) / total
        avg_contradictions = sum(b.contradictions for b in beliefs) / total
        avg_stability = sum(b.stability_score() for b in beliefs) / total
        
        # Get the most contradicted beliefs
        most_contradicted = sorted(beliefs, key=lambda b: b.contradictions, reverse=True)[:5]
        
        # Get the most recent updates
        recent_updates = sorted(beliefs, key=lambda b: b.last_updated, reverse=True)[:5]
        
        # Count by category
        categories = {}
        for belief in beliefs:
            categories[belief.category] = categories.get(belief.category, 0) + 1
        
        return {
            "total_beliefs": total,
            "stable_beliefs": stable,
            "unstable_beliefs": unstable,
            "unresolved_beliefs": unresolved,
            "quarantined_beliefs": quarantined,
            "boundary_score": self.boundary_score(),
            "avg_confidence": avg_confidence,
            "avg_entropy": avg_entropy,
            "avg_contradictions": avg_contradictions,
            "avg_stability": avg_stability,
            "most_contradicted": [b.id for b in most_contradicted],
            "recent_updates": [b.id for b in recent_updates],
            "categories": categories
        }
    
    def visualize_horizon(self, output_file: Optional[str] = None, 
                         highlight_ids: Optional[List[str]] = None) -> None:
        """
        Visualize the epistemic horizon as a scatter plot.
        
        Args:
            output_file: Path to save the visualization (if None, displays interactively)
            highlight_ids: List of belief IDs to highlight
        """
        if not self.beliefs:
            return
            
        # Prepare data
        beliefs = list(self.beliefs.values())
        x = [b.entropy for b in beliefs]
        y = [b.contradictions for b in beliefs]
        
        # Set up colors based on state
        colors = []
        for belief in beliefs:
            if belief.quarantined:
                colors.append('red')
            elif belief.is_stable(self.stable_threshold):
                colors.append('green')
            elif belief.is_unstable(self.unstable_threshold):
                colors.append('orange')
            else:
                colors.append('blue')
        
        # Prepare the plot
        plt.figure(figsize=(12, 8))
        
        # Create the scatter plot
        scatter = plt.scatter(x, y, c=colors, alpha=0.7, s=100)
        
        # Highlight specific beliefs if requested
        if highlight_ids:
            highlight_indices = [i for i, b in enumerate(beliefs) if b.id in highlight_ids]
            if highlight_indices:
                highlight_x = [x[i] for i in highlight_indices]
                highlight_y = [y[i] for i in highlight_indices]
                plt.scatter(highlight_x, highlight_y, c='purple', s=150, edgecolors='black', linewidth=2)
        
        # Add labels for a few interesting beliefs
        for i, belief in enumerate(beliefs):
            if belief.quarantined or (highlight_ids and belief.id in highlight_ids):
                plt.annotate(
                    belief.id,
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Stable'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Unresolved'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Unstable'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Quarantined')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        # Add labels and title
        plt.xlabel('Entropy (Uncertainty)')
        plt.ylabel('Contradictions')
        plt.title('Epistemic Horizon')
        
        # Add the boundary score
        score = self.boundary_score()
        plt.figtext(0.02, 0.02, f"Boundary Score: {score:.2f}", fontsize=12)
        
        # Save or show
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
    
    def visualize_categories(self, output_file: Optional[str] = None) -> None:
        """
        Visualize belief categories as a bar chart.
        
        Args:
            output_file: Path to save the visualization (if None, displays interactively)
        """
        if not self.beliefs:
            return
            
        # Count beliefs by category
        categories = {}
        for belief in self.beliefs.values():
            categories[belief.category] = categories.get(belief.category, 0) + 1
        
        # Sort categories by count
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare data
        cat_names = [c[0] for c in sorted_categories]
        cat_counts = [c[1] for c in sorted_categories]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create the bar chart
        bars = plt.bar(cat_names, cat_counts)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Add labels and title
        plt.xlabel('Categories')
        plt.ylabel('Number of Beliefs')
        plt.title('Beliefs by Category')
        
        # Rotate category labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
    
    def to_json(self) -> str:
        """
        Convert the epistemic horizon to a JSON string.
        
        Returns:
            JSON string representation
        """
        data = {
            "beliefs": {id: belief.to_dict() for id, belief in self.beliefs.items()},
            "thresholds": {
                "stable": self.stable_threshold,
                "unstable": self.unstable_threshold,
                "quarantine": self.quarantine_threshold
            },
            "stats": self.get_statistics(),
            "history": self.history
        }
        
        return json.dumps(data, indent=2)
    
    def save(self, file_path: str) -> bool:
        """
        Save the epistemic horizon to a file.
        
        Args:
            file_path: Path to save the data
            
        Returns:
            True if saved successfully
        """
        try:
            with open(file_path, 'w') as f:
                f.write(self.to_json())
            return True
        except Exception as e:
            print(f"Error saving epistemic horizon: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> 'EpistemicHorizon':
        """
        Load an epistemic horizon from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded EpistemicHorizon instance
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            horizon = cls(
                stable_threshold=data.get("thresholds", {}).get("stable", 0.6),
                unstable_threshold=data.get("thresholds", {}).get("unstable", -0.2),
                quarantine_threshold=data.get("thresholds", {}).get("quarantine", 3)
            )
            
            # Load beliefs
            for belief_data in data.get("beliefs", {}).values():
                node = BeliefNode.from_dict(belief_data)
                horizon.beliefs[node.id] = node
            
            # Load history if available
            horizon.history = data.get("history", [])
            
            return horizon
        except Exception as e:
            print(f"Error loading epistemic horizon: {e}")
            return cls()
    
    def _record_event(self, event_type: str, belief: BeliefNode, 
                     extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Record an event in the history."""
        event = {
            "timestamp": datetime.now().timestamp(),
            "event_type": event_type,
            "belief_id": belief.id,
            "belief_state": belief.belief_state(self.stable_threshold, self.unstable_threshold)
        }
        
        if extra_data:
            event.update(extra_data)
            
        self.history.append(event)


def create_epistemic_horizon_from_memories(
    memories: List[Dict[str, Any]],
    contradictions: Optional[List[Dict[str, Any]]] = None
) -> EpistemicHorizon:
    """
    Create an epistemic horizon from a list of memories and contradictions.
    
    Args:
        memories: List of memory objects
        contradictions: List of contradiction events
        
    Returns:
        EpistemicHorizon instance
    """
    horizon = EpistemicHorizon()
    
    # Add beliefs from memories
    for memory in memories:
        id = memory.get("id", str(hash(memory.get("content", ""))))
        content = memory.get("content", "")
        confidence = memory.get("confidence", 1.0)
        category = memory.get("category", "uncategorized")
        tags = memory.get("tags", [])
        source = memory.get("source")
        
        horizon.add_belief(
            id=id,
            content=content,
            confidence=confidence,
            category=category,
            tags=tags,
            source=source,
            metadata=memory
        )
    
    # Process contradictions if provided
    if contradictions:
        for contradiction in contradictions:
            belief_id = contradiction.get("belief_id")
            contradicting_id = contradiction.get("contradicting_id")
            evidence = contradiction.get("evidence")
            
            if belief_id:
                horizon.record_contradiction(belief_id, contradicting_id, evidence)
    
    return horizon 