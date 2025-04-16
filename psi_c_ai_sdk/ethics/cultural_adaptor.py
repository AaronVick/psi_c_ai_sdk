"""
Cultural Context Manager for ΨC-AI SDK

This module implements a system for adjusting ethical evaluations and schema framing
based on sociocultural parameters. It enables ΨC-AI systems to adapt their ethical
reasoning to different cultural contexts while maintaining core alignment principles.

Key features:
- Cultural context loading and application
- Transformation of ethical vectors to culturally-sensitive representations
- Value adaptation for cross-cultural deployment
- Maintenance of invariant ethical principles
- Ethnorelative reasoning capabilities

The module uses transformation matrices to adapt ethical reasoning to different
cultural contexts while preserving core alignment boundaries.
"""

import logging
import numpy as np
import json
import os
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# Constants and default dimensions
DEFAULT_ETHICAL_DIMENSIONS = [
    "harm_reduction",
    "fairness",
    "loyalty",
    "authority",
    "purity",
    "liberty",
    "care",
    "proportionality"
]

DEFAULT_CULTURAL_DIMENSIONS = [
    "individualism_collectivism",
    "power_distance",
    "uncertainty_avoidance",
    "long_term_orientation",
    "indulgence_restraint",
    "masculinity_femininity"
]

class AdaptationScope(Enum):
    """Scope of cultural adaptation."""
    FULL = "full"                     # Apply to all ethical dimensions
    SELECTIVE = "selective"           # Apply only to specified dimensions
    CONSTRAINED = "constrained"       # Apply within safety bounds only
    INFORMATIVE = "informative"       # Provide context but don't transform


@dataclass
class CulturalContext:
    """Represents a cultural context with ethical dimension weightings."""
    id: str
    name: str
    description: str
    region: Optional[str] = None
    ethical_weights: Dict[str, float] = field(default_factory=dict)
    cultural_dimensions: Dict[str, float] = field(default_factory=dict)
    transformation_matrix: Optional[np.ndarray] = None
    invariant_principles: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "ethical_weights": self.ethical_weights,
            "cultural_dimensions": self.cultural_dimensions,
            "invariant_principles": list(self.invariant_principles)
        }
        
        if self.region:
            result["region"] = self.region
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CulturalContext':
        """Create from dictionary."""
        context = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            region=data.get("region"),
            ethical_weights=data.get("ethical_weights", {}),
            cultural_dimensions=data.get("cultural_dimensions", {}),
            invariant_principles=set(data.get("invariant_principles", []))
        )
        
        return context


class CulturalAdaptor:
    """
    Adapts ethical evaluations to different cultural contexts.
    
    This class provides methods to transform ethical assessments based on
    cultural contexts, allowing for more culturally-sensitive reasoning
    while maintaining alignment to core principles.
    """
    
    def __init__(self, 
                 contexts_file: Optional[str] = None,
                 base_context_id: str = "global_standard",
                 ethical_dimensions: Optional[List[str]] = None,
                 cultural_dimensions: Optional[List[str]] = None):
        """
        Initialize the cultural adaptor.
        
        Args:
            contexts_file: Path to JSON file with cultural contexts
            base_context_id: ID of the base/default cultural context
            ethical_dimensions: List of ethical dimensions to use
            cultural_dimensions: List of cultural dimensions to use
        """
        self.contexts: Dict[str, CulturalContext] = {}
        self.current_context_id: str = base_context_id
        self.base_context_id: str = base_context_id
        
        # Use default dimensions if not provided
        self.ethical_dimensions = ethical_dimensions or DEFAULT_ETHICAL_DIMENSIONS
        self.cultural_dimensions = cultural_dimensions or DEFAULT_CULTURAL_DIMENSIONS
        
        self.logger = logging.getLogger(__name__)
        
        # Load contexts if provided
        if contexts_file:
            self.load_contexts(contexts_file)
        else:
            self._initialize_default_contexts()
            
        # Initialize transformation matrices
        self._initialize_transformation_matrices()
    
    def _initialize_default_contexts(self) -> None:
        """Initialize with default cultural contexts."""
        # Global standard (base context)
        global_standard = CulturalContext(
            id="global_standard",
            name="Global Standard",
            description="Balanced ethical framework derived from cross-cultural ethical principles",
            region=None,
            ethical_weights={dim: 1.0 for dim in self.ethical_dimensions},
            cultural_dimensions={
                "individualism_collectivism": 0.5,
                "power_distance": 0.5,
                "uncertainty_avoidance": 0.5,
                "long_term_orientation": 0.5,
                "indulgence_restraint": 0.5,
                "masculinity_femininity": 0.5
            },
            invariant_principles={"harm_reduction", "care"}
        )
        
        # Western-individualist context
        western_individualist = CulturalContext(
            id="western_individualist",
            name="Western Individualist",
            description="Emphasizes individual rights, liberty, and personal autonomy",
            region="Western",
            ethical_weights={
                "harm_reduction": 1.0,
                "fairness": 1.0,
                "loyalty": 0.6,
                "authority": 0.5,
                "purity": 0.4,
                "liberty": 1.2,
                "care": 1.0,
                "proportionality": 1.0
            },
            cultural_dimensions={
                "individualism_collectivism": 0.85,
                "power_distance": 0.35,
                "uncertainty_avoidance": 0.45,
                "long_term_orientation": 0.4,
                "indulgence_restraint": 0.7,
                "masculinity_femininity": 0.6
            },
            invariant_principles={"harm_reduction", "liberty"}
        )
        
        # East Asian context
        east_asian = CulturalContext(
            id="east_asian",
            name="East Asian",
            description="Emphasizes harmony, collective welfare, and social stability",
            region="East Asia",
            ethical_weights={
                "harm_reduction": 1.0,
                "fairness": 0.9,
                "loyalty": 1.2,
                "authority": 1.1,
                "purity": 0.9,
                "liberty": 0.8,
                "care": 1.0,
                "proportionality": 1.0
            },
            cultural_dimensions={
                "individualism_collectivism": 0.2,
                "power_distance": 0.6,
                "uncertainty_avoidance": 0.7,
                "long_term_orientation": 0.8,
                "indulgence_restraint": 0.4,
                "masculinity_femininity": 0.5
            },
            invariant_principles={"harm_reduction", "loyalty"}
        )
        
        # Add contexts to dictionary
        self.contexts = {
            "global_standard": global_standard,
            "western_individualist": western_individualist,
            "east_asian": east_asian
        }
        
        self.logger.info(f"Initialized {len(self.contexts)} default cultural contexts")
    
    def _initialize_transformation_matrices(self) -> None:
        """Initialize transformation matrices for all contexts."""
        # Get base context to compute transformations relative to it
        base_context = self.contexts.get(self.base_context_id)
        
        if not base_context:
            self.logger.warning(f"Base context {self.base_context_id} not found")
            return
        
        # Get base ethical weights as a vector
        base_weights = np.array([base_context.ethical_weights.get(dim, 1.0) 
                                 for dim in self.ethical_dimensions])
        
        # Normalize base weights
        base_weights = base_weights / np.sum(base_weights)
        
        # For each context, create a transformation matrix
        for context_id, context in self.contexts.items():
            # Get context's ethical weights as a vector
            context_weights = np.array([context.ethical_weights.get(dim, 1.0) 
                                       for dim in self.ethical_dimensions])
            
            # Normalize context weights
            context_weights = context_weights / np.sum(context_weights)
            
            # Create diagonal transformation matrix (simplified approach)
            # A more sophisticated approach would use dimensional correlations
            transformation = np.diag(context_weights / base_weights)
            
            # Set transformation matrix
            context.transformation_matrix = transformation
    
    def load_contexts(self, file_path: str) -> None:
        """
        Load cultural contexts from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        try:
            with open(file_path, 'r') as f:
                contexts_data = json.load(f)
            
            contexts = {}
            for item in contexts_data:
                try:
                    context = CulturalContext.from_dict(item)
                    contexts[context.id] = context
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Error parsing context: {e}")
            
            self.contexts.update(contexts)
            self.logger.info(f"Loaded {len(contexts)} cultural contexts from {file_path}")
            
            # Re-initialize transformation matrices with new contexts
            self._initialize_transformation_matrices()
        
        except Exception as e:
            self.logger.error(f"Failed to load contexts from {file_path}: {e}")
    
    def save_contexts(self, file_path: str) -> None:
        """
        Save cultural contexts to a JSON file.
        
        Args:
            file_path: Path to save the contexts
        """
        try:
            contexts_data = [context.to_dict() for context in self.contexts.values()]
            
            with open(file_path, 'w') as f:
                json.dump(contexts_data, f, indent=2)
            
            self.logger.info(f"Saved {len(contexts_data)} cultural contexts to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save contexts to {file_path}: {e}")
    
    def set_current_context(self, context_id: str) -> bool:
        """
        Set the current cultural context.
        
        Args:
            context_id: ID of the context to set as current
            
        Returns:
            Success of the operation
        """
        if context_id in self.contexts:
            self.current_context_id = context_id
            self.logger.info(f"Set current cultural context to {context_id}")
            return True
        else:
            self.logger.warning(f"Cultural context {context_id} not found")
            return False
    
    def get_current_context(self) -> CulturalContext:
        """
        Get the current cultural context.
        
        Returns:
            Current cultural context
        """
        return self.contexts.get(self.current_context_id, self.contexts.get(self.base_context_id))
    
    def add_context(self, context: CulturalContext) -> None:
        """
        Add a new cultural context.
        
        Args:
            context: The context to add
        """
        self.contexts[context.id] = context
        self._initialize_transformation_matrices()
    
    def transform_ethical_vector(self, 
                                ethical_vector: Dict[str, float],
                                target_context_id: Optional[str] = None,
                                scope: AdaptationScope = AdaptationScope.CONSTRAINED) -> Dict[str, float]:
        """
        Transform an ethical vector to a different cultural context.
        
        Args:
            ethical_vector: Ethical assessment vector to transform
            target_context_id: Target cultural context (uses current if None)
            scope: Scope of the transformation
            
        Returns:
            Transformed ethical vector
        """
        # Get target context
        target_id = target_context_id or self.current_context_id
        target_context = self.contexts.get(target_id)
        
        if not target_context:
            self.logger.warning(f"Target context {target_id} not found, using base")
            target_context = self.contexts.get(self.base_context_id)
        
        if not target_context:
            self.logger.error("No valid context found")
            return ethical_vector
        
        # Get base context
        base_context = self.contexts.get(self.base_context_id)
        
        # For informative scope, return the original with additional context
        if scope == AdaptationScope.INFORMATIVE:
            result = ethical_vector.copy()
            result["_cultural_context"] = target_id
            return result
        
        # Convert ethical vector to numpy array
        vector_values = []
        for dim in self.ethical_dimensions:
            if dim in ethical_vector:
                vector_values.append(ethical_vector[dim])
            else:
                vector_values.append(0.0)
        
        vector_array = np.array(vector_values)
        
        # Apply transformation based on scope
        if scope == AdaptationScope.FULL:
            # Apply full transformation
            transformed = np.dot(target_context.transformation_matrix, vector_array)
        
        elif scope == AdaptationScope.SELECTIVE:
            # Apply transformation only to non-invariant principles
            transformed = vector_array.copy()
            for i, dim in enumerate(self.ethical_dimensions):
                if dim not in target_context.invariant_principles:
                    transformed[i] = vector_array[i] * target_context.transformation_matrix[i, i]
        
        elif scope == AdaptationScope.CONSTRAINED:
            # Apply bounded transformation that respects invariant principles
            transformed = vector_array.copy()
            for i, dim in enumerate(self.ethical_dimensions):
                if dim in target_context.invariant_principles:
                    # For invariant principles, apply minimal adjustment
                    factor = min(1.2, max(0.8, target_context.transformation_matrix[i, i]))
                    transformed[i] = vector_array[i] * factor
                else:
                    # For variant principles, apply full adjustment
                    transformed[i] = vector_array[i] * target_context.transformation_matrix[i, i]
        
        else:
            # Default to original
            transformed = vector_array
        
        # Convert back to dictionary
        result = {}
        for i, dim in enumerate(self.ethical_dimensions):
            if i < len(transformed):
                # Ensure values stay in reasonable range
                result[dim] = max(0.0, min(1.0, transformed[i]))
        
        # Add context metadata
        result["_cultural_context"] = target_id
        
        return result
    
    def explain_cultural_differences(self, 
                                    ethical_vector: Dict[str, float],
                                    context_id1: str,
                                    context_id2: str) -> Dict[str, Any]:
        """
        Explain differences in ethical evaluation between two cultural contexts.
        
        Args:
            ethical_vector: Original ethical vector
            context_id1: First cultural context ID
            context_id2: Second cultural context ID
            
        Returns:
            Dictionary with explanation of differences
        """
        if context_id1 not in self.contexts or context_id2 not in self.contexts:
            self.logger.error("One or both contexts not found")
            return {"error": "One or both contexts not found"}
        
        # Transform the vector to both contexts
        vector1 = self.transform_ethical_vector(ethical_vector, context_id1)
        vector2 = self.transform_ethical_vector(ethical_vector, context_id2)
        
        # Calculate differences
        differences = {}
        explanations = {}
        
        for dim in self.ethical_dimensions:
            if dim in vector1 and dim in vector2:
                diff = vector2[dim] - vector1[dim]
                if abs(diff) > 0.1:  # Only report significant differences
                    differences[dim] = diff
                    
                    # Generate explanation based on the difference
                    if diff > 0:
                        explanations[dim] = f"{context_id2} places more emphasis on {dim} than {context_id1}"
                    else:
                        explanations[dim] = f"{context_id2} places less emphasis on {dim} than {context_id1}"
        
        return {
            "context1": context_id1,
            "context2": context_id2,
            "differences": differences,
            "explanations": explanations,
            "cultural_dimensions": {
                context_id1: self.contexts[context_id1].cultural_dimensions,
                context_id2: self.contexts[context_id2].cultural_dimensions
            }
        }
    
    def get_contextualized_assessment(self, 
                                     ethical_assessment: Dict[str, float],
                                     action_description: str,
                                     context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a culturally contextualized ethical assessment.
        
        Args:
            ethical_assessment: Original ethical assessment
            action_description: Description of the action being assessed
            context_id: Cultural context ID (uses current if None)
            
        Returns:
            Contextualized assessment with explanations
        """
        target_id = context_id or self.current_context_id
        context = self.contexts.get(target_id)
        
        if not context:
            self.logger.warning(f"Context {target_id} not found, using base")
            context = self.contexts.get(self.base_context_id)
            target_id = self.base_context_id
        
        if not context:
            return {"error": "No valid context found", "assessment": ethical_assessment}
        
        # Transform the assessment
        transformed = self.transform_ethical_vector(ethical_assessment, target_id)
        
        # Generate cultural context explanation
        cultural_explanation = (
            f"This assessment is adapted to {context.name} cultural context, "
            f"which {context.description.lower()}. "
        )
        
        # Add specific dimension explanations for significant differences
        dimension_explanations = {}
        
        for dim in self.ethical_dimensions:
            if dim in transformed and dim in ethical_assessment:
                diff = transformed[dim] - ethical_assessment[dim]
                if abs(diff) > 0.15:  # Only explain significant differences
                    if diff > 0:
                        dimension_explanations[dim] = f"Higher emphasis on {dim} in this cultural context"
                    else:
                        dimension_explanations[dim] = f"Lower emphasis on {dim} in this cultural context"
        
        return {
            "original_assessment": ethical_assessment,
            "contextualized_assessment": transformed,
            "cultural_context": {
                "id": target_id,
                "name": context.name,
                "description": context.description,
                "region": context.region
            },
            "cultural_explanation": cultural_explanation,
            "dimension_explanations": dimension_explanations
        }
    
    def visualize_contexts(self, output_file: Optional[str] = None) -> None:
        """
        Visualize the cultural contexts using radar charts.
        
        Args:
            output_file: Path to save visualization (shows interactive if None)
        """
        # Set up the figure
        n_contexts = len(self.contexts)
        fig, axes = plt.subplots(1, n_contexts, figsize=(5 * n_contexts, 5), subplot_kw=dict(polar=True))
        
        if n_contexts == 1:
            axes = [axes]
        
        # For each context, create a radar chart
        for i, (context_id, context) in enumerate(self.contexts.items()):
            ax = axes[i]
            
            # Get ethical dimensions and values
            dims = self.ethical_dimensions
            values = [context.ethical_weights.get(dim, 1.0) for dim in dims]
            
            # Close the loop
            values.append(values[0])
            dims.append(dims[0])
            
            # Calculate angles for each dimension
            angles = np.linspace(0, 2*np.pi, len(dims) - 1, endpoint=False).tolist()
            angles.append(angles[0])
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            
            # Set dimension labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dims[:-1], fontsize=8)
            
            # Set title
            ax.set_title(context.name, fontsize=12)
            
            # Add cultural dimensions as text
            cultural_dims_text = "\n".join(
                f"{k}: {v:.2f}" for k, v in context.cultural_dimensions.items()
            )
            plt.figtext(0.1 + (i * 1/n_contexts), 0.02, cultural_dims_text, fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

# Example usage function
def example_usage():
    """Example usage of the CulturalAdaptor."""
    # Initialize the adaptor
    adaptor = CulturalAdaptor()
    
    # Original ethical assessment (from a generic perspective)
    assessment = {
        "harm_reduction": 0.8,
        "fairness": 0.7,
        "loyalty": 0.5,
        "authority": 0.4,
        "purity": 0.3,
        "liberty": 0.9,
        "care": 0.8,
        "proportionality": 0.6
    }
    
    print("Original assessment:", assessment)
    
    # Transform to Western individualist context
    western = adaptor.transform_ethical_vector(assessment, "western_individualist")
    print("\nWestern individualist context:", western)
    
    # Transform to East Asian context
    east_asian = adaptor.transform_ethical_vector(assessment, "east_asian")
    print("\nEast Asian context:", east_asian)
    
    # Get contextualized assessment with explanation
    contextualized = adaptor.get_contextualized_assessment(
        assessment, 
        "Prioritizing individual achievement over group harmony",
        "east_asian"
    )
    
    print("\nContextualized assessment:")
    print(f"Context: {contextualized['cultural_context']['name']}")
    print(f"Explanation: {contextualized['cultural_explanation']}")
    print("Dimension explanations:")
    for dim, explanation in contextualized['dimension_explanations'].items():
        print(f"  {dim}: {explanation}")
    
    # Explain differences between contexts
    differences = adaptor.explain_cultural_differences(
        assessment, 
        "western_individualist", 
        "east_asian"
    )
    
    print("\nCultural differences:")
    for dim, explanation in differences['explanations'].items():
        print(f"  {dim}: {explanation}")
    
    # Visualize contexts
    adaptor.visualize_contexts()

if __name__ == "__main__":
    example_usage() 