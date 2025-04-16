"""
Narrative Signature Generator Module

This module provides functionality to compress an agent's reflection and mutation history
into a human-readable "signature" of its evolution. This allows for easy understanding
of an agent's cognitive journey and development over time.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

class NarrativeSignature:
    """Compresses an agent's reflection and mutation history into a human-readable signature."""
    
    def __init__(self, reflection_history: List[Dict[str, Any]] = None, 
                 schema_mutations: List[Dict[str, Any]] = None):
        """
        Initialize the NarrativeSignature generator.
        
        Args:
            reflection_history: List of reflection events with their details
            schema_mutations: List of schema mutation events with their details
        """
        self.reflection_history = reflection_history or []
        self.schema_mutations = schema_mutations or []
        self.origin_vector = []
        self.key_contradictions = []
        self.mutation_count = 0
        self.bias_shifts = []
        self.critical_thresholds = []
        self.cognitive_journey = []
    
    def compute_origin_vector(self, top_k: int = 3) -> List[str]:
        """
        Compute the origin vector that identifies the primary motivating factors.
        
        Args:
            top_k: Number of top motivating factors to include
            
        Returns:
            List of top motivating factors
        """
        if not self.reflection_history:
            return []
            
        # Extract concepts from early reflections
        early_reflections = self.reflection_history[:min(5, len(self.reflection_history))]
        concepts = []
        
        for reflection in early_reflections:
            if 'concepts' in reflection:
                concepts.extend(reflection['concepts'])
            elif 'themes' in reflection:
                concepts.extend(reflection['themes'])
            elif 'content' in reflection:
                # Extract key terms from content using simple frequency
                words = reflection['content'].lower().split()
                # Filter common words
                filtered = [w for w in words if len(w) > 4 and w not in 
                           ['about', 'would', 'could', 'should', 'there', 'their',
                            'these', 'those', 'while', 'where', 'which']]
                concepts.extend(filtered)
                
        # Count occurrences of concepts
        concept_counts = Counter(concepts)
        
        # Get top-k concepts
        self.origin_vector = [item[0] for item in concept_counts.most_common(top_k)]
        return self.origin_vector
    
    def identify_key_contradictions(self, threshold: float = 0.7) -> List[str]:
        """
        Identify key contradictions that led to significant cognitive development.
        
        Args:
            threshold: Contradiction significance threshold
            
        Returns:
            List of key contradiction identifiers
        """
        if not self.reflection_history:
            return []
            
        key_events = []
        
        for i, reflection in enumerate(self.reflection_history):
            if 'contradiction_level' in reflection and reflection['contradiction_level'] > threshold:
                key_events.append(f"reflection_{i+1}")
            elif 'contradictions' in reflection and reflection['contradictions']:
                # If we have a list of contradictions
                key_events.append(f"reflection_{i+1}")
            elif 'entropy_delta' in reflection and reflection['entropy_delta'] > 0.3:
                # Significant entropy shifts often indicate contradictions
                key_events.append(f"reflection_{i+1}")
                
        self.key_contradictions = key_events[:5]  # Limit to top 5
        return self.key_contradictions
    
    def count_schema_mutations(self) -> int:
        """
        Count the number of significant schema mutations.
        
        Returns:
            Number of schema mutations
        """
        if self.schema_mutations:
            self.mutation_count = len(self.schema_mutations)
        else:
            # Estimate from reflection history
            self.mutation_count = sum(1 for r in self.reflection_history 
                                    if r.get('schema_change', False) or 
                                    r.get('schema_mutation', False))
            
        return self.mutation_count
    
    def analyze_bias_shifts(self) -> Tuple[str, float]:
        """
        Analyze changes in cognitive biases and focus over time.
        
        Returns:
            Tuple of (description of dominant shift, magnitude)
        """
        if not self.reflection_history or len(self.reflection_history) < 3:
            return ("insufficient data", 0.0)
            
        # Simple analysis based on attribute presence or text cues
        early_reflections = self.reflection_history[:len(self.reflection_history)//3]
        late_reflections = self.reflection_history[-len(self.reflection_history)//3:]
        
        # Analyze shifts in attributes if available
        early_attrs = self._extract_agent_attributes(early_reflections)
        late_attrs = self._extract_agent_attributes(late_reflections)
        
        # Find the most significant shift
        shifts = []
        
        # Compare common attributes
        for attr in set(early_attrs.keys()) & set(late_attrs.keys()):
            early_val = early_attrs[attr]
            late_val = late_attrs[attr]
            
            # If numeric, calculate difference
            if isinstance(early_val, (int, float)) and isinstance(late_val, (int, float)):
                diff = late_val - early_val
                shifts.append((f"from {attr} {early_val:.1f} → {late_val:.1f}", abs(diff)))
            # If string, check if they're different
            elif isinstance(early_val, str) and isinstance(late_val, str) and early_val != late_val:
                shifts.append((f"from {attr} {early_val} → {late_val}", 1.0))
        
        if shifts:
            shifts.sort(key=lambda x: x[1], reverse=True)
            return shifts[0]
        
        # Fallback to text analysis if no structured attributes
        return self._analyze_text_shifts(early_reflections, late_reflections)
    
    def _extract_agent_attributes(self, reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract agent attributes from reflection data."""
        attributes = {}
        
        for r in reflections:
            if 'attributes' in r:
                attributes.update(r['attributes'])
            if 'metrics' in r:
                attributes.update(r['metrics'])
            
        return attributes
    
    def _analyze_text_shifts(self, early_reflections: List[Dict[str, Any]], 
                            late_reflections: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Analyze shifts based on text content."""
        # Define pairs of opposing attributes to check for
        opposing_pairs = [
            ('speed', 'depth'),
            ('exploration', 'exploitation'),
            ('creativity', 'consistency'),
            ('caution', 'risk'),
            ('breadth', 'focus')
        ]
        
        early_text = " ".join([r.get('content', '') for r in early_reflections if 'content' in r])
        late_text = " ".join([r.get('content', '') for r in late_reflections if 'content' in r])
        
        for first, second in opposing_pairs:
            early_first = early_text.lower().count(first)
            early_second = early_text.lower().count(second)
            late_first = late_text.lower().count(first)
            late_second = late_text.lower().count(second)
            
            early_ratio = early_first / (early_second + 1)  # +1 to avoid division by zero
            late_ratio = late_first / (late_second + 1)
            
            if early_ratio > 2 and late_ratio < 0.5:
                return f"from {first} → {second}", abs(early_ratio - late_ratio)
            elif early_ratio < 0.5 and late_ratio > 2:
                return f"from {second} → {first}", abs(early_ratio - late_ratio)
        
        return ("no significant bias shift detected", 0.0)
    
    def compute_goal_alignment_vector(self, 
                                     initial_goals: List[str], 
                                     current_state: Dict[str, Any]) -> float:
        """
        Computes the cosine drift between initial goals and current state.
        
        Args:
            initial_goals: List of initial goal statements
            current_state: Current agent state with goal indicators
            
        Returns:
            Cosine similarity between initial and current goals
        """
        # Simplified implementation - in a real system, this would use embeddings
        # to compare semantic similarity between initial and current goals
        
        # Placeholder for demonstration purposes
        goal_drift = np.random.uniform(0.7, 1.0)  # Simulate some value between 0.7-1.0
        return goal_drift
    
    def generate_signature(self) -> Dict[str, Any]:
        """
        Generate the complete narrative signature.
        
        Returns:
            Dictionary containing the narrative signature components
        """
        # Compute all components if not already done
        if not self.origin_vector:
            self.compute_origin_vector()
            
        if not self.key_contradictions:
            self.identify_key_contradictions()
            
        if self.mutation_count == 0:
            self.count_schema_mutations()
            
        bias_shift, shift_magnitude = self.analyze_bias_shifts()
        
        # Assemble the signature
        signature = {
            "origin_vector": self.origin_vector,
            "key_contradictions": self.key_contradictions,
            "schema_mutations": self.mutation_count,
            "dominant_bias_shift": bias_shift,
            "signature_version": "1.0"
        }
        
        return signature
    
    def to_json(self) -> str:
        """
        Convert the signature to a JSON string.
        
        Returns:
            JSON string representation of the signature
        """
        return json.dumps(self.generate_signature(), indent=2)
    
    def from_reflection_file(self, filename: str) -> 'NarrativeSignature':
        """
        Load reflection history from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            Self for method chaining
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                self.reflection_history = data
            elif isinstance(data, dict) and 'reflections' in data:
                self.reflection_history = data['reflections']
            elif isinstance(data, dict) and 'history' in data:
                self.reflection_history = data['history']
                
            return self
        except Exception as e:
            print(f"Error loading reflection file: {e}")
            return self
    
    def from_schema_mutation_file(self, filename: str) -> 'NarrativeSignature':
        """
        Load schema mutation history from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            Self for method chaining
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                self.schema_mutations = data
            elif isinstance(data, dict) and 'mutations' in data:
                self.schema_mutations = data['mutations']
            elif isinstance(data, dict) and 'history' in data:
                self.schema_mutations = data['history']
                
            return self
        except Exception as e:
            print(f"Error loading schema mutation file: {e}")
            return self

def generate_narrative_signature(reflection_file: Optional[str] = None, 
                               mutation_file: Optional[str] = None,
                               reflection_data: Optional[List[Dict[str, Any]]] = None,
                               mutation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Convenience function to generate a narrative signature from files or data.
    
    Args:
        reflection_file: Path to reflection history JSON file
        mutation_file: Path to schema mutation history JSON file
        reflection_data: Reflection history data
        mutation_data: Schema mutation data
        
    Returns:
        Narrative signature dictionary
    """
    signature = NarrativeSignature(reflection_data, mutation_data)
    
    if reflection_file:
        signature.from_reflection_file(reflection_file)
        
    if mutation_file:
        signature.from_schema_mutation_file(mutation_file)
        
    return signature.generate_signature() 