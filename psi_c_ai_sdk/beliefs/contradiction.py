"""
Contradiction Detection Module for ΨC-AI SDK

This module provides utilities for detecting contradictions between memories
or beliefs, supporting the belief revision system with different contradiction
detection strategies and confidence scoring.
"""

import logging
from enum import Enum, auto
from typing import List, Tuple, Dict, Any, Optional, Set, Callable, NamedTuple

from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.embeddings.similarity import calculate_semantic_similarity
from psi_c_ai_sdk.nlp.text_analysis import extract_key_propositions, extract_entities

logger = logging.getLogger(__name__)


class ContradictionType(Enum):
    """Types of contradictions that can be detected."""
    DIRECT = auto()          # Direct statement vs negation
    PROPERTY = auto()        # Conflicting property values
    TEMPORAL = auto()        # Temporal impossibility
    LOGICAL = auto()         # Logical inconsistency
    RELATIONAL = auto()      # Conflicting relationships
    NUMERICAL = auto()       # Numerical inconsistency
    CATEGORICAL = auto()     # Conflicting categories


class ContradictionResult(NamedTuple):
    """Result of a contradiction detection between two memories."""
    memory1: Memory
    memory2: Memory
    confidence: float
    contradiction_type: ContradictionType
    explanation: str


class ContradictionDetector:
    """
    Detector for finding contradictions between memories or beliefs.
    
    This class provides methods for identifying different types of contradictions
    using various strategies, including semantic analysis, entity extraction,
    and proposition comparison.
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.65,
        entity_match_threshold: float = 0.8,
        use_llm_verification: bool = True
    ):
        """
        Initialize the contradiction detector.
        
        Args:
            semantic_threshold: Threshold for semantic similarity (0.0-1.0)
            entity_match_threshold: Threshold for entity matching (0.0-1.0)
            use_llm_verification: Whether to use LLM for verification step
        """
        self.semantic_threshold = semantic_threshold
        self.entity_match_threshold = entity_match_threshold
        self.use_llm_verification = use_llm_verification
        
        # Detection strategies with weights
        self.strategies = {
            "direct_negation": {
                "weight": 0.35,
                "func": self._detect_direct_negation
            },
            "property_conflict": {
                "weight": 0.25,
                "func": self._detect_property_conflict
            },
            "temporal_conflict": {
                "weight": 0.2,
                "func": self._detect_temporal_conflict
            },
            "numerical_conflict": {
                "weight": 0.1,
                "func": self._detect_numerical_conflict
            },
            "categorical_conflict": {
                "weight": 0.1,
                "func": self._detect_categorical_conflict
            }
        }
        
        # Statistics
        self.stats = {
            "total_comparisons": 0,
            "contradictions_found": 0,
            "by_type": {ctype.name: 0 for ctype in ContradictionType}
        }
    
    def find_contradictions(
        self,
        memories: List[Memory],
        max_results: Optional[int] = None
    ) -> List[Tuple[Memory, Memory, float]]:
        """
        Find contradictions in a list of memories.
        
        Args:
            memories: List of memories to check for contradictions
            max_results: Maximum number of contradictions to return
            
        Returns:
            List of tuples with contradicting memories and confidence score
        """
        if len(memories) < 2:
            return []
        
        logger.info(f"Scanning for contradictions among {len(memories)} memories")
        
        results = []
        compared_pairs = set()
        
        # First, extract key propositions and entities for all memories
        memory_data = {}
        for memory in memories:
            # Skip memories without content
            if not hasattr(memory, 'text') or not memory.text:
                continue
                
            # Extract key data
            props = extract_key_propositions(memory.text)
            entities = extract_entities(memory.text)
            
            memory_data[memory.id] = {
                "memory": memory,
                "propositions": props,
                "entities": entities
            }
        
        # Compare pairs of memories for contradictions
        for i, mem1 in enumerate(memories[:-1]):
            for mem2 in memories[i+1:]:
                # Skip if we've already compared this pair
                pair_key = tuple(sorted([mem1.id, mem2.id]))
                if pair_key in compared_pairs:
                    continue
                
                compared_pairs.add(pair_key)
                self.stats["total_comparisons"] += 1
                
                # Skip empty or identical memories
                if (not hasattr(mem1, 'text') or not mem1.text or
                        not hasattr(mem2, 'text') or not mem2.text or
                        mem1.id == mem2.id):
                    continue
                
                # Check if memories have common entities (potential contradiction)
                mem1_entities = memory_data.get(mem1.id, {}).get("entities", [])
                mem2_entities = memory_data.get(mem2.id, {}).get("entities", [])
                
                if not self._has_common_entities(mem1_entities, mem2_entities):
                    continue
                
                # Apply all contradiction detection strategies
                contradiction_result = self._apply_strategies(mem1, mem2, memory_data)
                
                if contradiction_result:
                    self.stats["contradictions_found"] += 1
                    self.stats["by_type"][contradiction_result.contradiction_type.name] += 1
                    
                    results.append((
                        contradiction_result.memory1,
                        contradiction_result.memory2,
                        contradiction_result.confidence
                    ))
                    
                    logger.debug(
                        f"Found contradiction ({contradiction_result.contradiction_type.name}) "
                        f"between memories {mem1.id} and {mem2.id} with confidence {contradiction_result.confidence:.2f}"
                    )
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Apply limit if specified
        if max_results is not None and len(results) > max_results:
            results = results[:max_results]
        
        logger.info(f"Found {len(results)} contradictions among {self.stats['total_comparisons']} memory pairs")
        return results
    
    def _apply_strategies(
        self,
        memory1: Memory,
        memory2: Memory,
        memory_data: Dict[str, Dict[str, Any]]
    ) -> Optional[ContradictionResult]:
        """
        Apply all contradiction detection strategies to a pair of memories.
        
        Args:
            memory1: First memory to compare
            memory2: Second memory to compare
            memory_data: Precomputed data for all memories
            
        Returns:
            ContradictionResult if a contradiction is found, None otherwise
        """
        strategy_results = []
        
        # Apply each strategy
        for strategy_name, strategy_info in self.strategies.items():
            result = strategy_info["func"](memory1, memory2, memory_data)
            
            if result:
                # Weight the result by the strategy's importance
                weighted_result = (
                    result.confidence * strategy_info["weight"],
                    result.contradiction_type,
                    result.explanation
                )
                strategy_results.append(weighted_result)
        
        if not strategy_results:
            return None
        
        # Find the strongest contradiction
        strategy_results.sort(reverse=True)
        strongest = strategy_results[0]
        
        # If we have multiple strategies detecting contradictions,
        # increase our confidence proportionally
        confidence = strongest[0]
        if len(strategy_results) > 1:
            # Adjust confidence based on how many strategies found contradictions
            confidence = min(1.0, confidence * (1.0 + 0.1 * (len(strategy_results) - 1)))
        
        return ContradictionResult(
            memory1=memory1,
            memory2=memory2,
            confidence=confidence,
            contradiction_type=strongest[1],
            explanation=strongest[2]
        )
    
    def _has_common_entities(
        self,
        entities1: List[str],
        entities2: List[str]
    ) -> bool:
        """
        Check if two sets of entities have sufficient overlap.
        
        Args:
            entities1: Entities from first memory
            entities2: Entities from second memory
            
        Returns:
            True if the memories have common entities, False otherwise
        """
        if not entities1 or not entities2:
            return False
        
        # Check for exact matches first
        common_entities = set(entities1) & set(entities2)
        if common_entities:
            return True
        
        # If no exact matches, check for similar entities
        for e1 in entities1:
            for e2 in entities2:
                if calculate_semantic_similarity(e1, e2) > self.entity_match_threshold:
                    return True
        
        return False
    
    def _detect_direct_negation(
        self,
        memory1: Memory,
        memory2: Memory,
        memory_data: Dict[str, Dict[str, Any]]
    ) -> Optional[ContradictionResult]:
        """
        Detect direct negation contradictions (A vs not A).
        
        Args:
            memory1: First memory to compare
            memory2: Second memory to compare
            memory_data: Precomputed data for all memories
            
        Returns:
            ContradictionResult if a direct negation is found, None otherwise
        """
        props1 = memory_data.get(memory1.id, {}).get("propositions", [])
        props2 = memory_data.get(memory2.id, {}).get("propositions", [])
        
        if not props1 or not props2:
            return None
        
        # Check for direct negations
        for p1 in props1:
            for p2 in props2:
                # Check if one statement directly negates another
                negation_score = self._measure_negation(p1, p2)
                
                if negation_score > 0.7:  # Threshold for negation
                    return ContradictionResult(
                        memory1=memory1,
                        memory2=memory2,
                        confidence=negation_score,
                        contradiction_type=ContradictionType.DIRECT,
                        explanation=f"Direct negation: '{p1}' vs '{p2}'"
                    )
        
        return None
    
    def _detect_property_conflict(
        self,
        memory1: Memory,
        memory2: Memory,
        memory_data: Dict[str, Dict[str, Any]]
    ) -> Optional[ContradictionResult]:
        """
        Detect property conflicts (X has property A vs X has property B).
        
        Args:
            memory1: First memory to compare
            memory2: Second memory to compare
            memory_data: Precomputed data for all memories
            
        Returns:
            ContradictionResult if a property conflict is found, None otherwise
        """
        # Implementation for detecting conflicting property values
        # This would look for cases like "The car is red" vs "The car is blue"
        
        # Simplified implementation - in a real system, this would use
        # structured extraction of subject-property relations
        
        # Extract properties from propositions
        props1 = memory_data.get(memory1.id, {}).get("propositions", [])
        props2 = memory_data.get(memory2.id, {}).get("propositions", [])
        
        if not props1 or not props2:
            return None
        
        entities1 = memory_data.get(memory1.id, {}).get("entities", [])
        entities2 = memory_data.get(memory2.id, {}).get("entities", [])
        
        # Find common entities
        common_entities = []
        for e1 in entities1:
            for e2 in entities2:
                if e1 == e2 or calculate_semantic_similarity(e1, e2) > self.entity_match_threshold:
                    common_entities.append((e1, e2))
        
        if not common_entities:
            return None
        
        # Check for conflicting properties for common entities
        for e1, e2 in common_entities:
            # Extract properties for each entity
            props_for_e1 = self._extract_properties_for_entity(e1, props1)
            props_for_e2 = self._extract_properties_for_entity(e2, props2)
            
            for p1 in props_for_e1:
                for p2 in props_for_e2:
                    # Check if properties are mutually exclusive
                    if self._are_properties_conflicting(p1, p2):
                        return ContradictionResult(
                            memory1=memory1,
                            memory2=memory2,
                            confidence=0.8,  # Fixed confidence for now
                            contradiction_type=ContradictionType.PROPERTY,
                            explanation=f"Conflicting properties for '{e1}': '{p1}' vs '{p2}'"
                        )
        
        return None
    
    def _detect_temporal_conflict(
        self,
        memory1: Memory,
        memory2: Memory,
        memory_data: Dict[str, Dict[str, Any]]
    ) -> Optional[ContradictionResult]:
        """
        Detect temporal conflicts (impossible timeline).
        
        Args:
            memory1: First memory to compare
            memory2: Second memory to compare
            memory_data: Precomputed data for all memories
            
        Returns:
            ContradictionResult if a temporal conflict is found, None otherwise
        """
        # Check if both memories have timestamps
        if not (hasattr(memory1, 'timestamp') and hasattr(memory2, 'timestamp')):
            return None
        
        # Extract temporal claims from propositions
        props1 = memory_data.get(memory1.id, {}).get("propositions", [])
        props2 = memory_data.get(memory2.id, {}).get("propositions", [])
        
        entities1 = memory_data.get(memory1.id, {}).get("entities", [])
        entities2 = memory_data.get(memory2.id, {}).get("entities", [])
        
        # Find common entities
        common_entities = []
        for e1 in entities1:
            for e2 in entities2:
                if e1 == e2 or calculate_semantic_similarity(e1, e2) > self.entity_match_threshold:
                    common_entities.append((e1, e2))
        
        if not common_entities:
            return None
        
        # Check for contradicting temporal claims
        # This is a simplified implementation - a real system would
        # use more sophisticated temporal reasoning
        
        # Extract temporal claims
        temporal_claims1 = self._extract_temporal_claims(props1)
        temporal_claims2 = self._extract_temporal_claims(props2)
        
        for entity1, entity2 in common_entities:
            for claim1 in temporal_claims1:
                if entity1 in claim1['entity']:
                    for claim2 in temporal_claims2:
                        if entity2 in claim2['entity']:
                            # Check for temporal contradiction
                            if self._are_temporal_claims_contradictory(claim1, claim2):
                                return ContradictionResult(
                                    memory1=memory1,
                                    memory2=memory2,
                                    confidence=0.75,
                                    contradiction_type=ContradictionType.TEMPORAL,
                                    explanation=f"Temporal contradiction for '{entity1}': '{claim1['text']}' vs '{claim2['text']}'"
                                )
        
        return None
    
    def _detect_numerical_conflict(
        self,
        memory1: Memory,
        memory2: Memory,
        memory_data: Dict[str, Dict[str, Any]]
    ) -> Optional[ContradictionResult]:
        """
        Detect numerical contradictions (X = N vs X = M).
        
        Args:
            memory1: First memory to compare
            memory2: Second memory to compare
            memory_data: Precomputed data for all memories
            
        Returns:
            ContradictionResult if a numerical contradiction is found, None otherwise
        """
        # Extract numerical claims from propositions
        props1 = memory_data.get(memory1.id, {}).get("propositions", [])
        props2 = memory_data.get(memory2.id, {}).get("propositions", [])
        
        entities1 = memory_data.get(memory1.id, {}).get("entities", [])
        entities2 = memory_data.get(memory2.id, {}).get("entities", [])
        
        # Find common entities
        common_entities = []
        for e1 in entities1:
            for e2 in entities2:
                if e1 == e2 or calculate_semantic_similarity(e1, e2) > self.entity_match_threshold:
                    common_entities.append((e1, e2))
        
        if not common_entities:
            return None
        
        # Extract numerical claims
        numerical_claims1 = self._extract_numerical_claims(props1)
        numerical_claims2 = self._extract_numerical_claims(props2)
        
        for entity1, entity2 in common_entities:
            for claim1 in numerical_claims1:
                if entity1 in claim1['entity']:
                    for claim2 in numerical_claims2:
                        if entity2 in claim2['entity']:
                            # Check for numerical contradiction
                            if self._are_numerical_claims_contradictory(claim1, claim2):
                                return ContradictionResult(
                                    memory1=memory1,
                                    memory2=memory2,
                                    confidence=0.85,  # High confidence for numerical contradictions
                                    contradiction_type=ContradictionType.NUMERICAL,
                                    explanation=f"Numerical contradiction for '{entity1}': '{claim1['text']}' vs '{claim2['text']}'"
                                )
        
        return None
    
    def _detect_categorical_conflict(
        self,
        memory1: Memory,
        memory2: Memory,
        memory_data: Dict[str, Dict[str, Any]]
    ) -> Optional[ContradictionResult]:
        """
        Detect categorical contradictions (X belongs to category A vs B).
        
        Args:
            memory1: First memory to compare
            memory2: Second memory to compare
            memory_data: Precomputed data for all memories
            
        Returns:
            ContradictionResult if a categorical contradiction is found, None otherwise
        """
        # Extract categorical claims
        props1 = memory_data.get(memory1.id, {}).get("propositions", [])
        props2 = memory_data.get(memory2.id, {}).get("propositions", [])
        
        entities1 = memory_data.get(memory1.id, {}).get("entities", [])
        entities2 = memory_data.get(memory2.id, {}).get("entities", [])
        
        # Find common entities
        common_entities = []
        for e1 in entities1:
            for e2 in entities2:
                if e1 == e2 or calculate_semantic_similarity(e1, e2) > self.entity_match_threshold:
                    common_entities.append((e1, e2))
        
        if not common_entities:
            return None
        
        # Extract categorical claims
        categorical_claims1 = self._extract_categorical_claims(props1)
        categorical_claims2 = self._extract_categorical_claims(props2)
        
        for entity1, entity2 in common_entities:
            for claim1 in categorical_claims1:
                if entity1 in claim1['entity']:
                    for claim2 in categorical_claims2:
                        if entity2 in claim2['entity']:
                            # Check for categorical contradiction
                            if self._are_categories_mutually_exclusive(claim1['category'], claim2['category']):
                                return ContradictionResult(
                                    memory1=memory1,
                                    memory2=memory2,
                                    confidence=0.8,
                                    contradiction_type=ContradictionType.CATEGORICAL,
                                    explanation=f"Categorical contradiction for '{entity1}': '{claim1['category']}' vs '{claim2['category']}'"
                                )
        
        return None
    
    def _measure_negation(self, prop1: str, prop2: str) -> float:
        """
        Measure the degree to which one proposition negates another.
        
        Args:
            prop1: First proposition
            prop2: Second proposition
            
        Returns:
            Score indicating degree of negation (0.0-1.0)
        """
        # Simple implementation - check if one is the negation of the other
        # In a real system, this would be more sophisticated using NLU
        
        # Check similarity first - statements need to be talking about the same thing
        similarity = calculate_semantic_similarity(prop1, prop2)
        
        if similarity < self.semantic_threshold:
            return 0.0
        
        # Simplified negation detection - check for "not" and similar terms
        negation_terms = ["not", "never", "no", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't"]
        
        # Check if one statement contains a negation term and the other doesn't
        prop1_has_negation = any(term in prop1.lower().split() for term in negation_terms)
        prop2_has_negation = any(term in prop2.lower().split() for term in negation_terms)
        
        # Direct negation: one has negation, one doesn't, and they're otherwise similar
        if prop1_has_negation != prop2_has_negation and similarity > 0.7:
            return similarity * 0.9  # Slightly reduce confidence based on similarity
            
        return 0.0
    
    def _extract_properties_for_entity(
        self,
        entity: str,
        propositions: List[str]
    ) -> List[str]:
        """
        Extract properties for a given entity from propositions.
        
        Args:
            entity: Entity to extract properties for
            propositions: List of propositions to search
            
        Returns:
            List of property strings
        """
        # Simplified implementation - in a real system, this would use
        # more sophisticated NLP to extract subject-property relations
        properties = []
        for prop in propositions:
            if entity.lower() in prop.lower():
                # Extremely simplified - just get what follows the entity
                parts = prop.lower().split(entity.lower())
                if len(parts) > 1 and parts[1].strip():
                    properties.append(parts[1].strip())
        
        return properties
    
    def _are_properties_conflicting(self, prop1: str, prop2: str) -> bool:
        """
        Determine if two properties are conflicting.
        
        Args:
            prop1: First property
            prop2: Second property
            
        Returns:
            True if properties are conflicting, False otherwise
        """
        # This is a simplified implementation
        # In a real system, this would use knowledge about mutually exclusive properties
        
        # Check for known conflicting property pairs
        conflicting_pairs = [
            ("red", "blue"), ("tall", "short"), ("big", "small"),
            ("hot", "cold"), ("alive", "dead"), ("open", "closed"),
            ("on", "off"), ("new", "old"), ("clean", "dirty")
        ]
        
        # Convert to lowercase for comparison
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()
        
        # Check direct conflicts
        for pair in conflicting_pairs:
            if ((pair[0] in prop1_lower and pair[1] in prop2_lower) or
                    (pair[1] in prop1_lower and pair[0] in prop2_lower)):
                return True
        
        # More complex conflicts would require semantic understanding
        return False
    
    def _extract_temporal_claims(self, propositions: List[str]) -> List[Dict[str, Any]]:
        """
        Extract temporal claims from propositions.
        
        Args:
            propositions: List of propositions to analyze
            
        Returns:
            List of temporal claims
        """
        # Simplified implementation - in a real system, this would use
        # sophisticated NLP for temporal information extraction
        
        # Look for temporal indicators
        temporal_indicators = [
            "before", "after", "during", "while", "when",
            "yesterday", "today", "tomorrow", "now", "then",
            "earlier", "later", "ago", "since", "until",
            "january", "february", "march", "april", "may", "june", 
            "july", "august", "september", "october", "november", "december"
        ]
        
        results = []
        
        for prop in propositions:
            prop_lower = prop.lower()
            for indicator in temporal_indicators:
                if indicator in prop_lower.split():
                    # Extremely simplified - extract entity and temporal context
                    entity = self._extract_primary_entity(prop)
                    if entity:
                        results.append({
                            "entity": entity,
                            "temporal": indicator,
                            "text": prop,
                            "context": self._extract_temporal_context(prop, indicator)
                        })
                    break
        
        return results
    
    def _extract_primary_entity(self, text: str) -> str:
        """
        Extract the primary entity from a text.
        
        Args:
            text: Text to extract entity from
            
        Returns:
            Primary entity or empty string if none found
        """
        # This is a placeholder - in a real system, this would use NLP
        # to identify the grammatical subject or focus entity
        
        # Extremely simplified approach - just take the first noun
        words = text.split()
        if words:
            return words[0]  # Oversimplified - just return first word
        return ""
    
    def _extract_temporal_context(self, text: str, indicator: str) -> str:
        """
        Extract temporal context from text.
        
        Args:
            text: Text to extract from
            indicator: Temporal indicator word
            
        Returns:
            Temporal context string
        """
        # Simplistic extraction of temporal context
        parts = text.lower().split(indicator)
        if len(parts) > 1:
            return indicator + parts[1]
        return ""
    
    def _are_temporal_claims_contradictory(
        self,
        claim1: Dict[str, Any],
        claim2: Dict[str, Any]
    ) -> bool:
        """
        Determine if two temporal claims are contradictory.
        
        Args:
            claim1: First temporal claim
            claim2: Second temporal claim
            
        Returns:
            True if claims are contradictory, False otherwise
        """
        # Simplified implementation - in a real system, this would use
        # a temporal reasoning system to detect inconsistencies
        
        # Check for obvious contradictions like "today" vs "yesterday"
        incompatible_pairs = [
            ("today", "yesterday"), ("today", "tomorrow"),
            ("before", "after"), ("earlier", "later")
        ]
        
        for pair in incompatible_pairs:
            if ((claim1["temporal"] == pair[0] and claim2["temporal"] == pair[1]) or
                    (claim1["temporal"] == pair[1] and claim2["temporal"] == pair[0])):
                return True
        
        # More complex temporal reasoning would require a temporal logic system
        return False
    
    def _extract_numerical_claims(self, propositions: List[str]) -> List[Dict[str, Any]]:
        """
        Extract numerical claims from propositions.
        
        Args:
            propositions: List of propositions to analyze
            
        Returns:
            List of numerical claims
        """
        # Simplified implementation - in a real system, this would use
        # more sophisticated NLP to extract numerical information
        
        import re
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        
        results = []
        for prop in propositions:
            # Find all numbers in the proposition
            numbers = re.findall(number_pattern, prop)
            if numbers:
                entity = self._extract_primary_entity(prop)
                if entity:
                    results.append({
                        "entity": entity,
                        "value": float(numbers[0]),  # Just use the first number
                        "text": prop
                    })
        
        return results
    
    def _are_numerical_claims_contradictory(
        self,
        claim1: Dict[str, Any],
        claim2: Dict[str, Any]
    ) -> bool:
        """
        Determine if two numerical claims are contradictory.
        
        Args:
            claim1: First numerical claim
            claim2: Second numerical claim
            
        Returns:
            True if claims are contradictory, False otherwise
        """
        # Check if the values are different
        # Could include tolerance based on context
        return abs(claim1["value"] - claim2["value"]) > 0.001
    
    def _extract_categorical_claims(self, propositions: List[str]) -> List[Dict[str, Any]]:
        """
        Extract categorical claims from propositions.
        
        Args:
            propositions: List of propositions to analyze
            
        Returns:
            List of categorical claims
        """
        # Simplified implementation - in a real system, this would use
        # more sophisticated NLP to extract categorical information
        
        # Look for "is a" patterns
        categorical_patterns = ["is a", "is an", "are a", "are an", "belongs to"]
        
        results = []
        for prop in propositions:
            prop_lower = prop.lower()
            for pattern in categorical_patterns:
                if pattern in prop_lower:
                    parts = prop_lower.split(pattern)
                    if len(parts) > 1:
                        entity = parts[0].strip()
                        category = parts[1].strip()
                        if entity and category:
                            results.append({
                                "entity": entity,
                                "category": category,
                                "text": prop
                            })
                    break
        
        return results
    
    def _are_categories_mutually_exclusive(self, category1: str, category2: str) -> bool:
        """
        Determine if two categories are mutually exclusive.
        
        Args:
            category1: First category
            category2: Second category
            
        Returns:
            True if categories are mutually exclusive, False otherwise
        """
        # Simplified implementation - in a real system, this would use
        # a knowledge graph or ontology
        
        # Example mutually exclusive category pairs
        exclusive_pairs = [
            ("animal", "vegetable"), ("animal", "mineral"),
            ("vegetable", "mineral"), ("living", "inanimate"),
            ("fiction", "non-fiction"), ("true", "false")
        ]
        
        cat1 = category1.lower().strip()
        cat2 = category2.lower().strip()
        
        # Check direct conflicts
        for pair in exclusive_pairs:
            if ((pair[0] in cat1 and pair[1] in cat2) or
                    (pair[1] in cat1 and pair[0] in cat2)):
                return True
        
        # Check for semantic opposition
        similarity = calculate_semantic_similarity(cat1, cat2)
        if similarity < 0.2:  # Very dissimilar categories might be exclusive
            return True
            
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about contradiction detection.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy() 