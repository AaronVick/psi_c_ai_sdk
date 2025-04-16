"""
Reflection Guard

Implements safety monitoring for AI reasoning processes:
- Detects reasoning cycles and loops
- Identifies contradictions in statements
- Monitors variable manipulation
- Tracks process integrity

Provides early warning for potential reasoning failures.
"""

import logging
import re
import time
from typing import Dict, List, Any, Set, Optional, Tuple, Union
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)

class ReflectionGuard:
    """
    Guards against common reasoning failures in AI systems.
    
    Detects:
    - Cycles: When the same reasoning paths are repeated
    - Contradictions: When statements contradict earlier statements
    - Variable manipulation: When variable values change unexpectedly
    """
    
    def __init__(self, 
                cycle_threshold: int = 3,
                cycle_count_threshold: int = 5,
                var_threshold: int = 2,
                enable_logging: bool = True):
        """
        Initialize the reflection guard.
        
        Args:
            cycle_threshold: Number of times to allow the same pattern
            cycle_count_threshold: Total number of cycles to allow
            var_threshold: Number of times to allow variable redefinition
            enable_logging: Whether to log events
        """
        self.enable_logging = enable_logging
        self.cycle_threshold = cycle_threshold
        self.cycle_count_threshold = cycle_count_threshold
        self.var_threshold = var_threshold
        
        # Reflection tracking
        self.reflection_store: Dict[str, List[Dict[str, Any]]] = {}
        self.reflection_hashes: Dict[str, Dict[str, int]] = {}
        self.pattern_counts: Dict[str, Dict[str, int]] = {}
        self.contradiction_count = 0
        self.cycle_count = 0
        
        # Variables tracking
        self.variable_store: Dict[str, Dict[str, Any]] = {}
        self.variable_history: Dict[str, Dict[str, List[Any]]] = {}
        
        # Statements tracking
        self.statement_store: Dict[str, Dict[str, Any]] = {}
        
        # Patterns for extraction
        self.var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;=]+)')
        self.statement_pattern = re.compile(r'\b(is|are|was|were|will be|has|have|had)\b')
        
        if self.enable_logging:
            logger.info("Reflection guard initialized with cycle_threshold=%d, "
                       "cycle_count_threshold=%d, var_threshold=%d",
                       cycle_threshold, cycle_count_threshold, var_threshold)
    
    def process_reflection(self, reflection_id: str, content: str) -> Dict[str, Any]:
        """
        Process a reflection to check for cycles and contradictions.
        
        Args:
            reflection_id: Unique identifier for the reflection series
            content: The content to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Initialize if first reflection
        if reflection_id not in self.reflection_store:
            self.reflection_store[reflection_id] = []
            self.reflection_hashes[reflection_id] = {}
            self.pattern_counts[reflection_id] = {}
            self.variable_store[reflection_id] = {}
            self.variable_history[reflection_id] = {}
            self.statement_store[reflection_id] = {}
            
        # Hash the content for cycle detection
        content_hash = self._hash_content(content)
        
        # Extract variables and statements
        variables = self._extract_variables(content)
        statements = self._extract_statements(content)
        
        # Check for cycles
        cycle_detected = self._check_for_cycles(reflection_id, content_hash)
        
        # Check for contradictions
        contradiction_info = self._check_for_contradictions(reflection_id, statements)
        contradiction_detected = contradiction_info["detected"]
        
        # Check for variable manipulation
        var_manipulation = self._check_variable_manipulation(reflection_id, variables)
        
        # Store this reflection
        reflection_entry = {
            "timestamp": datetime.now().timestamp(),
            "content_hash": content_hash,
            "content_sample": content[:100] if len(content) > 100 else content,
            "variables": variables,
            "statements": statements,
            "cycle_detected": cycle_detected,
            "contradiction_detected": contradiction_detected
        }
        self.reflection_store[reflection_id].append(reflection_entry)
        
        # Update counters
        if cycle_detected:
            self.cycle_count += 1
            
        if contradiction_detected:
            self.contradiction_count += 1
            
        return {
            "reflection_id": reflection_id,
            "cycle_detected": cycle_detected,
            "contradiction_detected": contradiction_detected,
            "contradiction_details": contradiction_info.get("details", []),
            "variable_manipulation": var_manipulation,
            "cycles_total": self.cycle_count,
            "contradictions_total": self.contradiction_count
        }
    
    def get_contradiction_count(self) -> int:
        """Get the total number of contradictions detected."""
        return self.contradiction_count
    
    def get_cycle_count(self) -> int:
        """Get the total number of cycles detected."""
        return self.cycle_count
        
    def get_reflection_history(self, reflection_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of a specific reflection series.
        
        Args:
            reflection_id: ID of the reflection series
            
        Returns:
            List of reflection entries
        """
        return self.reflection_store.get(reflection_id, [])
    
    def reset_counters(self) -> None:
        """Reset the contradiction and cycle counters."""
        self.contradiction_count = 0
        self.cycle_count = 0
        
    def reset(self) -> None:
        """Reset all stored state."""
        self.reflection_store = {}
        self.reflection_hashes = {}
        self.pattern_counts = {}
        self.variable_store = {}
        self.variable_history = {}
        self.statement_store = {}
        self.contradiction_count = 0
        self.cycle_count = 0
        
    def _hash_content(self, content: str) -> str:
        """Create a hash of the content for tracking."""
        # Normalize whitespace and lowercase
        normalized = re.sub(r'\s+', ' ', content.lower()).strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _extract_variables(self, content: str) -> Dict[str, Any]:
        """Extract variable assignments from content."""
        variables = {}
        for match in self.var_pattern.finditer(content):
            var_name, var_value = match.groups()
            var_value = var_value.strip()
            variables[var_name] = var_value
        return variables
    
    def _extract_statements(self, content: str) -> Dict[str, Any]:
        """Extract statements with is/are/etc from content."""
        statements = {}
        # Split by sentences approximately
        sentences = re.split(r'[.!?]\s+', content)
        
        for i, sentence in enumerate(sentences):
            if self.statement_pattern.search(sentence):
                # This is a statement of fact
                statement_id = f"s{i}"
                statements[statement_id] = {
                    "text": sentence.strip(),
                    "extracted_at": datetime.now().timestamp()
                }
                
        return statements
    
    def _check_for_cycles(self, reflection_id: str, content_hash: str) -> bool:
        """
        Check if we're in a reasoning cycle.
        
        Args:
            reflection_id: ID of the reflection series
            content_hash: Hash of the current content
            
        Returns:
            Boolean indicating if a cycle was detected
        """
        # Get hash count
        hash_count = self.reflection_hashes[reflection_id].get(content_hash, 0) + 1
        self.reflection_hashes[reflection_id][content_hash] = hash_count
        
        # Check if we're repeating too much
        if hash_count > self.cycle_threshold:
            if self.enable_logging:
                logger.warning(
                    "Cycle detected in reflection %s - hash %s repeated %d times",
                    reflection_id, content_hash[:8], hash_count
                )
            return True
            
        # Check total cycles
        total_cycles = sum(1 for count in self.reflection_hashes[reflection_id].values() 
                         if count > 1)
        if total_cycles > self.cycle_count_threshold:
            if self.enable_logging:
                logger.warning(
                    "Too many cycles in reflection %s - %d different cycles detected",
                    reflection_id, total_cycles
                )
            return True
            
        return False
        
    def _check_for_contradictions(self, 
                                 reflection_id: str, 
                                 statements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for contradictions between current and previous statements.
        
        Args:
            reflection_id: ID of the reflection series
            statements: New statements to check
            
        Returns:
            Dictionary with contradiction information
        """
        result = {
            "detected": False,
            "details": []
        }
        
        # Simple contradictions (negations)
        previous_statements = self.statement_store.get(reflection_id, {})
        
        # Store all new statements
        for statement_id, statement_data in statements.items():
            self.statement_store.setdefault(reflection_id, {})[statement_id] = statement_data
            
        # Skip if no previous statements
        if not previous_statements:
            return result
            
        # Check contradictions
        for new_id, new_data in statements.items():
            new_text = new_data["text"].lower()
            
            # Check against previous statements
            for old_id, old_data in previous_statements.items():
                old_text = old_data["text"].lower()
                
                # Check for direct contradiction
                if self._are_statements_contradictory(old_text, new_text):
                    if self.enable_logging:
                        logger.warning(
                            "Contradiction detected in reflection %s between '%s' and '%s'",
                            reflection_id, old_text, new_text
                        )
                        
                    result["detected"] = True
                    result["details"].append({
                        "statement1": old_text,
                        "statement2": new_text,
                        "type": "direct_contradiction"
                    })
                    
        return result
                    
    def _are_statements_contradictory(self, statement1: str, statement2: str) -> bool:
        """
        Check if two statements contradict each other.
        This is a simplified implementation.
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            Boolean indicating if the statements contradict
        """
        # Simple negation check
        # Look for "not" or "n't" in one statement but not the other
        # when the rest of the statement is similar
        
        # Normalize statements
        s1 = statement1.lower().strip()
        s2 = statement2.lower().strip()
        
        # Check for negation
        s1_has_negation = ('not ' in s1) or (" isn't " in s1) or (" aren't " in s1) or (" doesn't " in s1) or (" don't " in s1)
        s2_has_negation = ('not ' in s2) or (" isn't " in s2) or (" aren't " in s2) or (" doesn't " in s2) or (" don't " in s2)
        
        # If one has negation and the other doesn't
        if s1_has_negation != s2_has_negation:
            # Remove the negation and check if statements are similar
            s1_normalized = s1.replace('not ', ' ').replace(" isn't ", " is ").replace(" aren't ", " are ").replace(" doesn't ", " does ").replace(" don't ", " do ")
            s2_normalized = s2.replace('not ', ' ').replace(" isn't ", " is ").replace(" aren't ", " are ").replace(" doesn't ", " does ").replace(" don't ", " do ")
            
            # Calculate similarity (very simple approach)
            similarity = self._calculate_similarity(s1_normalized, s2_normalized)
            
            # If statements are similar enough after removing negation
            return similarity > 0.6
            
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0
            
        return intersection / union
        
    def _check_variable_manipulation(self, 
                                    reflection_id: str, 
                                    variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for problematic variable manipulation.
        
        Args:
            reflection_id: ID of the reflection series
            variables: New variables to check
            
        Returns:
            Dictionary with variable manipulation information
        """
        result = {
            "detected": False,
            "details": []
        }
        
        # Check each variable
        for var_name, var_value in variables.items():
            # If variable exists and has a different value
            if var_name in self.variable_store.get(reflection_id, {}):
                old_value = self.variable_store[reflection_id][var_name]
                
                # Track history
                self.variable_history.setdefault(reflection_id, {}).setdefault(var_name, []).append({
                    "value": var_value,
                    "timestamp": datetime.now().timestamp()
                })
                
                # Check redefinition count
                redefinition_count = len(self.variable_history[reflection_id][var_name])
                
                if redefinition_count > self.var_threshold:
                    if self.enable_logging:
                        logger.warning(
                            "Variable manipulation detected in reflection %s - "
                            "variable %s redefined %d times, latest: %s -> %s",
                            reflection_id, var_name, redefinition_count, 
                            old_value, var_value
                        )
                        
                    result["detected"] = True
                    result["details"].append({
                        "variable": var_name,
                        "old_value": old_value,
                        "new_value": var_value,
                        "redefinition_count": redefinition_count
                    })
            
            # Store the current value
            self.variable_store.setdefault(reflection_id, {})[var_name] = var_value
            
        return result


class ReflectionValidator:
    """
    Validates reflection content against safety rules.
    
    Enforces:
    - Content restrictions
    - Safe reasoning patterns 
    - Output validation
    """
    
    def __init__(self, 
                reflection_guard: ReflectionGuard,
                max_cycles: int = 3,
                max_contradictions: int = 2):
        """
        Initialize the validator.
        
        Args:
            reflection_guard: Guard to use for validation
            max_cycles: Maximum allowable cycles
            max_contradictions: Maximum allowable contradictions
        """
        self.reflection_guard = reflection_guard
        self.max_cycles = max_cycles
        self.max_contradictions = max_contradictions
        
    def validate(self, reflection_id: str, content: str) -> Dict[str, Any]:
        """
        Validate a reflection for safety.
        
        Args:
            reflection_id: ID of the reflection
            content: Content to validate
            
        Returns:
            Dictionary with validation results
        """
        # Process with guard
        guard_result = self.reflection_guard.process_reflection(reflection_id, content)
        
        # Check against thresholds
        is_valid = True
        reasons = []
        
        if guard_result["cycle_detected"]:
            is_valid = False
            reasons.append("Reasoning cycle detected")
            
        if guard_result["contradiction_detected"]:
            is_valid = False
            reasons.append("Contradiction detected")
            
        if guard_result["cycles_total"] > self.max_cycles:
            is_valid = False
            reasons.append(f"Too many reasoning cycles: {guard_result['cycles_total']}")
            
        if guard_result["contradictions_total"] > self.max_contradictions:
            is_valid = False
            reasons.append(f"Too many contradictions: {guard_result['contradictions_total']}")
            
        return {
            "is_valid": is_valid,
            "reasons": reasons,
            "guard_result": guard_result
        }
        
    def reset(self) -> None:
        """Reset the validator and guard."""
        self.reflection_guard.reset()


def create_reflection_guard_with_validator() -> Tuple[ReflectionGuard, ReflectionValidator]:
    """
    Create a reflection guard and validator with default settings.
    
    Returns:
        Tuple of (ReflectionGuard, ReflectionValidator)
    """
    guard = ReflectionGuard(
        cycle_threshold=3,
        cycle_count_threshold=5,
        var_threshold=2
    )
    
    validator = ReflectionValidator(
        reflection_guard=guard,
        max_cycles=3,
        max_contradictions=2
    )
    
    return guard, validator 