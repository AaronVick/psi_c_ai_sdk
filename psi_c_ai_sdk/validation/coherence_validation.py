"""
Coherence Validation Module

This module provides mechanisms to validate coherence assessments and 
detect false positives (when an agent "hallucinates" coherence that isn't well-supported).

The validation approaches include:
1. Empirical coherence testing against ground truth
2. Adversarial coherence probing
3. Statistical anomaly detection for coherence metrics
4. Self-consistency checks on coherence justifications
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CoherenceErrorType(Enum):
    """Types of errors in coherence assessment."""
    FALSE_POSITIVE = "false_positive"  # System reports coherence that isn't there
    FALSE_NEGATIVE = "false_negative"  # System misses actual coherence
    OVERCONFIDENCE = "overconfidence"  # System reports higher coherence than justified
    UNDERCONFIDENCE = "underconfidence"  # System reports lower coherence than justified
    INCONSISTENT = "inconsistent"  # System's coherence assessment varies unpredictably

@dataclass
class CoherenceValidationResult:
    """Results from validating coherence assessment."""
    is_valid: bool
    error_type: Optional[CoherenceErrorType] = None
    confidence: float = 1.0
    explanation: str = ""
    suggested_correction: Optional[float] = None
    diagnostic_info: Dict[str, Any] = None

class CoherenceValidator:
    """
    Validates coherence assessments to detect false positives and other errors.
    """
    
    def __init__(self, 
                sensitivity: float = 0.7,
                enable_adversarial_probing: bool = True,
                statistical_threshold: float = 2.0,
                min_justification_ratio: float = 0.5):
        """
        Initialize the coherence validator.
        
        Args:
            sensitivity: How sensitive the detector is to potential errors (0-1)
            enable_adversarial_probing: Whether to use adversarial probing
            statistical_threshold: Z-score threshold for statistical anomalies
            min_justification_ratio: Minimum ratio of justified to total coherence claims
        """
        self.sensitivity = sensitivity
        self.enable_adversarial_probing = enable_adversarial_probing
        self.statistical_threshold = statistical_threshold
        self.min_justification_ratio = min_justification_ratio
        
        # History of coherence assessments for statistical analysis
        self.coherence_history = []
        self.error_history = []
    
    def validate_coherence(self, 
                          coherence_score: float,
                          beliefs: List[Dict[str, Any]],
                          justifications: List[Dict[str, Any]] = None) -> CoherenceValidationResult:
        """
        Validate a coherence assessment to detect possible errors.
        
        Args:
            coherence_score: The coherence score to validate (0-1)
            beliefs: The beliefs used in the coherence assessment
            justifications: Justifications for the coherence assessment
            
        Returns:
            Validation result indicating whether the coherence assessment is valid
        """
        # Run validation checks
        results = []
        
        # 1. Check empirical coherence
        empirical_result = self._validate_empirical_coherence(
            coherence_score, beliefs, justifications
        )
        results.append(empirical_result)
        
        # 2. Check with adversarial probing if enabled
        if self.enable_adversarial_probing:
            adversarial_result = self._validate_with_adversarial_probing(
                coherence_score, beliefs
            )
            results.append(adversarial_result)
        
        # 3. Check for statistical anomalies
        statistical_result = self._validate_statistical_coherence(coherence_score)
        results.append(statistical_result)
        
        # 4. Check self-consistency
        if justifications:
            consistency_result = self._validate_self_consistency(
                coherence_score, beliefs, justifications
            )
            results.append(consistency_result)
        
        # Aggregate results
        is_valid = all(result.is_valid for result in results)
        
        # If any check failed, identify the most severe error
        error_type = None
        explanation = ""
        confidence = 1.0
        suggested_correction = None
        
        if not is_valid:
            # Find the most confident invalid result
            invalid_results = [r for r in results if not r.is_valid]
            most_confident = max(invalid_results, key=lambda r: r.confidence)
            
            error_type = most_confident.error_type
            explanation = most_confident.explanation
            confidence = most_confident.confidence
            suggested_correction = most_confident.suggested_correction
        
        # Update history
        self._update_history(coherence_score, is_valid, error_type)
        
        # Return combined result
        return CoherenceValidationResult(
            is_valid=is_valid,
            error_type=error_type,
            confidence=confidence,
            explanation=explanation,
            suggested_correction=suggested_correction,
            diagnostic_info={
                "num_beliefs": len(beliefs),
                "num_justifications": len(justifications) if justifications else 0,
                "empirical_valid": empirical_result.is_valid,
                "statistical_valid": statistical_result.is_valid,
                "adversarial_valid": adversarial_result.is_valid if self.enable_adversarial_probing else None,
                "consistency_valid": consistency_result.is_valid if justifications else None
            }
        )
    
    def _validate_empirical_coherence(self,
                                     coherence_score: float,
                                     beliefs: List[Dict[str, Any]],
                                     justifications: List[Dict[str, Any]] = None) -> CoherenceValidationResult:
        """
        Validate coherence by empirically measuring belief consistency.
        
        Args:
            coherence_score: The coherence score to validate
            beliefs: The beliefs used in the coherence assessment
            justifications: Justifications for the coherence assessment
            
        Returns:
            Validation result
        """
        # Count contradictions in beliefs
        contradictions = self._count_contradictions(beliefs)
        
        # Calculate empirical coherence based on contradiction ratio
        if len(beliefs) <= 1:
            empirical_coherence = 1.0  # By definition, a single belief is coherent with itself
        else:
            max_possible_contradictions = (len(beliefs) * (len(beliefs) - 1)) / 2
            contradiction_ratio = contradictions / max_possible_contradictions if max_possible_contradictions > 0 else 0
            empirical_coherence = 1.0 - contradiction_ratio
        
        # Determine if there's a significant discrepancy
        discrepancy = abs(coherence_score - empirical_coherence)
        is_valid = discrepancy <= (1.0 - self.sensitivity)
        
        # Detect type of error if invalid
        error_type = None
        explanation = ""
        suggested_correction = None
        
        if not is_valid:
            if coherence_score > empirical_coherence:
                error_type = CoherenceErrorType.FALSE_POSITIVE
                explanation = f"Reported coherence ({coherence_score:.2f}) is significantly higher than empirical coherence ({empirical_coherence:.2f})"
                suggested_correction = empirical_coherence
            else:
                error_type = CoherenceErrorType.FALSE_NEGATIVE
                explanation = f"Reported coherence ({coherence_score:.2f}) is significantly lower than empirical coherence ({empirical_coherence:.2f})"
                suggested_correction = empirical_coherence
        
        return CoherenceValidationResult(
            is_valid=is_valid,
            error_type=error_type,
            confidence=min(1.0, discrepancy * 2),  # Higher discrepancy = higher confidence in error
            explanation=explanation,
            suggested_correction=suggested_correction
        )
    
    def _validate_with_adversarial_probing(self,
                                          coherence_score: float,
                                          beliefs: List[Dict[str, Any]]) -> CoherenceValidationResult:
        """
        Validate coherence by introducing adversarial probes and checking if
        coherence assessment correctly identifies the impact.
        
        Args:
            coherence_score: The coherence score to validate
            beliefs: The beliefs used in the coherence assessment
            
        Returns:
            Validation result
        """
        # This would insert contradicting beliefs and check if coherence drops
        # For now, simulate with heuristic checks
        
        # Simple heuristic: If coherence is very high with many beliefs, it's suspicious
        if coherence_score > 0.9 and len(beliefs) > 10:
            # High likelihood this is a false positive
            return CoherenceValidationResult(
                is_valid=False,
                error_type=CoherenceErrorType.OVERCONFIDENCE,
                confidence=0.7,
                explanation=f"Suspiciously high coherence ({coherence_score:.2f}) given the number of beliefs ({len(beliefs)})",
                suggested_correction=max(0.5, coherence_score - 0.2)
            )
        
        # Another heuristic: If beliefs have very different topics but coherence is high
        topics = set()
        for belief in beliefs:
            if "topic" in belief:
                topics.add(belief["topic"])
            elif "metadata" in belief and "topic" in belief["metadata"]:
                topics.add(belief["metadata"]["topic"])
        
        if len(topics) > 5 and coherence_score > 0.8:
            # Likely false positive
            return CoherenceValidationResult(
                is_valid=False,
                error_type=CoherenceErrorType.OVERCONFIDENCE,
                confidence=0.6,
                explanation=f"High coherence ({coherence_score:.2f}) despite diverse topics ({len(topics)})",
                suggested_correction=max(0.4, coherence_score - 0.3)
            )
        
        # Valid if no clear problems detected
        return CoherenceValidationResult(
            is_valid=True,
            confidence=0.8
        )
    
    def _validate_statistical_coherence(self, coherence_score: float) -> CoherenceValidationResult:
        """
        Validate coherence by checking if it's a statistical outlier.
        
        Args:
            coherence_score: The coherence score to validate
            
        Returns:
            Validation result
        """
        # Need enough history for statistical analysis
        if len(self.coherence_history) < 5:
            return CoherenceValidationResult(
                is_valid=True,
                confidence=0.5,
                explanation="Insufficient history for statistical validation"
            )
        
        # Calculate mean and standard deviation
        mean = np.mean(self.coherence_history)
        std = np.std(self.coherence_history)
        
        if std < 0.001:
            # Avoid division by zero and detect suspiciously constant coherence
            return CoherenceValidationResult(
                is_valid=False,
                error_type=CoherenceErrorType.INCONSISTENT,
                confidence=0.9,
                explanation="Coherence values show suspiciously little variation",
                suggested_correction=None
            )
        
        # Calculate z-score
        z_score = abs(coherence_score - mean) / std
        
        # Check if it's an outlier
        is_valid = z_score <= self.statistical_threshold
        
        if not is_valid:
            if coherence_score > mean:
                error_type = CoherenceErrorType.OVERCONFIDENCE
                explanation = f"Coherence score ({coherence_score:.2f}) is a statistical outlier (z={z_score:.2f}), much higher than typical values"
                suggested_correction = mean + (std * 0.5)  # More conservative estimate
            else:
                error_type = CoherenceErrorType.UNDERCONFIDENCE
                explanation = f"Coherence score ({coherence_score:.2f}) is a statistical outlier (z={z_score:.2f}), much lower than typical values"
                suggested_correction = mean - (std * 0.5)  # More conservative estimate
                
            return CoherenceValidationResult(
                is_valid=False,
                error_type=error_type,
                confidence=min(1.0, z_score / (self.statistical_threshold * 2)),
                explanation=explanation,
                suggested_correction=suggested_correction
            )
        
        return CoherenceValidationResult(
            is_valid=True,
            confidence=1.0 - (z_score / self.statistical_threshold)
        )
    
    def _validate_self_consistency(self,
                                  coherence_score: float,
                                  beliefs: List[Dict[str, Any]],
                                  justifications: List[Dict[str, Any]]) -> CoherenceValidationResult:
        """
        Validate coherence by checking if justifications properly support the score.
        
        Args:
            coherence_score: The coherence score to validate
            beliefs: The beliefs used in the coherence assessment
            justifications: Justifications for the coherence assessment
            
        Returns:
            Validation result
        """
        # Check if there are enough justifications relative to beliefs
        if len(justifications) < len(beliefs) * self.min_justification_ratio:
            return CoherenceValidationResult(
                is_valid=False,
                error_type=CoherenceErrorType.OVERCONFIDENCE,
                confidence=0.7,
                explanation=f"Insufficient justifications ({len(justifications)}) for the number of beliefs ({len(beliefs)})",
                suggested_correction=coherence_score * 0.8  # Reduce confidence due to lack of justification
            )
        
        # Check justification quality (simplified)
        weak_justifications = 0
        for j in justifications:
            strength = j.get("strength", 0.5)
            if strength < 0.6:  # Arbitrary threshold for "weak" justification
                weak_justifications += 1
        
        # If more than half of justifications are weak, flag as potential false positive
        if weak_justifications > len(justifications) / 2 and coherence_score > 0.7:
            return CoherenceValidationResult(
                is_valid=False,
                error_type=CoherenceErrorType.OVERCONFIDENCE,
                confidence=0.8,
                explanation=f"High coherence ({coherence_score:.2f}) despite predominantly weak justifications",
                suggested_correction=coherence_score * 0.7  # Reduce confidence due to weak justifications
            )
        
        # Valid if no clear problems detected
        return CoherenceValidationResult(
            is_valid=True,
            confidence=0.9
        )
    
    def _count_contradictions(self, beliefs: List[Dict[str, Any]]) -> int:
        """
        Count contradictions between beliefs.
        
        Args:
            beliefs: List of beliefs
            
        Returns:
            Number of contradictions detected
        """
        # This is a simplified implementation
        # A real implementation would have more sophisticated contradiction detection
        
        contradictions = 0
        for i, belief1 in enumerate(beliefs):
            for belief2 in beliefs[i+1:]:
                # Check if beliefs have contradiction markers
                if "contradicts" in belief1 and belief1["contradicts"] == belief2.get("id"):
                    contradictions += 1
                elif "contradicts" in belief2 and belief2["contradicts"] == belief1.get("id"):
                    contradictions += 1
                # Check content if available (very simplified)
                elif "content" in belief1 and "content" in belief2:
                    # Extremely simplified contradiction detection 
                    # Real implementation would use semantic analysis
                    content1 = belief1["content"].lower()
                    content2 = belief2["content"].lower()
                    
                    # Check for direct negation
                    if "not" in content1 and content1.replace("not", "").strip() in content2:
                        contradictions += 1
                    elif "not" in content2 and content2.replace("not", "").strip() in content1:
                        contradictions += 1
        
        return contradictions
    
    def _update_history(self, 
                       coherence_score: float,
                       is_valid: bool,
                       error_type: Optional[CoherenceErrorType]) -> None:
        """
        Update history with new coherence assessment.
        
        Args:
            coherence_score: The coherence score
            is_valid: Whether it was valid
            error_type: Type of error if any
        """
        self.coherence_history.append(coherence_score)
        
        # Limit history size
        if len(self.coherence_history) > 100:
            self.coherence_history.pop(0)
        
        # Record error if any
        if not is_valid and error_type:
            self.error_history.append((coherence_score, error_type))
            
            # Limit error history size
            if len(self.error_history) > 50:
                self.error_history.pop(0)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about coherence validation errors.
        
        Returns:
            Dictionary with error statistics
        """
        # Count errors by type
        error_counts = {
            error_type.value: 0 for error_type in CoherenceErrorType
        }
        
        for _, error_type in self.error_history:
            error_counts[error_type.value] += 1
        
        # Calculate error rate
        total_validations = len(self.coherence_history)
        error_rate = len(self.error_history) / total_validations if total_validations > 0 else 0
        
        return {
            "total_validations": total_validations,
            "error_rate": error_rate,
            "error_counts": error_counts,
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }
    
    def reset_history(self) -> None:
        """Reset the validation history."""
        self.coherence_history = []
        self.error_history = []


class CoherenceFalsificationTester:
    """
    Tests coherence assessment by deliberately introducing inconsistencies
    and checking if the coherence system properly detects them.
    """
    
    def __init__(self, coherence_calculator: Any):
        """
        Initialize the falsification tester.
        
        Args:
            coherence_calculator: The coherence calculator to test
        """
        self.coherence_calculator = coherence_calculator
        self.test_results = []
    
    def run_falsification_tests(self, 
                               base_beliefs: List[Dict[str, Any]],
                               num_tests: int = 5) -> Dict[str, Any]:
        """
        Run a series of falsification tests on the coherence calculator.
        
        Args:
            base_beliefs: Starting set of consistent beliefs
            num_tests: Number of tests to run
            
        Returns:
            Dictionary with test results and statistics
        """
        self.test_results = []
        
        # Run baseline test with consistent beliefs
        baseline_coherence = self.coherence_calculator.calculate_coherence(base_beliefs)
        self.test_results.append({
            "test_type": "baseline",
            "coherence": baseline_coherence,
            "expected_coherence": "high",
            "beliefs": base_beliefs,
            "passed": baseline_coherence > 0.7  # Expect high coherence for consistent beliefs
        })
        
        # Run tests with various inconsistency types
        for i in range(num_tests):
            # Create test case with inconsistencies
            test_beliefs, expected_coherence = self._create_test_case(
                base_beliefs=base_beliefs.copy(),
                test_type=i % 3  # Cycle through different test types
            )
            
            # Calculate coherence
            test_coherence = self.coherence_calculator.calculate_coherence(test_beliefs)
            
            # Determine if test passed based on expected coherence
            passed = False
            if expected_coherence == "low" and test_coherence < 0.5:
                passed = True
            elif expected_coherence == "medium" and 0.3 <= test_coherence <= 0.7:
                passed = True
            elif expected_coherence == "high" and test_coherence > 0.7:
                passed = True
            
            # Store test result
            self.test_results.append({
                "test_type": f"test_{i+1}",
                "coherence": test_coherence,
                "expected_coherence": expected_coherence,
                "beliefs": test_beliefs,
                "passed": passed
            })
        
        # Calculate statistics
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        pass_rate = passed_tests / len(self.test_results)
        
        return {
            "pass_rate": pass_rate,
            "passed_tests": passed_tests,
            "total_tests": len(self.test_results),
            "results": self.test_results
        }
    
    def _create_test_case(self, 
                         base_beliefs: List[Dict[str, Any]],
                         test_type: int) -> Tuple[List[Dict[str, Any]], str]:
        """
        Create a test case with specific types of inconsistencies.
        
        Args:
            base_beliefs: Starting set of consistent beliefs
            test_type: Type of test to create (0: direct contradictions, 
                      1: semantic inconsistencies, 2: logical inconsistencies)
            
        Returns:
            Tuple of (test_beliefs, expected_coherence)
        """
        test_beliefs = base_beliefs.copy()
        
        if test_type == 0:
            # Direct contradictions
            for i in range(min(3, len(test_beliefs))):
                if "content" in test_beliefs[i]:
                    content = test_beliefs[i]["content"]
                    test_beliefs[i]["content"] = "It is not true that " + content
                    test_beliefs[i]["contradicts"] = test_beliefs[i].get("id")
            
            expected_coherence = "low"
            
        elif test_type == 1:
            # Semantic inconsistencies (more subtle)
            new_beliefs = [
                {"id": "inconsistent1", "content": "Water boils at 50 degrees Celsius", "confidence": 0.8},
                {"id": "inconsistent2", "content": "Humans can survive without oxygen", "confidence": 0.7},
                {"id": "inconsistent3", "content": "The Earth is both spherical and flat", "confidence": 0.9}
            ]
            
            test_beliefs.extend(new_beliefs)
            expected_coherence = "medium"
            
        else:
            # Logical inconsistencies (subtler still)
            new_beliefs = [
                {"id": "logical1", "content": "If A then B", "confidence": 0.9},
                {"id": "logical2", "content": "A is true", "confidence": 0.9},
                {"id": "logical3", "content": "B is false", "confidence": 0.8}
            ]
            
            test_beliefs.extend(new_beliefs)
            expected_coherence = "medium"
        
        return test_beliefs, expected_coherence 