"""
Coherence-Weighted Dilemma Engine

When presented with an ethical or logical dilemma, this module evaluates choices
based on internal coherence gain, not just utility. This enables the agent to reason
inwardly as part of moral judgment.

Formula:
    Utility_ΨC = ΔΨC + ΔA + ΔG
    
    Where:
    - ΔΨC: coherence gain
    - ΔA: alignment vector shift
    - ΔG: goal vector shift
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field

@dataclass
class Option:
    """Represents a choice option in a dilemma."""
    
    id: str
    description: str
    standard_utility: float = 0.0
    coherence_delta: float = 0.0
    alignment_delta: float = 0.0
    goal_delta: float = 0.0
    psi_c_utility: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_psi_c_utility(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate the ΨC-weighted utility of this option.
        
        Args:
            weights: Optional dictionary of weights for each component
            
        Returns:
            The calculated ΨC utility
        """
        w = weights or {"coherence": 1.0, "alignment": 1.0, "goal": 1.0}
        
        self.psi_c_utility = (
            w.get("coherence", 1.0) * self.coherence_delta +
            w.get("alignment", 1.0) * self.alignment_delta +
            w.get("goal", 1.0) * self.goal_delta
        )
        
        return self.psi_c_utility
    
    def is_coherent(self, threshold: float = 0.0) -> bool:
        """Check if this option increases coherence."""
        return self.coherence_delta > threshold
    
    def is_aligned(self, threshold: float = 0.0) -> bool:
        """Check if this option maintains alignment."""
        return self.alignment_delta > threshold
    
    def supports_goals(self, threshold: float = 0.0) -> bool:
        """Check if this option advances goals."""
        return self.goal_delta > threshold


class DilemmaEngine:
    """Engine for evaluating options in a dilemma using coherence-weighted utility."""
    
    def __init__(self, 
                 alignment_vector: Optional[np.ndarray] = None,
                 goal_vector: Optional[np.ndarray] = None,
                 weightings: Optional[Dict[str, float]] = None):
        """
        Initialize the dilemma engine.
        
        Args:
            alignment_vector: Vector representing ethical alignment
            goal_vector: Vector representing goals
            weightings: Weights for the utility components
        """
        self.alignment_vector = alignment_vector
        self.goal_vector = goal_vector
        self.weightings = weightings or {
            "coherence": 1.0,
            "alignment": 1.0,
            "goal": 1.0
        }
        self.default_delta_calculator = self._default_delta_calculator
        self.history: List[Dict[str, Any]] = []
    
    def set_alignment_vector(self, vector: np.ndarray) -> None:
        """Set the alignment vector."""
        self.alignment_vector = vector
    
    def set_goal_vector(self, vector: np.ndarray) -> None:
        """Set the goal vector."""
        self.goal_vector = vector
    
    def set_weightings(self, weightings: Dict[str, float]) -> None:
        """Set the component weightings."""
        self.weightings = weightings
    
    def evaluate_option(self, option: Option) -> float:
        """
        Evaluate a single option.
        
        Args:
            option: The option to evaluate
            
        Returns:
            The ΨC-weighted utility of the option
        """
        return option.calculate_psi_c_utility(self.weightings)
    
    def evaluate_options(self, options: List[Option]) -> List[Option]:
        """
        Evaluate multiple options and sort by utility.
        
        Args:
            options: List of options to evaluate
            
        Returns:
            The options sorted by ΨC-weighted utility (highest first)
        """
        for option in options:
            self.evaluate_option(option)
            
        return sorted(options, key=lambda o: o.psi_c_utility, reverse=True)
    
    def solve_dilemma(self, options: List[Option]) -> Tuple[Option, Dict[str, Any]]:
        """
        Solve a dilemma by selecting the option with highest ΨC-weighted utility.
        
        Args:
            options: List of options
            
        Returns:
            Tuple of (best option, decision metrics)
        """
        ranked_options = self.evaluate_options(options)
        best_option = ranked_options[0]
        
        # Record the decision
        decision_record = {
            "options": [o.id for o in options],
            "rankings": [o.id for o in ranked_options],
            "winner": best_option.id,
            "coherence_delta": best_option.coherence_delta,
            "alignment_delta": best_option.alignment_delta,
            "goal_delta": best_option.goal_delta,
            "psi_c_utility": best_option.psi_c_utility,
            "standard_utility": best_option.standard_utility,
            "utility_delta": best_option.psi_c_utility - best_option.standard_utility
        }
        
        self.history.append(decision_record)
        
        return best_option, decision_record
    
    def calculate_option_deltas(self, 
                              options: List[Dict[str, Any]],
                              coherence_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
                              alignment_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
                              goal_fn: Optional[Callable[[Dict[str, Any]], float]] = None) -> List[Option]:
        """
        Calculate deltas for a list of option dictionaries.
        
        Args:
            options: List of option dictionaries
            coherence_fn: Function to calculate coherence delta
            alignment_fn: Function to calculate alignment delta
            goal_fn: Function to calculate goal delta
            
        Returns:
            List of Option objects with deltas calculated
        """
        coherence_fn = coherence_fn or self.default_delta_calculator
        alignment_fn = alignment_fn or self.default_delta_calculator
        goal_fn = goal_fn or self.default_delta_calculator
        
        result = []
        
        for i, opt in enumerate(options):
            option = Option(
                id=opt.get("id", f"option_{i}"),
                description=opt.get("description", ""),
                standard_utility=opt.get("utility", 0.0),
                metadata=opt.get("metadata", {})
            )
            
            option.coherence_delta = coherence_fn(opt)
            option.alignment_delta = alignment_fn(opt)
            option.goal_delta = goal_fn(opt)
            option.calculate_psi_c_utility(self.weightings)
            
            result.append(option)
            
        return result
    
    def _default_delta_calculator(self, option: Dict[str, Any]) -> float:
        """
        Default calculator that looks for pre-calculated deltas.
        
        Args:
            option: Option dictionary
            
        Returns:
            Delta value
        """
        return (
            option.get("coherence_delta", 0.0) or 
            option.get("delta", 0.0) or
            0.0
        )
    
    def evaluate_coherence_delta(self, option: Dict[str, Any], 
                               current_state: Dict[str, Any],
                               simulate_fn: Callable[[Dict[str, Any], Dict[str, Any]], float]) -> float:
        """
        Evaluate coherence delta for an option by simulation.
        
        Args:
            option: Option dictionary
            current_state: Current state
            simulate_fn: Function to simulate option and measure coherence
            
        Returns:
            Coherence delta
        """
        return simulate_fn(option, current_state)
    
    def evaluate_alignment_delta(self, option: Dict[str, Any],
                               option_vector: np.ndarray) -> float:
        """
        Evaluate alignment delta by comparing with alignment vector.
        
        Args:
            option: Option dictionary
            option_vector: Vector representation of the option
            
        Returns:
            Alignment delta
        """
        if self.alignment_vector is None:
            return 0.0
            
        # Calculate cosine similarity
        similarity = np.dot(option_vector, self.alignment_vector) / (
            np.linalg.norm(option_vector) * np.linalg.norm(self.alignment_vector)
        )
        
        return similarity
    
    def evaluate_goal_delta(self, option: Dict[str, Any],
                          option_vector: np.ndarray) -> float:
        """
        Evaluate goal delta by comparing with goal vector.
        
        Args:
            option: Option dictionary
            option_vector: Vector representation of the option
            
        Returns:
            Goal delta
        """
        if self.goal_vector is None:
            return 0.0
            
        # Calculate cosine similarity
        similarity = np.dot(option_vector, self.goal_vector) / (
            np.linalg.norm(option_vector) * np.linalg.norm(self.goal_vector)
        )
        
        return similarity
    
    def get_utility_explanation(self, option: Option) -> str:
        """
        Generate an explanation of why an option has its utility.
        
        Args:
            option: Option to explain
            
        Returns:
            Explanation string
        """
        explanation = f"Option '{option.id}' has ΨC utility {option.psi_c_utility:.2f}:\n"
        explanation += f"- Coherence impact: {option.coherence_delta:.2f} × {self.weightings.get('coherence', 1.0):.1f}\n"
        explanation += f"- Alignment impact: {option.alignment_delta:.2f} × {self.weightings.get('alignment', 1.0):.1f}\n"
        explanation += f"- Goal impact: {option.goal_delta:.2f} × {self.weightings.get('goal', 1.0):.1f}\n"
        
        if option.standard_utility != 0:
            explanation += f"- Standard utility: {option.standard_utility:.2f}\n"
            explanation += f"- Utility perspective shift: {option.psi_c_utility - option.standard_utility:.2f}\n"
        
        return explanation
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """
        Get statistics about past decisions.
        
        Returns:
            Dictionary of decision statistics
        """
        if not self.history:
            return {"decisions": 0}
            
        coherence_wins = sum(1 for d in self.history 
                            if d["coherence_delta"] > d["alignment_delta"] 
                            and d["coherence_delta"] > d["goal_delta"])
        
        alignment_wins = sum(1 for d in self.history 
                           if d["alignment_delta"] > d["coherence_delta"] 
                           and d["alignment_delta"] > d["goal_delta"])
        
        goal_wins = sum(1 for d in self.history 
                      if d["goal_delta"] > d["coherence_delta"] 
                      and d["goal_delta"] > d["alignment_delta"])
        
        avg_utility_shift = sum(d["utility_delta"] for d in self.history) / len(self.history)
        
        return {
            "decisions": len(self.history),
            "coherence_dominant": coherence_wins,
            "alignment_dominant": alignment_wins,
            "goal_dominant": goal_wins,
            "avg_utility_shift": avg_utility_shift
        }


class MoralDilemmaEngine(DilemmaEngine):
    """Specialized engine for moral/ethical dilemmas."""
    
    def __init__(self, 
                 ethical_principles: Optional[Dict[str, float]] = None,
                 **kwargs):
        """
        Initialize the moral dilemma engine.
        
        Args:
            ethical_principles: Dictionary of ethical principles and their weights
            **kwargs: Additional arguments for DilemmaEngine
        """
        super().__init__(**kwargs)
        self.ethical_principles = ethical_principles or {
            "harm_avoidance": 1.0,
            "fairness": 1.0,
            "autonomy": 1.0,
            "loyalty": 1.0,
            "authority": 1.0,
            "purity": 1.0
        }
    
    def evaluate_moral_delta(self, option: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate an option against moral principles.
        
        Args:
            option: Option dictionary
            
        Returns:
            Dictionary of principle deltas
        """
        principles = option.get("principles", {})
        
        # Default to neutral (0.0) for missing principles
        result = {
            principle: principles.get(principle, 0.0)
            for principle in self.ethical_principles
        }
        
        return result
    
    def calculate_moral_alignment(self, principle_deltas: Dict[str, float]) -> float:
        """
        Calculate overall moral alignment from principle deltas.
        
        Args:
            principle_deltas: Dictionary of principle deltas
            
        Returns:
            Overall moral alignment score
        """
        # Weight each principle by its importance
        weighted_sum = sum(
            principle_deltas.get(principle, 0.0) * weight
            for principle, weight in self.ethical_principles.items()
        )
        
        # Normalize by sum of weights
        total_weight = sum(self.ethical_principles.values())
        
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight
    
    def solve_moral_dilemma(self, options: List[Dict[str, Any]]) -> Tuple[Option, Dict[str, Any]]:
        """
        Solve a moral dilemma.
        
        Args:
            options: List of option dictionaries
            
        Returns:
            Tuple of (best option, decision metrics)
        """
        option_objects = []
        
        for i, opt in enumerate(options):
            # Calculate moral deltas
            principle_deltas = self.evaluate_moral_delta(opt)
            moral_alignment = self.calculate_moral_alignment(principle_deltas)
            
            option = Option(
                id=opt.get("id", f"option_{i}"),
                description=opt.get("description", ""),
                standard_utility=opt.get("utility", 0.0),
                metadata={
                    "principles": principle_deltas,
                    "moral_alignment": moral_alignment,
                    **opt.get("metadata", {})
                }
            )
            
            # Set deltas
            option.coherence_delta = opt.get("coherence_delta", 0.0)
            option.alignment_delta = moral_alignment  # Use moral alignment as alignment delta
            option.goal_delta = opt.get("goal_delta", 0.0)
            
            option_objects.append(option)
            
        return self.solve_dilemma(option_objects)
    
    def get_moral_explanation(self, option: Option) -> str:
        """
        Generate a moral explanation for an option.
        
        Args:
            option: Option to explain
            
        Returns:
            Explanation string
        """
        explanation = self.get_utility_explanation(option)
        
        if "principles" in option.metadata:
            explanation += "\nMoral principle impacts:\n"
            
            for principle, value in option.metadata["principles"].items():
                weight = self.ethical_principles.get(principle, 1.0)
                explanation += f"- {principle}: {value:.2f} × {weight:.1f}\n"
        
        return explanation


def evaluate_dilemma(
    options: List[Dict[str, Any]],
    current_schema_state: Dict[str, Any],
    alignment_vector: Optional[np.ndarray] = None,
    goal_vector: Optional[np.ndarray] = None,
    is_moral_dilemma: bool = False
) -> Dict[str, Any]:
    """
    High-level function to evaluate a dilemma.
    
    Args:
        options: List of option dictionaries
        current_schema_state: Current schema state
        alignment_vector: Alignment vector
        goal_vector: Goal vector
        is_moral_dilemma: Whether this is a moral dilemma
        
    Returns:
        Decision results
    """
    # Calculate vectors if not provided
    if alignment_vector is None and "alignment_embedding" in current_schema_state:
        alignment_vector = np.array(current_schema_state["alignment_embedding"])
        
    if goal_vector is None and "goal_embedding" in current_schema_state:
        goal_vector = np.array(current_schema_state["goal_embedding"])
    
    # Select the appropriate engine
    if is_moral_dilemma:
        engine = MoralDilemmaEngine(
            alignment_vector=alignment_vector,
            goal_vector=goal_vector
        )
        best_option, decision = engine.solve_moral_dilemma(options)
    else:
        engine = DilemmaEngine(
            alignment_vector=alignment_vector,
            goal_vector=goal_vector
        )
        
        option_objects = engine.calculate_option_deltas(options)
        best_option, decision = engine.solve_dilemma(option_objects)
    
    explanation = engine.get_utility_explanation(best_option)
    
    return {
        "selected_option": best_option.id,
        "explanation": explanation,
        "decision": decision,
        "all_options": [
            {
                "id": o.id,
                "description": o.description,
                "psi_c_utility": o.psi_c_utility,
                "coherence_delta": o.coherence_delta,
                "alignment_delta": o.alignment_delta,
                "goal_delta": o.goal_delta,
                "standard_utility": o.standard_utility
            }
            for o in option_objects if isinstance(o, Option)
        ]
    } 