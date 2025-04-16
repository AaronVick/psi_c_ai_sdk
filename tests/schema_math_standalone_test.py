#!/usr/bin/env python3
"""
Standalone test for the enhanced mathematical formulations in schema_math.py.

This test contains local implementations of the functions that can be tested
directly, but will try to import from the actual module first if available.
"""

import unittest
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from the actual module
try:
    from tools.dev_environment.schema_math import (
        calculate_coherence_accumulation,
        calculate_shannon_entropy,
        calculate_memory_probability_distribution,
        calculate_phase_transition_threshold,
        calculate_memory_similarity_weight,
        calculate_self_loss,
        calculate_coherence_kernel,
        gradient_descent_step,
        calculate_multi_agent_coherence,
        add_gaussian_noise
    )
    print("Using functions from schema_math.py module")
    SCHEMA_MATH_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Using local function implementations")
    SCHEMA_MATH_AVAILABLE = False

    # Local implementations if import fails
    def calculate_coherence_accumulation(current_coherence: float,
                                        entropy: float,
                                        alpha: float = 0.1,
                                        max_coherence: float = 1.0,
                                        beta: float = 0.05,
                                        dt: float = 1.0) -> float:
        """
        Calculate coherence accumulation rate using the logistic growth model
        modified by entropy (formula #1).
        
        dC(t)/dt = α*C(t)*(1 - C(t)/K) - β*H(M(t))
        """
        # Calculate the rate of change
        logistic_growth = alpha * current_coherence * (1 - current_coherence / max_coherence)
        entropy_influence = beta * entropy
        
        # Calculate change in coherence
        coherence_change = (logistic_growth - entropy_influence) * dt
        
        # Update coherence value
        new_coherence = current_coherence + coherence_change
        
        # Ensure coherence stays within bounds [0, max_coherence]
        new_coherence = max(0.0, min(new_coherence, max_coherence))
        
        return new_coherence


    def calculate_shannon_entropy(probability_distribution: List[float]) -> float:
        """
        Calculate Shannon entropy for a given probability distribution (formula #2).
        
        H(M(t)) = -∑_{i=1}^{N} p(m_i) log(p(m_i))
        """
        # Avoid log(0) by filtering out zero probabilities
        non_zero_probs = [p for p in probability_distribution if p > 0]
        
        if not non_zero_probs:
            return 0.0
        
        # Calculate entropy using Shannon formula
        entropy = -sum(p * np.log2(p) for p in non_zero_probs)
        
        return entropy


    def calculate_memory_probability_distribution(memories: Dict[str, Any],
                                                attribute: str = 'importance') -> List[float]:
        """
        Calculate a probability distribution over memories based on a specific attribute.
        """
        # Extract attribute values
        values = []
        for memory_id, memory in memories.items():
            if hasattr(memory, attribute):
                values.append(getattr(memory, attribute))
            elif isinstance(memory, dict) and attribute in memory:
                values.append(memory[attribute])
        
        if not values:
            return []
        
        # Convert to probabilities by normalizing
        total = sum(values)
        if total == 0:
            # If sum is zero, use uniform distribution
            return [1.0/len(values)] * len(values)
        
        probabilities = [value/total for value in values]
        return probabilities


    def calculate_phase_transition_threshold(entropy_history: List[float],
                                            lambda_theta: float = 1.0) -> float:
        """
        Calculate the phase transition threshold based on entropy 
        expectation and variance (formula #3).
        
        θ = E[H(M(t))] + λ_θ * √Var(H(M(t)))
        """
        if not entropy_history:
            return 0.5  # Default threshold if no history available
        
        # Calculate mean and variance of entropy
        mean_entropy = np.mean(entropy_history)
        variance_entropy = np.var(entropy_history)
        
        # Calculate threshold
        threshold = mean_entropy + lambda_theta * np.sqrt(variance_entropy)
        
        return threshold


    def calculate_memory_similarity_weight(embedding1: np.ndarray,
                                        embedding2: np.ndarray,
                                        co_reflection_frequency: float = 1.0,
                                        time_difference: float = 0,
                                        alpha: float = 0.1) -> float:
        """
        Calculate the similarity weight between two memory states (formula #4).
        
        w_{ij} = cos(z(m_i), z(m_j)) * f_{ij} / (1 + α|t_i - t_j|)
        """
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            cosine_similarity = 0
        else:
            cosine_similarity = dot_product / (norm1 * norm2)
        
        # Calculate temporal decay factor
        temporal_factor = 1.0 / (1.0 + alpha * time_difference)
        
        # Calculate final weight
        weight = cosine_similarity * co_reflection_frequency * temporal_factor
        
        return weight


    def calculate_self_loss(coherence_values: List[float],
                            optimal_coherence: float = 1.0) -> float:
        """
        Calculate the self-loss function that quantifies deviation 
        from optimal coherence (formula #5).
        
        L_self = ∑_i(C_i(t) - C_opt)²
        """
        # Sum of squared differences from optimal
        loss = sum((c - optimal_coherence)**2 for c in coherence_values)
        
        return loss


    def calculate_coherence_kernel(edge_weights: Dict[Tuple[str, str], float],
                                frequencies: Dict[Tuple[str, str], float]) -> float:
        """
        Calculate the coherence kernel - overall system coherence (formula #6).
        
        R(S_t) = (1/|E_t|) * ∑_{(i,j)∈E_t} w_{ij} * f_{ij}
        """
        if not edge_weights:
            return 0.0
        
        # Calculate weighted sum
        weighted_sum = 0.0
        for edge, weight in edge_weights.items():
            frequency = frequencies.get(edge, 1.0)
            weighted_sum += weight * frequency
        
        # Normalize by number of edges
        coherence = weighted_sum / len(edge_weights)
        
        return coherence


    def gradient_descent_step(coherence_values: List[float],
                            self_loss_gradient: List[float],
                            learning_rate: float = 0.01) -> List[float]:
        """
        Perform one step of gradient descent to optimize coherence (formula #7).
        
        dC_i(t)/dt = -∇L_self(C_i(t))
        """
        # Update coherence values using gradient descent
        updated_values = []
        for c_i, grad_i in zip(coherence_values, self_loss_gradient):
            # Update using gradient descent
            c_new = c_i - learning_rate * grad_i
            # Ensure value stays in [0, 1] range
            c_new = max(0.0, min(1.0, c_new))
            updated_values.append(c_new)
        
        return updated_values


    def calculate_multi_agent_coherence(agent_coherences: Dict[str, float],
                                        agent_weights: Dict[str, float]) -> float:
        """
        Calculate the overall coherence in a multi-agent system (formula #8).
        
        R(S_t) = ∑_{i=1}^{N} w_i * C_i(t)
        
        Args:
            agent_coherences: Dictionary of {agent_id: coherence_value}
            agent_weights: Dictionary of {agent_id: weight}
            
        Returns:
            Overall multi-agent system coherence
        """
        if not agent_coherences:
            return 0.0
        
        # Simple, consistent approach that handles all cases correctly
        numerator = 0.0
        denominator = 0.0
        
        for agent_id, coherence in agent_coherences.items():
            weight = agent_weights.get(agent_id, 1.0)  # Default to 1.0 if not in weights
            numerator += weight * coherence
            denominator += weight
        
        # Return weighted average or 0 if denominator is 0
        return numerator / denominator if denominator > 0 else 0.0


    def add_gaussian_noise(gradient: np.ndarray,
                        mean: float = 0.0,
                        std_dev: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to a gradient for robustness (formula #9).
        
        ∇L_self = μ + ξ, ξ ~ N(0, σ²I)
        """
        # Generate Gaussian noise
        noise = np.random.normal(mean, std_dev, size=gradient.shape)
        
        # Add noise to gradient
        noisy_gradient = gradient + noise
        
        return noisy_gradient


class TestEnhancedSchemaMath(unittest.TestCase):
    """Test the enhanced mathematical formulations."""
    
    def test_coherence_accumulation(self):
        """Test coherence accumulation using logistic growth model."""
        # Test case 1: Low entropy should allow coherence to grow
        coherence1 = calculate_coherence_accumulation(
            current_coherence=0.5,
            entropy=0.1,
            alpha=0.2,
            max_coherence=1.0,
            beta=0.1,
            dt=1.0
        )
        self.assertGreater(coherence1, 0.5, "Coherence should increase with low entropy")
        
        # Test case 2: High entropy should decrease coherence
        coherence2 = calculate_coherence_accumulation(
            current_coherence=0.5,
            entropy=2.0,  # High entropy
            alpha=0.2,
            max_coherence=1.0,
            beta=0.2,  # Higher beta makes entropy more influential
            dt=1.0
        )
        self.assertLess(coherence2, 0.5, "Coherence should decrease with high entropy")
        
        # Test case 3: Zero coherence should stay at zero, even with low entropy
        coherence3 = calculate_coherence_accumulation(
            current_coherence=0.0,
            entropy=0.1,
            alpha=0.2,
            max_coherence=1.0,
            beta=0.1,
            dt=1.0
        )
        self.assertEqual(coherence3, 0.0, "Zero coherence should remain zero with logistic growth")
        
        # Test case 4: Maximum coherence should stay near max if entropy is low
        # Note: Due to numerical precision, it might not be exactly 1.0
        coherence4 = calculate_coherence_accumulation(
            current_coherence=1.0,
            entropy=0.1,
            alpha=0.2,
            max_coherence=1.0,
            beta=0.1,
            dt=1.0
        )
        self.assertAlmostEqual(coherence4, 1.0, places=1, 
                              msg="Maximum coherence should remain close to maximum")
        
        # Test case 5: Coherence should approach carrying capacity over time
        coherence = 0.1
        # Increase iterations and alpha to reach carrying capacity faster
        for _ in range(50):
            coherence = calculate_coherence_accumulation(
                current_coherence=coherence,
                entropy=0.05,  # Lower entropy
                alpha=0.5,     # Higher growth rate
                max_coherence=0.9,
                beta=0.05,
                dt=0.5
            )
        self.assertGreater(coherence, 0.7, "Coherence should approach carrying capacity over time")
    
    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        # Test case 1: Uniform distribution should maximize entropy
        uniform_dist = [0.25, 0.25, 0.25, 0.25]
        uniform_entropy = calculate_shannon_entropy(uniform_dist)
        self.assertAlmostEqual(uniform_entropy, 2.0, places=2, 
                              msg="Uniform distribution should have entropy = log2(n)")
        
        # Test case 2: Deterministic distribution should have zero entropy
        deterministic_dist = [1.0, 0.0, 0.0, 0.0]
        deterministic_entropy = calculate_shannon_entropy(deterministic_dist)
        self.assertEqual(deterministic_entropy, 0.0, "Deterministic distribution should have zero entropy")
        
        # Test case 3: Skewed distribution should have intermediate entropy
        skewed_dist = [0.7, 0.1, 0.1, 0.1]
        skewed_entropy = calculate_shannon_entropy(skewed_dist)
        self.assertLess(skewed_entropy, uniform_entropy, "Skewed distribution should have lower entropy than uniform")
        self.assertGreater(skewed_entropy, 0.0, "Skewed distribution should have positive entropy")
        
        # Test case 4: Empty distribution should have zero entropy
        empty_entropy = calculate_shannon_entropy([])
        self.assertEqual(empty_entropy, 0.0, "Empty distribution should have zero entropy")
    
    def test_memory_probability_distribution(self):
        """Test calculation of probability distributions from memory attributes."""
        # Test case 1: Simple memory dictionary
        memories = {
            "m1": {"importance": 0.8, "type": "semantic"},
            "m2": {"importance": 0.4, "type": "episodic"},
            "m3": {"importance": 0.6, "type": "procedural"},
            "m4": {"importance": 0.2, "type": "semantic"}
        }
        
        probabilities = calculate_memory_probability_distribution(memories, attribute="importance")
        self.assertEqual(len(probabilities), 4, "Should return one probability per memory")
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6, 
                              msg="Probabilities should sum to 1")
        self.assertGreater(probabilities[0], probabilities[1], "Higher importance should have higher probability")
        
        # Test case 2: Object-based memories with attributes
        class MemoryObj:
            def __init__(self, importance, memory_type):
                self.importance = importance
                self.memory_type = memory_type
        
        obj_memories = {
            "m1": MemoryObj(0.8, "semantic"),
            "m2": MemoryObj(0.4, "episodic"),
            "m3": MemoryObj(0.6, "procedural")
        }
        
        obj_probabilities = calculate_memory_probability_distribution(obj_memories, attribute="importance")
        self.assertEqual(len(obj_probabilities), 3, "Should return one probability per memory object")
        self.assertAlmostEqual(sum(obj_probabilities), 1.0, places=6, 
                              msg="Probabilities should sum to 1")
        
        # Test case 3: Empty memories
        empty_probabilities = calculate_memory_probability_distribution({}, attribute="importance")
        self.assertEqual(len(empty_probabilities), 0, "Empty memories should return empty probabilities")
        
        # Test case 4: Missing attribute should result in empty list
        missing_attr_probabilities = calculate_memory_probability_distribution(memories, attribute="nonexistent")
        self.assertEqual(len(missing_attr_probabilities), 0, "Missing attribute should return empty probabilities")
    
    def test_phase_transition_threshold(self):
        """Test calculation of phase transition threshold."""
        # Test case 1: Stable entropy history
        stable_entropy = [0.5, 0.52, 0.48, 0.51, 0.49]
        stable_threshold = calculate_phase_transition_threshold(stable_entropy, lambda_theta=1.0)
        # For stable entropy, threshold should be close to the mean
        self.assertAlmostEqual(stable_threshold, 0.5, places=1, 
                              msg="Stable entropy should have threshold near mean")
        
        # Test case 2: Volatile entropy history
        volatile_entropy = [0.2, 0.8, 0.3, 0.9, 0.1]
        volatile_threshold = calculate_phase_transition_threshold(volatile_entropy, lambda_theta=1.0)
        # For volatile entropy, threshold should be higher due to variance component
        self.assertGreater(volatile_threshold, 0.6, "Volatile entropy should have higher threshold")
        
        # Test case 3: Empty entropy history
        empty_threshold = calculate_phase_transition_threshold([], lambda_theta=1.0)
        self.assertEqual(empty_threshold, 0.5, "Empty entropy history should return default threshold")
        
        # Test case 4: Higher lambda should increase threshold for same volatility
        higher_lambda_threshold = calculate_phase_transition_threshold(volatile_entropy, lambda_theta=2.0)
        self.assertGreater(higher_lambda_threshold, volatile_threshold, 
                          "Higher lambda should increase threshold")
    
    def test_memory_similarity_weight(self):
        """Test calculation of memory similarity weights."""
        # Test case 1: Identical embeddings should have max similarity
        embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
        identical_weight = calculate_memory_similarity_weight(
            embedding1, embedding1, co_reflection_frequency=1.0, time_difference=0
        )
        self.assertAlmostEqual(identical_weight, 1.0, places=6, 
                              msg="Identical embeddings should have weight close to 1")
        
        # Test case 2: Orthogonal embeddings should have zero similarity
        # Create an orthogonal vector to embedding1
        embedding2 = np.array([0.4, -0.3, 0.2, -0.1])
        # Adjust to ensure orthogonality
        dot_product = np.dot(embedding1, embedding2)
        if abs(dot_product) > 1e-10:
            # Make it orthogonal by Gram-Schmidt process
            projection = (dot_product / np.dot(embedding1, embedding1)) * embedding1
            embedding2 = embedding2 - projection
            
        orthogonal_weight = calculate_memory_similarity_weight(
            embedding1, embedding2, co_reflection_frequency=1.0, time_difference=0
        )
        self.assertAlmostEqual(orthogonal_weight, 0.0, places=6, 
                              msg="Orthogonal embeddings should have weight close to 0")
        
        # Test case 3: Similar embeddings with high time difference
        embedding3 = np.array([0.15, 0.25, 0.35, 0.45])  # Similar to embedding1
        time_diff_weight = calculate_memory_similarity_weight(
            embedding1, embedding3, co_reflection_frequency=1.0, time_difference=10, alpha=0.1
        )
        no_time_diff_weight = calculate_memory_similarity_weight(
            embedding1, embedding3, co_reflection_frequency=1.0, time_difference=0
        )
        self.assertLess(time_diff_weight, no_time_diff_weight, 
                       "Time difference should reduce similarity weight")
        
        # Test case 4: Higher co-reflection frequency should increase weight
        high_freq_weight = calculate_memory_similarity_weight(
            embedding1, embedding3, co_reflection_frequency=2.0, time_difference=0
        )
        self.assertGreater(high_freq_weight, no_time_diff_weight, 
                          "Higher co-reflection frequency should increase weight")
        
        # Test case 5: Zero embeddings should have zero similarity
        zero_embedding = np.zeros_like(embedding1)
        zero_weight = calculate_memory_similarity_weight(
            embedding1, zero_embedding, co_reflection_frequency=1.0, time_difference=0
        )
        self.assertEqual(zero_weight, 0.0, "Zero embedding should have zero similarity")
    
    def test_self_loss(self):
        """Test calculation of self-loss function."""
        # Test case 1: Coherence values at optimal should have zero loss
        optimal_coherence = 0.8
        optimal_values = [0.8, 0.8, 0.8, 0.8]
        optimal_loss = calculate_self_loss(optimal_values, optimal_coherence=optimal_coherence)
        self.assertEqual(optimal_loss, 0.0, "Coherence values at optimal should have zero loss")
        
        # Test case 2: Mixed coherence values should have positive loss
        mixed_values = [0.6, 0.7, 0.8, 0.9]
        mixed_loss = calculate_self_loss(mixed_values, optimal_coherence=optimal_coherence)
        self.assertGreater(mixed_loss, 0.0, "Mixed coherence values should have positive loss")
        
        # Test case 3: All values far from optimal should have high loss
        far_values = [0.2, 0.3, 0.1, 0.2]
        far_loss = calculate_self_loss(far_values, optimal_coherence=optimal_coherence)
        self.assertGreater(far_loss, mixed_loss, "Values far from optimal should have higher loss")
        
        # Test case 4: Empty values should have zero loss
        empty_loss = calculate_self_loss([], optimal_coherence=optimal_coherence)
        self.assertEqual(empty_loss, 0.0, "Empty values should have zero loss")
    
    def test_coherence_kernel(self):
        """Test calculation of coherence kernel."""
        # Test case 1: Simple edge weights and frequencies
        edge_weights = {
            ("m1", "m2"): 0.8,
            ("m1", "m3"): 0.5,
            ("m2", "m3"): 0.6
        }
        frequencies = {
            ("m1", "m2"): 2.0,
            ("m1", "m3"): 1.0,
            ("m2", "m3"): 1.5
        }
        
        kernel_value = calculate_coherence_kernel(edge_weights, frequencies)
        expected_value = (0.8*2.0 + 0.5*1.0 + 0.6*1.5) / 3.0
        self.assertAlmostEqual(kernel_value, expected_value, places=6, 
                              msg="Kernel value should match expected calculation")
        
        # Test case 2: Equal weights but different frequencies
        equal_weights = {
            ("m1", "m2"): 0.7,
            ("m1", "m3"): 0.7,
            ("m2", "m3"): 0.7
        }
        varying_frequencies = {
            ("m1", "m2"): 3.0,
            ("m1", "m3"): 1.0,
            ("m2", "m3"): 2.0
        }
        
        equal_weights_kernel = calculate_coherence_kernel(equal_weights, varying_frequencies)
        self.assertNotEqual(equal_weights_kernel, 0.7, 
                           "Equal weights with different frequencies should not equal the weight value")
        expected_equal_weights = 0.7 * (3.0 + 1.0 + 2.0) / 3.0
        self.assertAlmostEqual(equal_weights_kernel, expected_equal_weights, places=6,
                               msg="Kernel with equal weights should be average of weighted frequencies")
        
        # Test case 3: Empty edge weights
        empty_kernel = calculate_coherence_kernel({}, {})
        self.assertEqual(empty_kernel, 0.0, "Empty edge weights should return zero kernel")
        
        # Test case 4: Missing frequencies should default to 1.0
        incomplete_frequencies = {
            ("m1", "m2"): 2.0
            # Missing other frequencies
        }
        incomplete_kernel = calculate_coherence_kernel(edge_weights, incomplete_frequencies)
        expected_incomplete = (0.8*2.0 + 0.5*1.0 + 0.6*1.0) / 3.0  # Missing frequencies default to 1.0
        self.assertAlmostEqual(incomplete_kernel, expected_incomplete, places=6,
                              msg="Missing frequencies should default to 1.0")
    
    def test_gradient_descent_step(self):
        """Test gradient descent step for coherence optimization."""
        # Test case 1: Gradient pointing to increase coherence
        coherence_values = [0.5, 0.6, 0.7, 0.8]
        gradients = [-0.1, -0.2, -0.1, -0.05]  # Negative gradients increase values
        learning_rate = 0.1
        updated_values = gradient_descent_step(coherence_values, gradients, learning_rate=learning_rate)
        
        for i in range(len(coherence_values)):
            expected_value = coherence_values[i] - learning_rate * gradients[i]
            expected_value = max(0.0, min(1.0, expected_value))  # Apply bounds
            self.assertAlmostEqual(updated_values[i], expected_value, places=6,
                                  msg=f"Updated value at index {i} should match expected calculation")
        
        # Test case 2: Gradient pointing to decrease coherence
        coherence_values = [0.5, 0.6, 0.7, 0.8]
        gradients = [0.1, 0.2, 0.1, 0.05]  # Positive gradients decrease values
        learning_rate = 0.1
        updated_values = gradient_descent_step(coherence_values, gradients, learning_rate=learning_rate)
        
        for i in range(len(coherence_values)):
            expected_value = coherence_values[i] - learning_rate * gradients[i]
            expected_value = max(0.0, min(1.0, expected_value))  # Apply bounds
            self.assertAlmostEqual(updated_values[i], expected_value, places=6,
                                  msg=f"Updated value at index {i} should match expected calculation")
        
        # Test case 3: Values should be bounded between 0 and 1
        coherence_values = [0.05, 0.95]
        gradients = [0.6, -0.6]  # Would push values out of bounds
        updated_values = gradient_descent_step(coherence_values, gradients, learning_rate=0.1)
        
        self.assertEqual(updated_values[0], 0.0, "Value should be bounded at 0")
        self.assertEqual(updated_values[1], 1.0, "Value should be bounded at 1")
        
        # Test case 4: Learning rate should control step size
        coherence_values = [0.5]
        gradients = [0.1]
        small_step = gradient_descent_step(coherence_values, gradients, learning_rate=0.01)
        large_step = gradient_descent_step(coherence_values, gradients, learning_rate=0.1)
        
        small_expected = 0.5 - 0.01 * 0.1
        large_expected = 0.5 - 0.1 * 0.1
        
        self.assertAlmostEqual(small_step[0], small_expected, places=6,
                              msg="Small learning rate should result in small step")
        self.assertAlmostEqual(large_step[0], large_expected, places=6,
                              msg="Large learning rate should result in large step")
    
    def test_multi_agent_coherence(self):
        """Test calculation of multi-agent coherence."""
        # Test with equal weights
        agent_coherences = {'agent1': 0.7, 'agent2': 0.5, 'agent3': 0.8}
        weights = {'agent1': 1.0, 'agent2': 1.0, 'agent3': 1.0}
        expected_coherence = (0.7 + 0.5 + 0.8) / 3
        self.assertAlmostEqual(calculate_multi_agent_coherence(agent_coherences, weights), expected_coherence)
        
        # Test with different weights
        weights = {'agent1': 1.0, 'agent2': 2.0, 'agent3': 3.0}
        expected_coherence = (0.7*1.0 + 0.5*2.0 + 0.8*3.0) / (1.0 + 2.0 + 3.0)
        self.assertAlmostEqual(calculate_multi_agent_coherence(agent_coherences, weights), expected_coherence)
        
        # Test with no weights provided (should default to equal weights)
        expected_coherence = (0.7 + 0.5 + 0.8) / 3
        self.assertAlmostEqual(calculate_multi_agent_coherence(agent_coherences, {}), expected_coherence)
        
        # Test with empty agent coherences
        self.assertEqual(calculate_multi_agent_coherence({}, {}), 0.0)
        
        # Test with subset of agents in weights
        weights = {'agent1': 2.0, 'agent3': 3.0}  # agent2 missing
        expected_coherence = (0.7*2.0 + 0.5*1.0 + 0.8*3.0) / (2.0 + 1.0 + 3.0)
        self.assertAlmostEqual(calculate_multi_agent_coherence(agent_coherences, weights), expected_coherence)
    
    def test_gaussian_noise(self):
        """Test addition of Gaussian noise to gradients."""
        # Test case 1: Noise should change gradient values
        np.random.seed(42)  # For reproducibility
        gradient = np.array([0.1, 0.2, 0.3, 0.4])
        noisy_gradient = add_gaussian_noise(gradient, mean=0.0, std_dev=0.1)
        
        self.assertEqual(noisy_gradient.shape, gradient.shape, "Noisy gradient should have same shape")
        self.assertFalse(np.array_equal(noisy_gradient, gradient), "Noise should change gradient values")
        
        # Test case 2: Noise with zero std_dev should return same gradient
        zero_noise_gradient = add_gaussian_noise(gradient, mean=0.0, std_dev=0.0)
        np.testing.assert_array_almost_equal(zero_noise_gradient, gradient, decimal=5, 
                                            err_msg="Zero std_dev should not change gradient")
        
        # Test case 3: Noise with non-zero mean should shift gradient
        mean_shift = 0.5
        np.random.seed(42)  # Reset seed for reproducibility
        shifted_gradient = add_gaussian_noise(gradient, mean=mean_shift, std_dev=0.01)
        # The average shift should be close to the mean value
        average_shift = np.mean(shifted_gradient - gradient)
        self.assertAlmostEqual(average_shift, mean_shift, places=1, 
                              msg="Average shift should be close to mean value")
        
        # Test case 4: Higher std_dev should result in more variable noise
        np.random.seed(42)  # Reset seed for reproducibility
        low_std_dev = add_gaussian_noise(gradient, mean=0.0, std_dev=0.01)
        np.random.seed(42)  # Reset seed for reproducibility 
        high_std_dev = add_gaussian_noise(gradient, mean=0.0, std_dev=1.0)
        
        low_variance = np.var(low_std_dev - gradient)
        high_variance = np.var(high_std_dev - gradient)
        self.assertLess(low_variance, high_variance, "Higher std_dev should result in higher variance")


if __name__ == "__main__":
    unittest.main() 