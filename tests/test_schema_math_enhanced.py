#!/usr/bin/env python3
"""
Test the enhanced mathematical formulations in schema_math.py.

This test suite validates the correctness of the newly added mathematical
functions for coherence dynamics, entropy modeling, and other advanced
schema operations.
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Updated import path to match the actual structure
    from tools.dev_environment.schema_math import (
        # New enhanced functions
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
    SCHEMA_MATH_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    SCHEMA_MATH_AVAILABLE = False


@unittest.skipIf(not SCHEMA_MATH_AVAILABLE, "Enhanced schema math functions not available")
class TestEnhancedSchemaMath(unittest.TestCase):
    """Test the enhanced mathematical formulations in schema_math.py."""
    
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
        
        # Test case 3: Zero coherence should still grow if entropy is low
        coherence3 = calculate_coherence_accumulation(
            current_coherence=0.0,
            entropy=0.1,
            alpha=0.2,
            max_coherence=1.0,
            beta=0.1,
            dt=1.0
        )
        self.assertEqual(coherence3, 0.0, "Zero coherence should remain zero (can't grow from nothing)")
        
        # Test case 4: Maximum coherence should stay at max if entropy is low
        coherence4 = calculate_coherence_accumulation(
            current_coherence=1.0,
            entropy=0.1,
            alpha=0.2,
            max_coherence=1.0,
            beta=0.1,
            dt=1.0
        )
        self.assertEqual(coherence4, 1.0, "Maximum coherence should remain at maximum")
        
        # Test case 5: Coherence should approach carrying capacity over time
        coherence = 0.1
        for _ in range(20):
            coherence = calculate_coherence_accumulation(
                current_coherence=coherence,
                entropy=0.1,
                alpha=0.2,
                max_coherence=0.9,  # Lower carrying capacity
                beta=0.05,
                dt=0.5
            )
        self.assertGreater(coherence, 0.7, "Coherence should approach carrying capacity over time")
    
    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        # Test case 1: Uniform distribution should maximize entropy
        uniform_dist = [0.25, 0.25, 0.25, 0.25]
        uniform_entropy = calculate_shannon_entropy(uniform_dist)
        self.assertAlmostEqual(uniform_entropy, 2.0, msg="Uniform distribution should have entropy = log2(n)", delta=0.01)
        
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
        self.assertAlmostEqual(sum(probabilities), 1.0, msg="Probabilities should sum to 1", delta=0.01)
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
        self.assertAlmostEqual(sum(obj_probabilities), 1.0, msg="Probabilities should sum to 1", delta=0.01)
        
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
        self.assertAlmostEqual(stable_threshold, 0.5, msg="Stable entropy should have threshold near mean", delta=0.1)
        
        # Test case 2: Volatile entropy history
        volatile_entropy = [0.2, 0.8, 0.3, 0.9, 0.1]
        volatile_threshold = calculate_phase_transition_threshold(volatile_entropy, lambda_theta=1.0)
        self.assertGreater(volatile_threshold, 0.8, "Volatile entropy should have higher threshold")
        
        # Test case 3: Empty entropy history
        empty_threshold = calculate_phase_transition_threshold([], lambda_theta=1.0)
        self.assertEqual(empty_threshold, 0.5, "Empty entropy history should return default threshold")
        
        # Test case 4: Higher lambda should increase threshold for same volatility
        higher_lambda_threshold = calculate_phase_transition_threshold(volatile_entropy, lambda_theta=2.0)
        self.assertGreater(higher_lambda_threshold, volatile_threshold, "Higher lambda should increase threshold")
    
    def test_memory_similarity_weight(self):
        """Test calculation of memory similarity weights."""
        # Test case 1: Identical embeddings should have max similarity
        embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
        identical_weight = calculate_memory_similarity_weight(
            embedding1, embedding1, co_reflection_frequency=1.0, time_difference=0
        )
        self.assertAlmostEqual(identical_weight, 1.0, msg="Identical embeddings should have weight close to 1", delta=0.01)
        
        # Test case 2: Orthogonal embeddings should have zero similarity
        embedding2 = np.array([0.4, -0.3, 0.2, -0.1])  # Orthogonal to embedding1
        orthogonal_weight = calculate_memory_similarity_weight(
            embedding1, embedding2, co_reflection_frequency=1.0, time_difference=0
        )
        self.assertAlmostEqual(orthogonal_weight, 0.0, msg="Orthogonal embeddings should have weight close to 0", delta=0.01)
        
        # Test case 3: Similar embeddings with high time difference
        embedding3 = np.array([0.15, 0.25, 0.35, 0.45])  # Similar to embedding1
        time_diff_weight = calculate_memory_similarity_weight(
            embedding1, embedding3, co_reflection_frequency=1.0, time_difference=10, alpha=0.1
        )
        no_time_diff_weight = calculate_memory_similarity_weight(
            embedding1, embedding3, co_reflection_frequency=1.0, time_difference=0
        )
        self.assertLess(time_diff_weight, no_time_diff_weight, "Time difference should reduce similarity weight")
        
        # Test case 4: Higher co-reflection frequency should increase weight
        high_freq_weight = calculate_memory_similarity_weight(
            embedding1, embedding3, co_reflection_frequency=2.0, time_difference=0
        )
        self.assertGreater(high_freq_weight, no_time_diff_weight, "Higher co-reflection frequency should increase weight")
        
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
        self.assertAlmostEqual(kernel_value, expected_value, msg="Kernel value should match expected calculation", delta=0.01)
        
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
        self.assertNotEqual(equal_weights_kernel, 0.7, "Equal weights with different frequencies should not equal the weight value")
        
        # Test case 3: Empty edge weights
        empty_kernel = calculate_coherence_kernel({}, {})
        self.assertEqual(empty_kernel, 0.0, "Empty edge weights should return zero kernel")
        
        # Test case 4: Missing frequencies should default to 1.0
        incomplete_frequencies = {
            ("m1", "m2"): 2.0
            # Missing other frequencies
        }
        incomplete_kernel = calculate_coherence_kernel(edge_weights, incomplete_frequencies)
        self.assertGreater(incomplete_kernel, 0.0, "Kernel should use default frequency of 1.0 for missing edges")
    
    def test_gradient_descent_step(self):
        """Test gradient descent step for coherence optimization."""
        # Test case 1: Gradient pointing to increase coherence
        coherence_values = [0.5, 0.6, 0.7, 0.8]
        gradients = [-0.1, -0.2, -0.1, -0.05]  # Negative gradients increase values
        updated_values = gradient_descent_step(coherence_values, gradients, learning_rate=0.1)
        
        for i in range(len(coherence_values)):
            self.assertGreater(updated_values[i], coherence_values[i], "Negative gradient should increase coherence")
        
        # Test case 2: Gradient pointing to decrease coherence
        coherence_values = [0.5, 0.6, 0.7, 0.8]
        gradients = [0.1, 0.2, 0.1, 0.05]  # Positive gradients decrease values
        updated_values = gradient_descent_step(coherence_values, gradients, learning_rate=0.1)
        
        for i in range(len(coherence_values)):
            self.assertLess(updated_values[i], coherence_values[i], "Positive gradient should decrease coherence")
        
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
        
        small_diff = coherence_values[0] - small_step[0]
        large_diff = coherence_values[0] - large_step[0]
        self.assertLess(small_diff, large_diff, "Larger learning rate should cause larger step")
    
    def test_multi_agent_coherence(self):
        """Test calculation of multi-agent coherence."""
        # Test case 1: Equal weights
        agent_coherences = {
            "agent1": 0.8,
            "agent2": 0.6,
            "agent3": 0.7
        }
        equal_weights = {
            "agent1": 1.0,
            "agent2": 1.0,
            "agent3": 1.0
        }
        
        equal_coherence = calculate_multi_agent_coherence(agent_coherences, equal_weights)
        expected_equal = (0.8 + 0.6 + 0.7) / 3.0
        self.assertAlmostEqual(equal_coherence, expected_equal, msg="Equal weights should average coherences", delta=0.01)
        
        # Test case 2: Different weights
        diff_weights = {
            "agent1": 2.0,  # More important agent
            "agent2": 0.5,  # Less important agent
            "agent3": 1.0   # Neutral importance
        }
        
        weighted_coherence = calculate_multi_agent_coherence(agent_coherences, diff_weights)
        expected_weighted = (0.8*2.0 + 0.6*0.5 + 0.7*1.0) / (2.0 + 0.5 + 1.0)
        self.assertAlmostEqual(weighted_coherence, expected_weighted, msg="Should calculate weighted average", delta=0.01)
        
        # Test case 3: No weights provided
        no_weights_coherence = calculate_multi_agent_coherence(agent_coherences, {})
        self.assertAlmostEqual(no_weights_coherence, expected_equal, msg="No weights should default to equal weights", delta=0.01)
        
        # Test case 4: Empty agent coherences
        empty_coherence = calculate_multi_agent_coherence({}, diff_weights)
        self.assertEqual(empty_coherence, 0.0, "Empty agent coherences should return zero")
    
    def test_gaussian_noise(self):
        """Test addition of Gaussian noise to gradients."""
        # Test case 1: Noise should change gradient values
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
        shifted_gradient = add_gaussian_noise(gradient, mean=mean_shift, std_dev=0.01)
        # Each element should be approximately shifted by the mean value
        for i in range(len(gradient)):
            self.assertGreater(shifted_gradient[i], gradient[i], "Positive mean should increase gradient values")
        
        # Test case 4: Higher std_dev should result in more variable noise
        low_std_dev = add_gaussian_noise(gradient, mean=0.0, std_dev=0.01)
        high_std_dev = add_gaussian_noise(gradient, mean=0.0, std_dev=1.0)
        
        low_variance = np.var(low_std_dev - gradient)
        high_variance = np.var(high_std_dev - gradient)
        self.assertLess(low_variance, high_variance, "Higher std_dev should result in higher variance")


if __name__ == "__main__":
    unittest.main() 