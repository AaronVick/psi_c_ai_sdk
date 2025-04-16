"""
Unit tests for the Dynamic Threshold functionality.
"""

import unittest
import time
import numpy as np
from psi_c_ai_sdk.psi_c.dynamic_threshold import DynamicThreshold
from psi_c_ai_sdk.psi_c.psi_operator import PsiCOperator
from psi_c_ai_sdk.psi_c.toolkit import PsiToolkit
from psi_c_ai_sdk.memory.memory import MemoryStore, Memory


class TestDynamicThreshold(unittest.TestCase):
    """Test cases for the DynamicThreshold class."""
    
    def test_threshold_initialization(self):
        """Test that the dynamic threshold is initialized with correct parameters."""
        dt = DynamicThreshold(base_threshold=0.7, sensitivity=0.3)
        self.assertEqual(dt.base_threshold, 0.7)
        self.assertEqual(dt.sensitivity, 0.3)
        self.assertEqual(dt.history, [])
    
    def test_record_state(self):
        """Test recording entropy and coherence states."""
        dt = DynamicThreshold()
        dt.record_state(0.5, 0.7)
        self.assertEqual(len(dt.history), 1)
        self.assertEqual(dt.history[0]["entropy"], 0.5)
        self.assertEqual(dt.history[0]["coherence"], 0.7)
    
    def test_dynamic_adjustment(self):
        """Test that threshold adjusts based on entropy and coherence drift."""
        dt = DynamicThreshold(base_threshold=0.7, sensitivity=0.5)
        
        # Start with stable state
        dt.record_state(0.2, 0.8)  # Low entropy, high coherence
        
        # Add increasing entropy, decreasing coherence
        for i in range(5):
            entropy = 0.2 + (i * 0.1)  # Increasing entropy
            coherence = 0.8 - (i * 0.1)  # Decreasing coherence
            dt.record_state(entropy, coherence)
        
        # Calculate the threshold
        threshold = dt.get_current_threshold()
        
        # Threshold should be higher than base due to increasing entropy and decreasing coherence
        self.assertTrue(threshold > dt.base_threshold)
        
        # Get detailed metrics
        metrics = dt.get_drift_metrics()
        self.assertGreater(metrics["entropy_drift_rate"], 0)
        self.assertLess(metrics["coherence_drift_rate"], 0)
        self.assertGreater(metrics["adjustment"], 0)
    
    def test_threshold_bounds(self):
        """Test that threshold respects min and max bounds."""
        dt = DynamicThreshold(
            base_threshold=0.5,
            sensitivity=10.0,  # Very high sensitivity to force threshold to bounds
            min_threshold=0.1,
            max_threshold=0.9
        )
        
        # Add rapidly increasing entropy
        for i in range(10):
            dt.record_state(i * 0.1, 0.5)
        
        # Threshold should hit upper bound
        threshold = dt.get_current_threshold()
        self.assertLessEqual(threshold, 0.9)
        
        # Reset and try with rapidly decreasing entropy
        dt.reset()
        for i in range(10):
            dt.record_state(1.0 - (i * 0.1), 0.5)
        
        # Threshold should hit lower bound
        threshold = dt.get_current_threshold()
        self.assertGreaterEqual(threshold, 0.1)
    
    def test_integration_with_psi_operator(self):
        """Test integration with PsiCOperator."""
        memory_store = MemoryStore()
        
        # Create operator with dynamic threshold
        operator = PsiCOperator(
            memory_store,
            use_dynamic_threshold=True,
            dynamic_threshold_config={
                "base_threshold": 0.6,
                "sensitivity": 0.3
            }
        )
        
        # Verify dynamic threshold was created
        self.assertIsNotNone(operator.dynamic_threshold)
        self.assertEqual(operator.dynamic_threshold.base_threshold, 0.6)
        
        # Add some memories to trigger threshold updates
        for i in range(5):
            memory = Memory(
                content=f"Test memory {i}",
                importance=1.0 - (i * 0.1)  # Decreasing importance
            )
            memory_store.add_memory(memory)
            
            # Force a psi calculation
            operator.calculate_psi_c()
            
            # Small delay to create time difference
            time.sleep(0.1)
        
        # Get metrics and verify threshold is adaptive
        metrics = operator.psi_index()
        self.assertEqual(metrics["threshold_type"], "dynamic")
        self.assertIn("threshold_adjustment", metrics)
    
    def test_toolkit_integration(self):
        """Test integration with PsiToolkit."""
        memory_store = MemoryStore()
        
        # Create operator with dynamic threshold
        operator = PsiCOperator(
            memory_store,
            use_dynamic_threshold=True
        )
        
        # Create toolkit
        toolkit = PsiToolkit(operator)
        
        # Add some memories
        for i in range(3):
            memory = Memory(
                content=f"Test memory {i}",
                importance=1.0
            )
            memory_store.add_memory(memory)
        
        # Get the threshold metrics
        threshold_metrics = toolkit.get_threshold_metrics()
        self.assertEqual(threshold_metrics["type"], "dynamic")
        self.assertTrue(threshold_metrics["is_adaptive"])
        
        # Get coherence health
        health = toolkit.get_coherence_health()
        self.assertEqual(health["threshold"]["type"], "dynamic")
        
        # Test reset function
        toolkit.reset_dynamic_threshold()
        self.assertEqual(len(operator.dynamic_threshold.history), 0)


if __name__ == "__main__":
    unittest.main() 