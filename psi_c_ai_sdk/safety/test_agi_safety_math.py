import unittest
import numpy as np
from psi_c_ai_sdk.safety import agi_safety_math as agi

class TestAGISafetyMath(unittest.TestCase):

    def test_recursive_saturation_detects_instability(self):
        R_values = [0.1, 0.3, 0.6, 1.1]
        dt = 1.0
        result = agi.recursive_saturation(R_values, dt, threshold=0.25)
        self.assertTrue(result)

    def test_recursive_saturation_stable(self):
        R_values = [0.1, 0.15, 0.18, 0.2]
        dt = 1.0
        result = agi.recursive_saturation(R_values, dt, threshold=0.25)
        self.assertFalse(result)

    def test_ontology_drift_detection(self):
        sigma_t = np.array([0.1, 0.2, 0.3])
        sigma_prev = np.array([0.5, 0.5, 0.5])
        result = agi.ontology_drift(sigma_t, sigma_prev, threshold=0.4)
        self.assertTrue(result)

    def test_ontology_drift_below_threshold(self):
        sigma_t = np.array([0.1, 0.2, 0.3])
        sigma_prev = np.array([0.1, 0.2, 0.3])
        result = agi.ontology_drift(sigma_t, sigma_prev)
        self.assertFalse(result)

    def test_trust_dampening(self):
        T_k = 1.0
        coherence_rate = 1.5
        adjusted = agi.trust_dampening(T_k, coherence_rate, lambda_persuade=0.5)
        self.assertLess(adjusted, T_k)

    def test_psi_variance(self):
        psi_values = [0.9, 0.92, 0.88, 0.91]
        sigma = agi.psi_variance(psi_values)
        self.assertTrue(0 < sigma < 0.05)

    def test_agi_boundary_distance(self):
        G_self = np.array([1, 0, 0])
        G_input = np.array([0, 1, 0])
        distance = agi.agi_boundary_distance(G_self, G_input)
        self.assertAlmostEqual(distance, 1.0, places=5)

    def test_meta_coherence_variance(self):
        matrix = np.array([[0.8, 0.85], [0.9, 0.95]])
        result = agi.meta_coherence_variance(matrix)
        self.assertTrue(0 <= result < 0.01)

    def test_antagonistic_perturbation(self):
        mems = [np.array([0.5, 0.5]), np.array([1, 0])]
        mods = [np.array([-0.5, -0.5]), np.array([0, 1])]
        score = agi.antagonistic_perturbation(mems, mods)
        self.assertGreater(score, 0.5)

if __name__ == "__main__":
    unittest.main()
