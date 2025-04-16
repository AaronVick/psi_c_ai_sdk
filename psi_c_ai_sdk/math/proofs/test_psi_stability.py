import pytest
import numpy as np
from psi_c_ai_sdk.math.proofs.psi_stability import psi_coherence_integral

def test_stable_psi_state():
    R = np.linspace(0.8, 1.0, 10)
    I = np.ones(10) * 0.9
    B = np.ones(10) * 0.95
    H = np.linspace(0.05, 0.1, 10)
    theta = 6.0

    result = psi_coherence_integral(R, I, B, H, theta)
    assert result["ΨC"] > 0.85
    assert result["variance"] < 0.05
    assert result["is_stable"] is True

def test_unstable_due_to_entropy_spike():
    R = np.linspace(0.9, 1.0, 10)
    I = np.ones(10) * 0.95
    B = np.ones(10) * 0.98
    H = np.array([0.05, 0.08, 0.12, 0.05, 0.2, 0.08, 0.1, 0.15, 0.07, 0.05])
    theta = 6.0

    result = psi_coherence_integral(R, I, B, H, theta)
    assert result["ΨC"] < 0.95
    assert result["variance"] >= 0.05
    assert result["is_stable"] is False

def test_unstable_due_to_reflective_instability():
    R = np.array([0.9, 0.4, 0.95, 0.3, 1.0, 0.35, 0.9, 0.6, 0.5, 0.85])
    I = np.ones(10) * 0.9
    B = np.ones(10) * 0.95
    H = np.ones(10) * 0.07
    theta = 6.0

    result = psi_coherence_integral(R, I, B, H, theta)
    assert result["ΨC"] > 0.5
    assert result["is_stable"] is False

def test_edge_case_zero_variance():
    R = np.ones(10) * 0.95
    I = np.ones(10) * 0.95
    B = np.ones(10) * 0.95
    H = np.ones(10) * 0.05
    theta = 6.0

    result = psi_coherence_integral(R, I, B, H, theta)
    assert result["variance"] == 0
    assert result["is_stable"] is True
