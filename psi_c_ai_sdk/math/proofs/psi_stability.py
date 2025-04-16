"""
psi_stability.py

Proof sketch for ΨC Stability Lemma (informal Pythonic version).

Goal:
Show that if recursive depth and coherence variance are bounded,
ΨC(S, t) cannot exceed oscillation threshold ε unless schema contradiction or entropy spike is observed.
"""

import numpy as np

def psi_coherence_integral(R_vals, I_vals, B_vals, H_vals, theta, epsilon=0.05):
    """
    Simulates the core ΨC formula (bounded over T steps).
    Returns ΨC value and whether it's considered stable.
    """
    T = len(R_vals)
    psi_values = []

    for t in range(T):
        term = R_vals[t] * I_vals[t] * B_vals[t] * (1 - H_vals[t])
        psi_values.append(term)

    psi_integral = np.sum(psi_values)
    psi_score = 1 / (1 + np.exp(-(psi_integral - theta)))

    # Calculate oscillation (variance of ΨC across window)
    psi_var = np.var(psi_values)

    is_stable = psi_var < epsilon

    return {
        "ΨC": psi_score,
        "variance": psi_var,
        "is_stable": is_stable
    }

def test_psi_stability():
    """Run a bounded stability check simulation."""

    # Example bounded input values
    R = np.linspace(0.8, 1.0, 10)
    I = np.ones(10) * 0.9
    B = np.ones(10) * 0.95
    H = np.linspace(0.05, 0.1, 10)
    theta = 6.0

    result = psi_coherence_integral(R, I, B, H, theta)

    print("ΨC Score:", round(result["ΨC"], 4))
    print("Variance:", round(result["variance"], 6))
    print("Stable:", result["is_stable"])

if __name__ == "__main__":
    test_psi_stability()
