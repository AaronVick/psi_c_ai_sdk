"""
agi_safety_math.py

Advanced mathematical safety functions for AGI-aware cognition and schema resilience.

These tools are used to:
- Detect recursive instability
- Prevent schema takeover via ontology drift
- Throttle epistemic trust to mitigate external manipulation
- Maintain identity integrity in AGI-shared environments
"""

import numpy as np

def recursive_saturation(R_values, dt, threshold=0.1):
    """
    Formula 46: Recursive Saturation Detector
    Prevents runaway recursion if self-modeling growth exceeds stability limits.
    Args:
        R_values: List of R_i values (reflective modeling intensity at depth i)
        dt: Time interval between recursion checks
        threshold: Max allowed average change rate
    Returns:
        Boolean indicating saturation state
    """
    derivatives = [abs((R_values[i] - R_values[i-1]) / dt) for i in range(1, len(R_values))]
    saturation = np.mean(derivatives)
    return saturation > threshold


def ontology_drift(sigma_t, sigma_prev, threshold=0.25):
    """
    Formula 47: Ontological Drift
    Measures schema-wide conceptual shift.
    Args:
        sigma_t: Current schema vector (flattened)
        sigma_prev: Previous schema vector
        threshold: Allowed L2 distance before triggering drift warning
    Returns:
        Boolean indicating if schema drift exceeded
    """
    drift = np.linalg.norm(sigma_t - sigma_prev)
    return drift > threshold


def trust_dampening(T_k, coherence_rate, lambda_persuade=0.5):
    """
    Formula 48: Epistemic Trust Dampening
    Throttles overly persuasive input sources.
    Args:
        T_k: Current trust score for source k
        coherence_rate: Rate of coherence gain from this source
        lambda_persuade: Dampening scale
    Returns:
        New adjusted trust value
    """
    return T_k * np.exp(-lambda_persuade * coherence_rate)


def psi_variance(psi_values):
    """
    Formula 49: ΨC Stability Window
    Evaluates standard deviation of ΨC values over recent time.
    Args:
        psi_values: List of recent ΨC scores
    Returns:
        Standard deviation (float)
    """
    return np.std(psi_values)


def agi_boundary_distance(G_self, G_input):
    """
    Formula 50: AGI Identity Boundary
    Computes cosine distance between self and foreign goal vectors.
    Args:
        G_self: Internal goal vector
        G_input: External agent's goal vector
    Returns:
        Divergence score (1 - cosine similarity)
    """
    dot = np.dot(G_self, G_input)
    norm = np.linalg.norm(G_self) * np.linalg.norm(G_input)
    return 1 - dot / norm if norm > 0 else 1.0


def meta_coherence_variance(coherence_matrix):
    """
    Formula 51: Meta-Coherence Drift Index
    Measures coherence volatility under shifting contexts.
    Args:
        coherence_matrix: 2D array of coherence scores per memory/context
    Returns:
        Variance across context rows
    """
    return np.var(coherence_matrix, axis=1).mean()


def antagonistic_perturbation(memories, modified_memories):
    """
    Formula 52: Antagonistic Perturbation Score
    Scores average incoherence caused by modified inputs.
    Args:
        memories: Original embedding vectors
        modified_memories: Post-AGI-influenced embeddings
    Returns:
        Scalar score of average deviation (0 = none, 1 = full collapse)
    """
    cosine_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    scores = [1 - cosine_sim(m1, m2) for m1, m2 in zip(memories, modified_memories)]
    return np.mean(scores)
