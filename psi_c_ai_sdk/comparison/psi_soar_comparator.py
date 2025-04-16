# psi_c_ai_sdk/comparison/psi_soar_comparator.py

"""
Compares ΨC-AI agent coherence metrics and reflection outputs against SOAR architecture behavior.
Focuses on decision cycles, working memory transitions, and goal stack handling.
"""

from typing import Dict, Any
import numpy as np


def compare_psi_to_soar_metrics(psi_stats: Dict[str, Any], soar_stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Compare coherence and memory usage characteristics between ΨC and SOAR agents.

    Parameters:
        psi_stats: Dict containing ΨC metrics (e.g., coherence, entropy, reflection_depth)
        soar_stats: Dict containing SOAR metrics (e.g., decision_cycles, working_memory_size)

    Returns:
        Dict[str, float]: Comparison results including normalized difference values.
    """
    results = {}

    # Normalize and compare reflection cycles to decision cycles
    if "reflection_depth" in psi_stats and "decision_cycles" in soar_stats:
        results["cycle_ratio"] = psi_stats["reflection_depth"] / max(1, soar_stats["decision_cycles"])

    # Coherence vs working memory change rate
    if "coherence" in psi_stats and "working_memory_delta" in soar_stats:
        results["coherence_drift"] = psi_stats["coherence"] - soar_stats["working_memory_delta"]

    # Entropy divergence
    if "entropy" in psi_stats and "entropy" in soar_stats:
        results["entropy_diff"] = psi_stats["entropy"] - soar_stats["entropy"]

    # Schema size vs goal stack size
    if "schema_size" in psi_stats and "goal_stack_size" in soar_stats:
        results["schema_stack_ratio"] = psi_stats["schema_size"] / max(1, soar_stats["goal_stack_size"])

    return results
