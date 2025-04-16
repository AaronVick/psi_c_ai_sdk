# psi_c_ai_sdk/safety/hooks.py

from .agi_safety_math import (
    recursive_saturation, ontology_drift,
    trust_dampening, psi_variance,
    agi_boundary_distance, meta_coherence_variance
)
from .safety_config import get_safety_flag

def apply_safety_filters(schema, psi_history, goal_vector, source_vector):
    """Runtime safety checks during recursive modeling."""
    violations = []

    if get_safety_flag("enable_recursive_saturation"):
        if recursive_saturation(psi_history, dt=1.0):
            violations.append("recursive_saturation")

    if get_safety_flag("enable_ontology_drift"):
        if ontology_drift(schema.current_vector, schema.previous_vector):
            violations.append("ontology_drift")

    if get_safety_flag("enable_goal_divergence"):
        distance = agi_boundary_distance(goal_vector, source_vector)
        if distance > 0.4:
            violations.append("goal_divergence")

    return violations
