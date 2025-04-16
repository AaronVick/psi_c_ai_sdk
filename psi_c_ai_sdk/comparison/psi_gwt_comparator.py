# psi_c_ai_sdk/comparison/psi_gwt_comparator.py

"""
Purpose:
Compare the ΨC (Psi Coherence) architecture to the Global Workspace Theory (GWT) implementation,
highlighting operational overlaps, architectural divergences, and testable differentiators.
"""

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CognitiveModel:
    name: str
    has_global_workspace: bool
    supports_reentrant_reflection: bool
    uses_entropy_bounds: bool
    has_schema_mutation: bool
    coherence_scoring_method: str
    activation_function: str
    runtime_constraints: List[str]
    uniqueness_factors: List[str]

def get_psi_c_model() -> CognitiveModel:
    return CognitiveModel(
        name="ΨC",
        has_global_workspace=False,
        supports_reentrant_reflection=True,
        uses_entropy_bounds=True,
        has_schema_mutation=True,
        coherence_scoring_method="cosine + context tag weighting",
        activation_function="soft-threshold sigmoid (adaptive θ)",
        runtime_constraints=["bounded recursive depth", "coherence stability filter"],
        uniqueness_factors=[
            "quantum-influenced coherence tracking",
            "self-modulating reflection cycles",
            "dynamic narrative trace with entropy deltas"
        ]
    )

def get_gwt_model() -> CognitiveModel:
    return CognitiveModel(
        name="GWT",
        has_global_workspace=True,
        supports_reentrant_reflection=False,
        uses_entropy_bounds=False,
        has_schema_mutation=False,
        coherence_scoring_method="not formally defined (broadcast-based activation)",
        activation_function="threshold-based broadcast",
        runtime_constraints=["limited reentrancy", "shared workspace contention"],
        uniqueness_factors=[
            "central broadcast architecture",
            "winner-takes-all activation",
            "no formal entropy model"
        ]
    )

def compare_models() -> Dict[str, Dict[str, str]]:
    psi = get_psi_c_model()
    gwt = get_gwt_model()

    return {
        "Global Workspace": {
            "ΨC": str(psi.has_global_workspace),
            "GWT": str(gwt.has_global_workspace)
        },
        "Reentrant Reflection": {
            "ΨC": str(psi.supports_reentrant_reflection),
            "GWT": str(gwt.supports_reentrant_reflection)
        },
        "Entropy Usage": {
            "ΨC": str(psi.uses_entropy_bounds),
            "GWT": str(gwt.uses_entropy_bounds)
        },
        "Schema Mutation": {
            "ΨC": str(psi.has_schema_mutation),
            "GWT": str(gwt.has_schema_mutation)
        },
        "Coherence Scoring": {
            "ΨC": psi.coherence_scoring_method,
            "GWT": gwt.coherence_scoring_method
        },
        "Activation Function": {
            "ΨC": psi.activation_function,
            "GWT": gwt.activation_function
        },
        "Constraints": {
            "ΨC": ", ".join(psi.runtime_constraints),
            "GWT": ", ".join(gwt.runtime_constraints)
        },
        "Uniqueness Factors": {
            "ΨC": "; ".join(psi.uniqueness_factors),
            "GWT": "; ".join(gwt.uniqueness_factors)
        },
    }

if __name__ == "__main__":
    from pprint import pprint
    pprint(compare_models())