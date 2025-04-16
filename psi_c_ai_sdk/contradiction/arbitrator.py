# File: psi_c_ai_sdk/contradiction/arbitrator.py

"""
Belief arbitration logic for resolving contradictory memories using
coherence, trust, and entropy metrics from the Epistemic Firewall.

Implements soft rational adjudication to ensure reflective integrity.
"""

from psi_c_ai_sdk.epistemic.epistemic_status import EpistemicStatus
from psi_c_ai_sdk.logging.audit_log import AuditLog

epistemic = EpistemicStatus()
audit = AuditLog()

def resolve_conflict(memory_a: dict, memory_b: dict) -> str:
    """
    Resolves contradiction between two memory items by scoring them
    based on coherence, trustworthiness, and entropy.
    Returns the memory_id of the belief to retain.
    """
    a_id = memory_a["id"]
    b_id = memory_b["id"]
    a_source = memory_a["source"]
    b_source = memory_b["source"]

    trust_a = epistemic.get_trust(a_source)
    trust_b = epistemic.get_trust(b_source)

    entropy_a = epistemic.persuasion_entropy.get(a_source, 0.0)
    entropy_b = epistemic.persuasion_entropy.get(b_source, 0.0)

    coherence_a = memory_a.get("coherence", 0.5)
    coherence_b = memory_b.get("coherence", 0.5)

    score_a = trust_a * 0.5 + coherence_a * 0.3 - entropy_a * 0.2
    score_b = trust_b * 0.5 + coherence_b * 0.3 - entropy_b * 0.2

    retained = a_id if score_a >= score_b else b_id
    discarded = b_id if retained == a_id else a_id

    audit.record_event("CONTRADICTION_RESOLVED", {
        "kept": retained,
        "discarded": discarded,
        "score_a": round(score_a, 4),
        "score_b": round(score_b, 4),
        "trust_a": trust_a,
        "trust_b": trust_b,
        "coherence_a": coherence_a,
        "coherence_b": coherence_b,
        "entropy_a": entropy_a,
        "entropy_b": entropy_b
    })

    return retained
