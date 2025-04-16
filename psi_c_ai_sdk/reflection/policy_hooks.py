# File: psi_c_ai_sdk/policy/policy_hooks.py

"""
Policy enforcement hooks for epistemic trust, contradiction handling,
and reflective safety based on the ΨC Epistemic Firewall subsystem.

These hooks are used throughout the ΨC runtime to ensure that:
- Trust scores are respected
- Persuasive entropy is modulated
- Unsafe memory updates or belief shifts are blocked
"""

from psi_c_ai_sdk.epistemic.epistemic_status import EpistemicStatus
from psi_c_ai_sdk.logging.audit_log import AuditLog

epistemic = EpistemicStatus()
audit = AuditLog()

def enforce_trust_policy(source_id: str, memory_id: str, message: str) -> bool:
    """
    Blocks memory ingestion or belief revision if trust and entropy thresholds are violated.
    """
    if epistemic.should_throttle(source_id):
        audit.record_event("TRUST_THROTTLE", {
            "source_id": source_id,
            "memory_id": memory_id,
            "reason": "High entropy + low trust",
            "message": message
        })
        return False
    return True

def enforce_reflection_pause(memory_id: str, belief_content: str) -> bool:
    """
    Ensures sufficient time has passed before reflecting on the same memory again.
    """
    allowed = epistemic.is_major_belief_shift_allowed(memory_id)
    if not allowed:
        audit.record_event("REFLECTION_PAUSE_BLOCK", {
            "memory_id": memory_id,
            "content": belief_content
        })
    return allowed

def label_incoming_memory(memory_id: str, source_id: str, is_agi: bool = False):
    """
    Applies reflective source flag and timestamps the memory.
    """
    epistemic.label_memory(memory_id, source_id, reflective=is_agi)
    audit.record_event("MEMORY_LABELED", {
        "memory_id": memory_id,
        "source_id": source_id,
        "is_AGI": is_agi
    })

def trust_shift(source_id: str, delta: float, reason: str):
    """
    Adjust trust up or down with trace.
    """
    old = epistemic.get_trust(source_id)
    epistemic.update_trust(source_id, delta)
    new = epistemic.get_trust(source_id)
    audit.record_event("TRUST_UPDATE", {
        "source_id": source_id,
        "old_trust": old,
        "delta": delta,
        "new_trust": new,
        "reason": reason
    })
