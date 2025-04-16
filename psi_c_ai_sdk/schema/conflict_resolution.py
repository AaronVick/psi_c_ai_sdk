# psi_c_ai_sdk/schema/conflict_resolution.py

def resolve_belief_conflict(a, b, coherence_fn, trust_fn, entropy_fn) -> Any:
    """
    Decide between two conflicting beliefs a and b.
    """
    score_a = coherence_fn(a) + trust_fn(a) - entropy_fn(a)
    score_b = coherence_fn(b) + trust_fn(b) - entropy_fn(b)
    return a if score_a >= score_b else b
