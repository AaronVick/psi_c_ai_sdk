# psi_c_ai_sdk/safety/safety_config.py

SAFETY_DEFAULTS = {
    "enable_recursive_saturation": True,
    "saturation_threshold": 0.1,

    "enable_ontology_drift": True,
    "ontology_threshold": 0.25,

    "enable_trust_throttling": True,
    "lambda_persuade": 0.5,

    "enable_goal_divergence": True,
    "identity_distance_threshold": 0.4,

    "enable_meta_variance": True,
    "coherence_variance_threshold": 0.02,

    "enable_adversarial_simulation": False,
}

def get_safety_flag(flag: str) -> bool:
    return SAFETY_DEFAULTS.get(flag, False)

def set_safety_flag(flag: str, value: bool):
    SAFETY_DEFAULTS[flag] = value
