# psi_c_ai_sdk/comparison/psi_actr_comparator.py

"""
Compare ΨC-AI architecture against ACT-R (Adaptive Control of Thought - Rational)
across key architectural dimensions: memory, modularity, coherence, and learning.
This module identifies convergence and divergence for theoretical positioning.
"""

def compare_memory_models():
    return {
        "ΨC": "Vector-based episodic + graph schema memory with coherence scoring.",
        "ACT-R": "Declarative vs procedural chunks stored in separate modules."
    }

def compare_learning():
    return {
        "ΨC": "Reflection-based schema mutation + contradiction arbitration.",
        "ACT-R": "Production rule compilation, utility-based reinforcement learning."
    }

def compare_reasoning():
    return {
        "ΨC": "Recursive coherence evaluation with entropy and contradiction signals.",
        "ACT-R": "Symbolic production rule firing matched against goals and memory."
    }

def compare_modularity():
    return {
        "ΨC": "Multi-layered modular system: memory, schema, reflection, entropy, alignment.",
        "ACT-R": "Modules for vision, motor, goal, declarative memory, etc., all interfaced by a central production system."
    }

def compare_novelty():
    return {
        "ΨC": "Includes coherence-driven reflective meta-cognition and entropy tracking not native to ACT-R.",
        "ACT-R": "Well-defined cognitive model but lacks real-time contradiction handling and schema evolution."
    }

def run_comparison():
    return {
        "Memory": compare_memory_models(),
        "Learning": compare_learning(),
        "Reasoning": compare_reasoning(),
        "Modularity": compare_modularity(),
        "Novel Features": compare_novelty()
    }
