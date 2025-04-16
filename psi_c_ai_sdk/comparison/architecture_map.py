# psi_c_ai_sdk/comparison/architecture_map.py

"""
Compares ΨC system modules to prior cognitive architectures and highlights architectural divergence.
"""

architecture_comparison = [
    {
        "system": "ACT-R",
        "similarity": ["Declarative memory", "Procedural control", "Chunking"],
        "divergence": [
            "ΨC integrates coherence scoring per memory, not just activation strength",
            "Reflection-based schema mutation is absent in ACT-R",
            "ΨC supports introspective trace propagation and ethical alignment vectors"
        ]
    },
    {
        "system": "SOAR",
        "similarity": ["Goal-driven behavior", "Working memory", "Learning from impasses"],
        "divergence": [
            "ΨC has continuous coherence monitoring across memory graph",
            "ΨC implements schema annealing with entropy penalties",
            "Ethical alignment is core to ΨC decision-making"
        ]
    },
    {
        "system": "Global Workspace Theory (GWT)",
        "similarity": ["Centralized broadcast mechanism", "Attention routing"],
        "divergence": [
            "ΨC replaces attention with weighted coherence across schema nodes",
            "No explicit 'workspace'; ΨC relies on reflective cycles and entropy metrics",
            "ΨC integrates recursive meta-models with bounded depth"
        ]
    },
    {
        "system": "IIT (Integrated Information Theory)",
        "similarity": ["Focus on internal informational structure"],
        "divergence": [
            "ΨC is operational and implementable in live agents",
            "ΨC tracks schema mutation, contradiction, entropy, identity evolution",
            "IIT’s Φ is static; ΨC’s Ψ-index adapts dynamically over time"
        ]
    }
]

def list_divergences():
    print("\nΨC Architectural Divergence Report:\n")
    for arch in architecture_comparison:
        print(f"Compared to {arch['system']}:")
        for div in arch["divergence"]:
            print(f"  - {div}")
        print()

if __name__ == "__main__":
    list_divergences()
