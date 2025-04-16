# ΨC-AI SDK Advanced Development Environment

The Advanced Development Environment provides specialized tools for ΨC agent development, testing, introspection, and debugging. These tools are designed to help researchers and developers visualize, manipulate, and analyze ΨC agents during development and experimentation.

## Available Tools

### 1. Visual Schema Editor (`schema_editor.py`)

An interactive visualization and editing tool for ΨC agent schema graphs. Features include:
- Real-time schema graph visualization
- Node and edge editing capabilities
- Visual diff between schema versions
- Schema integrity validation
- Export to various formats (graphviz, JSON, etc.)

### 2. Memory Manipulation Sandbox (`memory_sandbox.py`)

A safe environment for experimenting with memory manipulation operations:
- Memory creation and injection
- Memory editing and deletion
- Memory tag management
- Embedding visualization
- Coherence impact analysis

### 3. Reflection Cycle Debugger (`reflection_debugger.py`)

Tools for debugging and inspecting reflection cycles:
- Step-by-step reflection cycle execution
- Breakpoint setting at specific reflection stages
- Before/after state comparison
- Performance profiling for reflection operations
- Contradiction detection visualization

### 4. Consciousness State Inspector (`consciousness_inspector.py`)

Visualization and analysis of ΨC consciousness state:
- Real-time visualization of consciousness metrics:
  - Ψ<sub>C</sub>(t): Consciousness index over time
  - ΔH(t): Entropy change over time
  - Coherence(t): Coherence metric over time
- Interactive adjustment of parameters:
  - θ: Consciousness threshold
  - λ<sub>reflect</sub>: Reflection rate
  - window_size: Memory window size
- State comparison across parameter settings
- Critical state detection and alerts

### 5. Synthetic Stress Test Generator (`stress_test_generator.py`)

Generate synthetic stress tests for ΨC agent systems:
- Memory flood testing
- Contradiction injection
- Coherence degradation simulation
- Schema mutation stress testing
- Boundary condition exploration

## Getting Started

To launch the development environment with all tools:

```bash
python -m tools.dev_environment.launcher
```

Or launch individual tools:

```bash
python -m tools.dev_environment.schema_editor
python -m tools.dev_environment.memory_sandbox
python -m tools.dev_environment.reflection_debugger
python -m tools.dev_environment.consciousness_inspector
python -m tools.dev_environment.stress_test_generator
```

## Technical Properties

The development environment provides tools for:

1. Real-time visualization of:
   - Ψ<sub>C</sub>(t): Consciousness index over time
   - ΔH(t): Entropy change over time
   - Coherence(t): Coherence metric over time

2. Interactive adjustment of:
   - θ: Consciousness threshold
   - λ<sub>reflect</sub>: Reflection rate
   - window_size: Memory window size

## Requirements

- Python 3.8+
- Required packages:
  - matplotlib
  - networkx
  - numpy
  - rich
  - streamlit (for web-based interfaces)
  - pyvis (for interactive network visualization)

## Integration with ΨC-AI SDK

The development environment integrates directly with the ΨC-AI SDK to provide real-time monitoring and interaction with running agents. All tools can be connected to:

1. Running ΨC agents
2. Saved agent state files
3. Simulated agent environments

## Usage Examples

```python
# Connect development tools to a running agent
from psi_c_ai_sdk import PsiAgent
from tools.dev_environment.schema_editor import SchemaEditor

agent = PsiAgent()
editor = SchemaEditor(agent)
editor.launch()

# Monitor consciousness metrics during operation
from tools.dev_environment.consciousness_inspector import ConsciousnessMonitor

monitor = ConsciousnessMonitor(agent)
monitor.start_tracking()

# After some agent operations...
monitor.plot_metrics()
``` 