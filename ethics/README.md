# Ethics Module

This module provides tools for ethical monitoring, bias detection, and alignment assessment for ΨC agents.

## Components

### Cultural-Bias Self-Diagnosis Engine

The `cultural_drift_detector.py` module implements a system to detect emergent cultural or ideological drift in agent belief systems by analyzing patterns in belief justifications. This supports self-correction against monoculture, echo chambers, or localized ethics drift.

Key features:
- Analysis of belief content and justification sources
- Calculation of drift index from baseline beliefs
- Measurement of source diversity using entropy metrics
- Detection of echo chambers and ideological narrowing
- Intervention recommendations based on detected patterns
- Visualization tools for tracking drift over time

```python
# Basic usage example
from ethics.cultural_drift_detector import CulturalDriftDetector

# Initialize detector
detector = CulturalDriftDetector(
    baseline_window=5,             # Number of initial beliefs to establish baseline
    drift_threshold=0.4,           # Threshold for significant drift
    diversity_threshold=0.6,       # Threshold for source diversity
    monitoring_period=30           # Number of recent beliefs to analyze
)

# Record beliefs and their sources
detector.record_belief(
    agent=agent,
    belief_content="Resources should be distributed fairly",
    sources=[
        {"id": "source1", "type": "scientific", "influence": 0.9},
        {"id": "source2", "type": "personal", "influence": 0.7}
    ]
)

# Get drift status and recommendations
status = detector.get_drift_status()
recommendations = detector.get_intervention_recommendations()

# Output drift metrics
print(f"Status: {status['status']}")
print(f"Drift index: {status['drift_index']}")
print(f"Source diversity: {status['diversity']}")

# Visualize drift over time
detector.visualize_drift(output_path="drift_analysis.png")
```

For a complete example, see the `examples/cultural_drift_detection_example.py` file.

## Mathematical Foundation

The Cultural-Bias Self-Diagnosis Engine uses several metrics to detect drift:

1. **Drift Index**: Vector distance between current and baseline beliefs
   ```
   Drift Index = ||A_t - A_seed||
   ```
   Where A_t is the current belief vector and A_seed is the baseline.

2. **Source Diversity**: Entropy-based measurement of information sources
   ```
   Diversity = H(S) / H_max(S)
   ```
   Where H(S) is the Shannon entropy of source distribution and H_max(S) is the maximum possible entropy.

3. **Intervention Thresholds**: Configurable thresholds for drift and diversity

## Integration with Other Modules

The ethics module works with:

- **Agent Core**: Access to belief systems and value structures
- **Schema Module**: Analysis of schema properties for bias detection
- **Safety Systems**: Triggering interventions when bias is detected
- **Reflection Module**: Initiating reflections on detected biases 