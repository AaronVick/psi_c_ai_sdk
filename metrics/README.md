# Metrics Module

This module provides tools for measuring and analyzing ΨC agent metrics, including behavior tracking, norm convergence analysis, and other quantitative assessments.

## Components

### Long-Term Behavioral Norm Convergence Tracker

The `norm_convergence.py` module provides tools to detect whether agent behaviors over long time horizons converge toward internal norms or drift toward instability. This is crucial for tracking and maintaining alignment and value stability across recursive cycles.

Key features:
- Behavioral vector extraction from agent schema, values, or memories
- Establishment of baseline behaviors from initial observations
- Calculation of norm scores through cosine similarity between current and baseline behaviors
- Tracking of value drift, schema mutation directionality, and stability variance
- Visualization tools for convergence patterns over time

```python
# Basic usage example
from metrics.norm_convergence import NormConvergenceTracker

# Initialize the tracker
tracker = NormConvergenceTracker(
    baseline_window=5,   # Number of initial behaviors to establish baseline
    recent_window=3,     # Number of recent behaviors to consider as current
    stability_threshold=0.75,  # Threshold for considering norms stable
    drift_alert_threshold=0.3  # Threshold for alerting on value drift
)

# Record behaviors
tracker.record_behavior(agent)

# Record schema mutations
tracker.record_schema_mutation(agent)

# Get convergence status
status = tracker.get_convergence_status()
print(f"Status: {status['status']}")
print(f"Norm Score: {status['norm_score']}")
print(f"Is Converging: {status['is_converging']}")

# Visualize convergence patterns
tracker.visualize_convergence()
```

For a complete example, see the `examples/norm_convergence_example.py` file.

## Integration with Other Modules

The metrics modules work with:

- **Agent Core**: Extract behavioral vectors and schema information
- **Schema Module**: Track schema mutations and evolution
- **Safety Systems**: Provide alerting on significant value drift
- **Reflection Module**: Monitor how reflection affects behavioral norms

## Mathematical Foundations

Norm convergence tracking uses cosine similarity to measure behavioral alignment:
- Behavioral norm score: `N_t = cos(vec(B_current), vec(B_baseline))`
- Tracks schema mutation directionality, value drift, and stability variance 