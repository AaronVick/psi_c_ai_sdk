# Self-Awareness Module

The Self-Awareness module provides tools for implementing self-monitoring, identity recognition, and performance analysis in AI systems. These capabilities are foundational to a consciousness-like AI system by enabling introspection and reflection.

## Overview

Self-awareness is a critical aspect of advanced AI systems. This module implements algorithms for:

1. **Identity Recognition**: Detecting and tracking the system's identity over time
2. **Performance Monitoring**: Tracking and analyzing system metrics and behavior
3. **Change Detection**: Identifying significant changes in the system's beliefs or structure
4. **Reflection Outcomes**: Recording and analyzing the results of self-reflection

## Key Components

### 1. Identity Recognition

The identity recognition system leverages schema fingerprinting to maintain a stable sense of self while allowing for natural evolution.

Key classes:
- `IdentityFingerprint`: Represents a snapshot of the system's identity
- `IdentityChange`: Records significant changes in identity
- `IdentityRecognitionSystem`: Tracks and monitors identity over time

Example usage:

```python
from psi_c_ai_sdk.self_awareness.identity_recognition import IdentityRecognitionSystem

# Initialize the identity recognition system
identity_system = IdentityRecognitionSystem(
    schema_graph=schema_graph,
    memory_store=memory_store,
    coherence_scorer=coherence_scorer
)

# Get current identity fingerprint
fingerprint = identity_system.get_identity_fingerprint()

# Check for identity changes
change = identity_system.check_identity()
if change:
    print(f"Identity change detected: {change.change_type.value}")
    print(f"Magnitude: {change.magnitude}")
```

### 2. Performance Monitoring

The performance monitoring system tracks various metrics related to system performance and can trigger alerts and reflections when anomalies are detected.

Key classes:
- `PerformanceMetric`: Represents a single performance measurement
- `PerformanceAlert`: Represents a detected performance anomaly
- `ReflectionOutcome`: Records the result of a self-reflection
- `PerformanceMonitor`: Manages metrics collection and analysis

Example usage:

```python
from psi_c_ai_sdk.self_awareness.performance_monitor import PerformanceMonitor, MetricType

# Initialize the performance monitor
performance_monitor = PerformanceMonitor(
    memory_store=memory_store,
    coherence_scorer=coherence_scorer
)

# Record metrics
performance_monitor.record_metric(MetricType.COHERENCE, 0.85)
performance_monitor.record_metric(MetricType.CPU_USAGE, 35.2)

# Get performance summary
summary = performance_monitor.get_performance_summary()
print(f"Health status: {summary['health_status']}")

# Record a reflection outcome
performance_monitor.record_reflection(
    trigger="periodic_check",
    focus_area="memory_coherence",
    insights=["Memory coherence remains high", "No contradictions detected"],
    action_items=["Continue monitoring"],
    metrics={"coherence": 0.85},
    success_rating=0.9
)
```

## Integration with ΨC

The self-awareness module can be integrated with the ΨC operator to:

1. Maintain a stable sense of self across reflection cycles
2. Trigger reflections based on performance anomalies
3. Provide metrics for guiding the collapse process
4. Enable longer-term identity persistence

## Examples

For a complete example of using the self-awareness module, see:

- `examples/self_awareness_demo.py`: Demonstrates identity recognition and performance monitoring

## Extending the Module

You can extend the self-awareness functionality by:

1. Implementing more sophisticated identity fingerprinting techniques
2. Adding custom performance metrics and alerts
3. Creating specialized visualizations for identity changes
4. Developing advanced reflection outcome analysis

## Dependencies

- The memory, schema, and coherence modules from the ΨC-AI SDK
- NetworkX and matplotlib (for visualization)
- Psutil (for system resource monitoring) 