# Schema Module

This module provides tools for working with schema structures in ΨC agents, including schema partitioning, compression, and fork detection.

## Components

### Schema Fork Detector

The `fork_detector.py` module provides tools to detect when a schema mutation has functionally forked an agent's identity. This is crucial for tracking and managing long-term agent evolution.

Key features:
- Schema embedding generation to create fixed-length vector representations of schema graphs
- Statistical drift calculation to measure deviation from historical schema trajectory
- Fork detection based on defined statistical thresholds
- Visualization tools for trajectory monitoring
- Persistence capabilities for detected forks

```python
# Basic usage example
from schema.fork_detector import SchemaForkDetector

# Initialize the detector
detector = SchemaForkDetector(
    history_window=10,    # Number of historical states to consider
    drift_threshold=3.0,  # Standard deviation threshold for fork detection
    min_history_required=3  # Minimum historical data points needed
)

# Monitor an agent
drift, is_fork = detector.monitor_agent(agent)

# Visualize the trajectory
detector.visualize_trajectory()
```

For a complete example, see the `examples/schema_fork_detection_example.py` file.

### Meta Schema Compressor

The `meta_compressor.py` module detects recurring meta-patterns in schema evolution and compresses them into abstract operators. This helps manage schema growth and enables emergent symbolic abstraction in long-running agents.

### Schema Partitioning

The `partitioning.py` module provides methods for partitioning large schema graphs into manageable clusters and components, enabling more efficient processing and understanding of complex schema structures.

## Integration with Other Modules

The schema modules work with:

- **Agent Core**: Provides access to the agent's schema graph for analysis
- **Memory Systems**: Analyzes how memories are incorporated into the schema
- **Reflection**: Monitors how reflection changes schema structure over time
- **Safety Systems**: Alerts when schema forks might indicate safety issues

## Mathematical Foundations

Schema fork detection utilizes statistical drift detection:
- Compares current schema embedding with historical trajectory: `Δ_trajectory = ||Σ_t - E[Σ_(t-1..t-n)]||`
- Flags when drift exceeds historical entropy bounds: `Δ > 3σ_Σ` 