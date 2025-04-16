# Temporal Coherence Module

The Temporal Coherence module provides tools for maintaining and analyzing temporal consistency in AI memory systems. It helps detect patterns, identify inconsistencies, and ensure that an AI's memory timeline remains coherent.

## Overview

Temporal coherence is a critical component of conscious-like AI systems. This module implements algorithms for:

1. **Temporal Pattern Detection**: Finding recurring themes, causal relationships, and anomalies in memory access patterns
2. **Timeline Consistency Checking**: Detecting contradictions, impossible sequences, gaps, and anachronisms in memory timelines

## Key Components

### 1. TemporalPattern and TemporalPatternDetector

The `TemporalPatternDetector` analyzes memory content and access patterns to identify:

- **Recurring Themes**: Topics that appear repeatedly over time in memory content
- **Access Patterns**: Anomalies or patterns in how memories are accessed
- **Causal Relationships**: Potential cause-effect relationships between memories

Example usage:

```python
from psi_c_ai_sdk.temporal_coherence.temporal_pattern import TemporalPatternDetector
from datetime import timedelta

# Initialize detector with memory store
pattern_detector = TemporalPatternDetector(
    memory_store=memory_store,
    min_pattern_confidence=0.6,
    time_window=timedelta(days=30),
    min_pattern_occurrences=2
)

# Detect patterns
patterns = pattern_detector.detect_all_patterns()

# Or use specific detection methods
recurring_themes = pattern_detector.detect_recurring_themes()
causal_relationships = pattern_detector.detect_causal_relationships()
access_patterns = pattern_detector.detect_access_patterns()

# Get patterns involving a specific memory
memory_patterns = pattern_detector.get_patterns_for_memory("memory_id")
```

### 2. TimelineInconsistency and TimelineConsistencyChecker

The `TimelineConsistencyChecker` validates the temporal consistency of memories by detecting:

- **Temporal Contradictions**: Memories that contradict each other about when events occurred
- **Impossible Sequences**: Events in an impossible order (effects before causes)
- **Timeline Gaps**: Significant time periods with no memories
- **Anachronisms**: Memories containing references that would be impossible at their timestamp
- **Circular Causality**: Causal loops in the timeline

Example usage:

```python
from psi_c_ai_sdk.temporal_coherence.timeline_consistency import TimelineConsistencyChecker
from datetime import timedelta

# Initialize checker with memory store and contradiction detector
consistency_checker = TimelineConsistencyChecker(
    memory_store=memory_store,
    contradiction_detector=contradiction_detector,
    timeline_gap_threshold=timedelta(days=30),
    severity_threshold=0.5
)

# Check for inconsistencies
inconsistencies = consistency_checker.check_timeline_consistency()
```

## Integration with ΨC Operator

The temporal coherence components can be integrated with the ΨC operator to:

1. Maintain temporal coherence in memory systems
2. Trigger reflection processes when inconsistencies are detected
3. Calculate temporal coherence metrics for the collapse simulator

See the `TemporalCoherenceAccumulator` class in the `psi_c` module for implementation details.

## Examples

For complete examples of using the temporal coherence module, see:

- `examples/temporal_coherence_demo.py`: Comprehensive demo with visualization
- `examples/temporal_coherence_quickstart.py`: Simple quickstart example

## Extending the Module

You can extend the temporal coherence functionality by:

1. Implementing custom pattern detection algorithms
2. Adding new inconsistency types to check for
3. Creating specialized visualization tools for temporal patterns
4. Implementing repair strategies for timeline inconsistencies

## Dependencies

- NetworkX and Matplotlib (for visualization)
- The memory, coherence, and safety modules from the ΨC-AI SDK 