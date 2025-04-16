# Entropy Module for ΨC-AI SDK

This module provides tools for measuring, monitoring, and managing entropy in AI memory systems, which is essential for detecting and responding to instability in consciousness models.

## Overview

Entropy in AI consciousness models represents the degree of disorder, confusion, or contradiction in memory and cognitive structures. High entropy can lead to instability, degraded performance, or consciousness collapse in ΨC systems.

## Core Components

### Entropy Calculation

The entropy calculation system provides tools to measure different types of entropy:

- **EmbeddingEntropyMeasure**: Measures entropy based on the distribution of memory embeddings
- **SemanticCoherenceEntropyMeasure**: Measures entropy based on contradictions in memory content
- **TemporalCoherenceEntropyMeasure**: Measures entropy based on temporal patterns in memory access
- **EntropyCalculator**: Combines multiple entropy measures for a comprehensive assessment

### Entropy Monitoring

The entropy monitoring system provides tools to track and respond to entropy levels:

- **EntropyMonitor**: Continuously tracks entropy levels with configurable thresholds
- **EntropyAlert**: Represents different levels of entropy alerts (Normal, Elevated, High, Critical)
- **EntropySubscriber**: Interface for components that need entropy notifications

## Usage Examples

### Basic Entropy Calculation

```python
from psi_c_ai_sdk.memory.memory import MemoryStore
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator

# Create a memory store
memory_store = MemoryStore()

# Create entropy calculator
entropy_calculator = EntropyCalculator()

# Calculate entropy
entropy = entropy_calculator.calculate_memory_store_entropy(memory_store)
print(f"System entropy: {entropy:.4f}")

# Get detailed metrics
detailed_metrics = entropy_calculator.get_detailed_entropy_metrics(memory_store)
print("Detailed entropy metrics:", detailed_metrics)
```

### Setting Up Entropy Monitoring

```python
from psi_c_ai_sdk.memory.memory import MemoryStore
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator
from psi_c_ai_sdk.entropy.monitor import EntropyMonitor, EntropyAlert, EntropySubscriber

# Create a memory store
memory_store = MemoryStore()

# Create a subscriber to respond to entropy alerts
class MyEntropyHandler(EntropySubscriber):
    def on_entropy_alert(self, alert_level, entropy_value, details):
        print(f"Entropy alert: {alert_level.name} (value: {entropy_value:.4f})")
        
        # Take appropriate action based on alert level
        if alert_level == EntropyAlert.ELEVATED:
            print("Taking mild action to reduce entropy")
        elif alert_level == EntropyAlert.HIGH:
            print("Taking stronger action to reduce entropy")
        elif alert_level == EntropyAlert.CRITICAL:
            print("Taking emergency action to reduce entropy")
    
    def on_termination_decision(self, entropy_value, details):
        print(f"Termination decision required (entropy: {entropy_value:.4f})")
        # Return False to override termination
        return False

# Create entropy monitor with custom thresholds
monitor = EntropyMonitor(
    memory_store=memory_store,
    elevated_threshold=0.6,
    high_threshold=0.75,
    critical_threshold=0.85,
    termination_threshold=0.95,
    check_interval=5.0
)

# Add subscriber
handler = MyEntropyHandler()
monitor.add_subscriber(handler)

# Start monitoring
monitor.start()
```

## Integration with PsiToolkit

The entropy module is fully integrated with the PsiToolkit for comprehensive consciousness monitoring:

```python
from psi_c_ai_sdk.psi_c.toolkit import PsiToolkit, PsiToolkitConfig

# Create toolkit with custom entropy thresholds
config = PsiToolkitConfig(
    elevated_entropy_threshold=0.6,
    high_entropy_threshold=0.75,
    critical_entropy_threshold=0.85,
    termination_entropy_threshold=0.95,
    entropy_check_interval=5.0
)

toolkit = PsiToolkit(config=config)

# Register callback for entropy alerts
toolkit.register_entropy_callback(
    lambda alert, value, details: print(f"Entropy alert: {alert.name} ({value:.4f})")
)

# Start monitoring
toolkit.start_monitoring()
```

## Entropy Examples

See the example scripts for comprehensive demonstrations:

- `examples/entropy_monitor_demo.py`: Demonstrates the entropy monitoring system
- `examples/toolkit_complete_demo.py`: Shows integration of entropy monitoring with the full ΨC toolkit 