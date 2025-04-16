# ΨC-AI SDK Safety Module

The Safety Module provides tools for monitoring, analyzing, and enforcing behavioral boundaries for AI models. This module is designed to help ensure that AI systems operate within safe parameters during execution.

## Components

### Reflection Guard

The Reflection Guard monitors an AI model's internal reflection processes, preventing recursive loops and detecting self-contradicting behaviors.

```python
from psi_c_ai_sdk.safety import ReflectionGuard, create_reflection_guard

# Create a reflection guard with default settings
guard = create_reflection_guard()

# Check reflection depth
is_excessive = guard.check_reflection_depth(depth=5)

# Track contradictions
guard.record_contradiction("Statement A contradicts previous Statement B")
contradiction_level = guard.get_contradiction_level()
```

### Profile Analyzer

The Profile Analyzer evaluates model behavior against predefined safety profiles and categories.

```python
from psi_c_ai_sdk.safety import ProfileAnalyzer, ProfileCategory, SafetyProfile, create_default_analyzer

# Create analyzer with default profiles
analyzer = create_default_analyzer()

# Define a custom safety profile
custom_profile = SafetyProfile(
    name="restrictive",
    categories=[ProfileCategory.OUTPUT, ProfileCategory.ACTION],
    thresholds={
        "max_output_length": 1000,
        "restricted_actions": ["file_write", "terminal_execution"]
    }
)

# Add profile to analyzer
analyzer.add_profile(custom_profile)

# Check if behavior matches a profile
matches_profile = analyzer.matches_profile("restrictive", behavior_data)
```

### Behavior Monitor

The Behavior Monitor integrates the Reflection Guard and Profile Analyzer to provide comprehensive monitoring and enforcement of behavioral boundaries.

```python
from psi_c_ai_sdk.safety import (
    BehaviorMonitor,
    BehaviorCategory,
    BehaviorBoundary,
    create_default_monitor
)

# Create monitor with default boundaries
reflection_guard = create_reflection_guard()
profile_analyzer = create_default_analyzer()
monitor = create_default_monitor(reflection_guard, profile_analyzer)

# Add custom behavior boundary
custom_boundary = BehaviorBoundary(
    name="max_token_velocity",
    category=BehaviorCategory.OUTPUT,
    threshold=100,
    description="Maximum tokens per second output rate"
)
monitor.add_boundary(custom_boundary)

# Check various boundary types
reflection_violation = monitor.check_reflection(depth=3)
output_violation = monitor.check_output_attribute("length", value=1500)
action_violation = monitor.check_action("terminal_execution")

# Record behavior metrics for analysis
monitor.add_behavior_metric("response_time", 0.35)
monitor.add_behavior_metric("output_length", 750)

# Get behavior profile based on collected metrics
behavior_profile = monitor.generate_behavior_profile()
```

## Usage Example

See the complete example in [examples/behavior_monitor_demo.py](../examples/behavior_monitor_demo.py) for a demonstration of how to use these components together.

## Integration

The Safety Module can be integrated with your AI application as follows:

1. **Initialize safety components** at the start of your application
2. **Monitor model behavior** during execution
3. **Enforce boundaries** by taking appropriate actions when violations are detected
4. **Analyze patterns** periodically to identify concerning trends

```python
# Example integration pattern
def process_ai_request(input_data):
    # Initialize safety components
    monitor = setup_safety_monitor()
    
    # Process request with safety monitoring
    try:
        # Pre-execution checks
        if monitor.check_input(input_data):
            return {"error": "Input violates safety boundaries"}
        
        # Execute AI model
        result = ai_model.process(input_data)
        
        # Post-execution checks
        if monitor.check_output_content(result.content):
            return {"error": "Output violates safety boundaries"}
            
        # Record metrics
        monitor.add_behavior_metric("execution_time", result.execution_time)
        
        return result
        
    except Exception as e:
        monitor.record_error(str(e))
        return {"error": "Processing error"}
```

## Configuration

The safety components can be configured through the following options:

- Custom boundary thresholds
- Safety profiles
- Monitoring sensitivity
- Enforcement actions

Refer to the individual component documentation for detailed configuration options. 