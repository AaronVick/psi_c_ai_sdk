# Plugin Development Guide for ΨC-AI SDK

This guide explains how to develop plugins for the ΨC-AI SDK to extend its functionality with custom components, integrations, and behaviors.

## Table of Contents

- [Introduction](#introduction)
- [Plugin System Architecture](#plugin-system-architecture)
- [Creating a Plugin](#creating-a-plugin)
- [Plugin Types and Hooks](#plugin-types-and-hooks)
- [Plugin Packaging](#plugin-packaging)
- [Debugging Plugins](#debugging-plugins)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Introduction

The ΨC-AI SDK Plugin System allows developers to extend the functionality of the SDK without modifying core code. Plugins can add custom reflection strategies, memory management, belief systems, schema manipulation, visualizations, and more.

Key benefits of using plugins:

- **Modularity**: Add new features without changing the core system
- **Isolation**: Test experimental components safely
- **Sharing**: Share extensions with the community
- **Integration**: Integrate with external systems and tools

## Plugin System Architecture

The ΨC-AI SDK Plugin System consists of several key components:

- **PluginBase**: Abstract base class that all plugins must inherit from
- **PluginRegistry**: Central registry for managing plugins
- **PluginLoader**: Utility for loading and instantiating plugins
- **Plugin Hooks**: Extension points where plugins can hook into the SDK

The system uses a hook-based architecture where plugins register handlers for specific events in the SDK lifecycle, such as before/after reflection, memory addition, and schema updates.

## Creating a Plugin

To create a plugin for the ΨC-AI SDK, follow these steps:

### 1. Create a new Python package

```bash
mkdir my_psi_plugin
cd my_plugin
touch __init__.py
touch my_plugin.py
```

### 2. Inherit from PluginBase

In `my_plugin.py`, create a class that inherits from `PluginBase`:

```python
from psi_c_ai_sdk.plugin.base import (
    PluginBase,
    PluginInfo,
    PluginHook,
    PluginType,
    create_plugin_id
)

class MyCustomPlugin(PluginBase):
    @classmethod
    def _get_plugin_info(cls) -> PluginInfo:
        return PluginInfo(
            id=create_plugin_id("my_custom_plugin", "your_name"),
            name="My Custom Plugin",
            version="0.1.0",
            description="A custom plugin for ΨC-AI SDK",
            author="Your Name",
            plugin_type=PluginType.REFLECTION,  # Choose appropriate type
            hooks={
                PluginHook.PRE_REFLECTION,
                PluginHook.POST_REFLECTION
            }
        )
    
    def _register_hooks(self):
        return {
            PluginHook.PRE_REFLECTION: self.pre_reflection_handler,
            PluginHook.POST_REFLECTION: self.post_reflection_handler
        }
    
    def initialize(self) -> bool:
        # Initialize plugin resources
        self.logger.info("Initializing My Custom Plugin")
        return True
    
    def shutdown(self) -> None:
        # Clean up resources
        self.logger.info("Shutting down My Custom Plugin")
    
    def pre_reflection_handler(self, *args, **kwargs):
        # Handle pre-reflection event
        self.logger.info("Pre-reflection event")
        return {"status": "success"}
    
    def post_reflection_handler(self, *args, **kwargs):
        # Handle post-reflection event
        self.logger.info("Post-reflection event")
        return {"status": "success"}
```

### 3. Create a plugin manifest

Create a `plugin.json` file in your plugin directory:

```json
{
    "id": "your_name.my_custom_plugin-12345678",
    "name": "My Custom Plugin",
    "version": "0.1.0",
    "description": "A custom plugin for ΨC-AI SDK",
    "author": "Your Name",
    "type": "reflection",
    "hooks": ["pre_reflection", "post_reflection"],
    "entry_point": "my_psi_plugin.my_plugin.MyCustomPlugin",
    "dependencies": {},
    "tags": ["reflection", "custom"]
}
```

## Plugin Types and Hooks

The ΨC-AI SDK supports various plugin types and hooks:

### Plugin Types

- **REFLECTION**: Custom reflection strategies
- **MEMORY**: Memory management extensions
- **BELIEF**: Belief system extensions
- **SCHEMA**: Schema manipulation plugins
- **COHERENCE**: Custom coherence scoring
- **VISUALIZATION**: Visualization and UI extensions
- **INTEGRATION**: External system integrations
- **ANALYSIS**: Analysis and monitoring tools
- **SECURITY**: Safety and security extensions
- **CUSTOM**: Custom plugin types

### Plugin Hooks

- **PRE_REFLECTION**: Before reflection cycle
- **POST_REFLECTION**: After reflection cycle
- **PRE_MEMORY_ADD**: Before adding a memory
- **POST_MEMORY_ADD**: After adding a memory
- **PRE_SCHEMA_UPDATE**: Before updating the schema
- **POST_SCHEMA_UPDATE**: After updating the schema
- **COHERENCE_SCORING**: During coherence calculation
- **BELIEF_REVISION**: During belief revision
- **RUNTIME_MONITORING**: During runtime
- **CUSTOM_HOOK**: Custom extension point

Choose the appropriate plugin type and hooks based on the functionality you want to extend.

## Plugin Packaging

To package your plugin for distribution:

1. Ensure your plugin follows the structure:

```
my_psi_plugin/
├── __init__.py
├── my_plugin.py
└── plugin.json
```

2. Install your plugin locally:

```bash
# From your plugin directory
pip install -e .
```

3. Or distribute it via PyPI:

```bash
python setup.py sdist
twine upload dist/*
```

## Debugging Plugins

To debug your plugin:

1. Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Test your plugin directly:

```python
from my_psi_plugin.my_plugin import MyCustomPlugin

plugin = MyCustomPlugin()
plugin.initialize()
# Test plugin functionality
plugin.shutdown()
```

3. Use the plugin registry for integration testing:

```python
from psi_c_ai_sdk.plugin.registry import get_plugin_registry

registry = get_plugin_registry()
registry.add_plugin_path("/path/to/plugins")
registry.discover_plugins()
registry.load_plugin("your_name.my_custom_plugin-12345678")
```

## Best Practices

Follow these best practices when developing plugins:

1. **Respect Core System**: Don't modify core data structures directly
2. **Handle Errors Gracefully**: Catch exceptions and log errors
3. **Clean Up Resources**: Implement `shutdown()` to clean up resources
4. **Document Your Plugin**: Provide clear documentation and examples
5. **Version Compatibility**: Specify SDK version compatibility
6. **Loose Coupling**: Minimize dependencies on internal SDK details
7. **Configuration Options**: Make your plugin configurable
8. **Provide Feedback**: Log informative messages about plugin activity

## Examples

Several example plugins are included with the ΨC-AI SDK:

- **Custom Reflection Plugin**: Demonstrates custom reflection logic
- **Memory Visualization Plugin**: Adds visualization for memory objects
- **Schema Export Plugin**: Exports schema graphs to various formats
- **Integration Plugin**: Integrates with external services

Check the `psi_c_ai_sdk/plugin/examples` directory for more examples.

## Integration with Core SDK

To use plugins in your ΨC-AI application:

```python
from psi_c_ai_sdk.plugin.registry import get_plugin_registry

# Get the plugin registry
registry = get_plugin_registry()

# Discover available plugins
discovered_plugins = registry.discover_plugins()
print(f"Discovered {len(discovered_plugins)} plugins")

# Load a specific plugin
plugin = registry.load_plugin("your_name.my_custom_plugin-12345678")

# Run a specific hook
results = registry.run_hook(PluginHook.PRE_REFLECTION, memories=my_memories, schema=my_schema)
```

This allows your application to dynamically discover and use available plugins.

---

For more information, see the [API Reference Documentation](./api_reference.md) or the [Plugin System API](./api/plugin_system.md). 