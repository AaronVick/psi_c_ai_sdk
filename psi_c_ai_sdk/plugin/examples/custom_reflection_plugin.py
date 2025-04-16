"""
Custom Reflection Plugin Example for ΨC-AI SDK

This module demonstrates how to create a custom reflection plugin for the ΨC-AI SDK.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from psi_c_ai_sdk.plugin.base import (
    PluginBase,
    PluginInfo,
    PluginHook,
    PluginType,
    create_plugin_id
)
from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.schema.schema import SchemaGraph


class CustomReflectionPlugin(PluginBase):
    """
    Example plugin that adds custom reflection logic to the ΨC-AI SDK.
    
    This plugin demonstrates how to hook into the reflection system and add
    custom logic for memory processing and schema updates.
    """
    
    @classmethod
    def _get_plugin_info(cls) -> PluginInfo:
        """Get metadata about the plugin."""
        return PluginInfo(
            id=create_plugin_id("custom_reflection", "psi_c_example"),
            name="Custom Reflection Plugin",
            version="0.1.0",
            description="A custom reflection plugin that demonstrates the plugin system",
            author="ΨC-AI SDK Team",
            plugin_type=PluginType.REFLECTION,
            hooks={
                PluginHook.PRE_REFLECTION,
                PluginHook.POST_REFLECTION,
                PluginHook.PRE_MEMORY_ADD
            },
            tags=["reflection", "example", "tutorial"]
        )
    
    def _register_hooks(self) -> Dict[PluginHook, Any]:
        """Register the hooks that this plugin implements."""
        return {
            PluginHook.PRE_REFLECTION: self.pre_reflection_handler,
            PluginHook.POST_REFLECTION: self.post_reflection_handler,
            PluginHook.PRE_MEMORY_ADD: self.pre_memory_add_handler
        }
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        self.logger.info("Initializing Custom Reflection Plugin")
        
        # Set default config if not provided
        if not self.config:
            self.config = {
                "importance_threshold": 0.7,
                "tag_memories": True,
                "memory_prefix": "[CR] "
            }
        
        # Track statistics
        self.stats = {
            "memories_processed": 0,
            "memories_modified": 0,
            "reflections_augmented": 0
        }
        
        self.logger.info("Custom Reflection Plugin initialized")
        return True
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.logger.info(
            f"Custom Reflection Plugin shutdown. Stats: {self.stats}"
        )
    
    def pre_reflection_handler(
        self, 
        memories: List[Memory], 
        schema: Optional[SchemaGraph] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for pre-reflection events.
        
        This method is called before the reflection cycle starts.
        
        Args:
            memories: Memories that will be reflected upon
            schema: Current schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        self.logger.info(f"Pre-reflection: Processing {len(memories)} memories")
        
        # Example: Prioritize certain types of memories
        importance_threshold = self.config.get("importance_threshold", 0.7)
        high_importance_count = 0
        
        for memory in memories:
            if getattr(memory, "importance", 0) >= importance_threshold:
                high_importance_count += 1
                
                # Add metadata to track plugin processing
                if memory.metadata is None:
                    memory.metadata = {}
                
                memory.metadata["processed_by_custom_reflection"] = True
        
        self.stats["memories_processed"] += len(memories)
        
        return {
            "total_memories": len(memories),
            "high_importance_memories": high_importance_count,
            "schema_nodes": len(schema.get_all_nodes()) if schema else 0
        }
    
    def post_reflection_handler(
        self, 
        result: Dict[str, Any], 
        schema: Optional[SchemaGraph] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-reflection events.
        
        This method is called after the reflection cycle completes.
        
        Args:
            result: Results from the reflection cycle
            schema: Updated schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        self.logger.info("Post-reflection: Processing reflection results")
        
        # Example: Track reflection metrics
        coherence_before = result.get("coherence_before", 0)
        coherence_after = result.get("coherence_after", 0)
        coherence_improvement = coherence_after - coherence_before
        
        # Add custom metadata to reflection result
        custom_result = {
            "coherence_improvement": coherence_improvement,
            "plugin_assessment": (
                "significant_improvement" if coherence_improvement > 0.2
                else "moderate_improvement" if coherence_improvement > 0.05
                else "minimal_impact"
            )
        }
        
        self.stats["reflections_augmented"] += 1
        
        return custom_result
    
    def pre_memory_add_handler(
        self, 
        memory: Memory, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for pre-memory-add events.
        
        This method is called before a memory is added to the memory store.
        
        Args:
            memory: Memory that will be added
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        if self.config.get("tag_memories", True):
            # Example: Add a tag to memories processed by this plugin
            if not hasattr(memory, "tags") or memory.tags is None:
                memory.tags = []
            
            if "custom_reflection_plugin" not in memory.tags:
                memory.tags.append("custom_reflection_plugin")
            
            # Modify memory content if specified in config
            if self.config.get("memory_prefix") and hasattr(memory, "content"):
                prefix = self.config["memory_prefix"]
                if not memory.content.startswith(prefix):
                    memory.content = f"{prefix}{memory.content}"
                    self.stats["memories_modified"] += 1
        
        return {"memory_id": memory.id, "modified": True}


# This allows the plugin to be loaded directly
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the plugin
    plugin = CustomReflectionPlugin()
    plugin.initialize()
    
    # Test the pre-reflection handler
    test_memory = Memory(
        id="test1",
        content="This is a test memory",
        importance=0.8
    )
    
    result = plugin.pre_memory_add_handler(test_memory)
    print(f"Pre-memory-add result: {result}")
    print(f"Modified memory: {test_memory.content}")
    
    plugin.shutdown() 