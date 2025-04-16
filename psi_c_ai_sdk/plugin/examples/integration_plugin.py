"""
Integration Plugin for ΨC-AI SDK

This module provides a plugin for integrating the ΨC-AI SDK with external services
via REST APIs, webhooks, and data transformation utilities.
"""

import json
import logging
import os
import requests
import time
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urljoin

from psi_c_ai_sdk.plugin.base import (
    PluginBase,
    PluginInfo,
    PluginHook,
    PluginType,
    create_plugin_id
)
from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.schema.schema import SchemaGraph


class IntegrationPlugin(PluginBase):
    """
    Plugin for integrating the ΨC-AI SDK with external services.
    
    This plugin provides functionality to integrate with external services
    via REST APIs, webhooks, and data transformation utilities.
    """
    
    @classmethod
    def _get_plugin_info(cls) -> PluginInfo:
        """Get metadata about the plugin."""
        return PluginInfo(
            id=create_plugin_id("integration", "psi_c_example"),
            name="Integration Plugin",
            version="0.1.0",
            description="A plugin for integrating the ΨC-AI SDK with external services",
            author="ΨC-AI SDK Team",
            plugin_type=PluginType.INTEGRATION,
            hooks={
                PluginHook.POST_REFLECTION,
                PluginHook.POST_MEMORY_ADD,
                PluginHook.POST_SCHEMA_UPDATE,
                PluginHook.RUNTIME_MONITORING
            },
            tags=["integration", "api", "webhook", "external", "example"]
        )
    
    def _register_hooks(self) -> Dict[PluginHook, Any]:
        """Register the hooks that this plugin implements."""
        return {
            PluginHook.POST_REFLECTION: self.post_reflection_handler,
            PluginHook.POST_MEMORY_ADD: self.post_memory_add_handler,
            PluginHook.POST_SCHEMA_UPDATE: self.post_schema_update_handler,
            PluginHook.RUNTIME_MONITORING: self.runtime_monitoring_handler
        }
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        self.logger.info("Initializing Integration Plugin")
        
        # Set default config if not provided
        if not self.config:
            self.config = {
                "api_endpoints": {
                    "memory": None,
                    "schema": None,
                    "reflection": None,
                    "monitoring": None
                },
                "webhooks": {
                    "memory": None,
                    "schema": None,
                    "reflection": None,
                    "monitoring": None
                },
                "api_auth": {
                    "type": None,  # "basic", "bearer", "api_key", etc.
                    "username": None,
                    "password": None,
                    "token": None,
                    "key_name": None,
                    "key_value": None
                },
                "rate_limit": 60,  # requests per minute
                "retry_count": 3,
                "timeout": 10,
                "enabled_hooks": ["POST_REFLECTION", "POST_SCHEMA_UPDATE"]
            }
        
        # Track statistics
        self.stats = {
            "api_calls": 0,
            "webhook_calls": 0,
            "errors": 0,
            "last_api_call": None
        }
        
        # Initialize rate limiting
        self.last_request_time = 0
        
        self.logger.info("Integration Plugin initialized")
        return True
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.logger.info(
            f"Integration Plugin shutdown. Stats: {self.stats}"
        )
    
    def post_reflection_handler(
        self,
        result: Dict[str, Any],
        memories: Optional[List[Memory]] = None,
        schema: Optional[SchemaGraph] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-reflection events.
        
        This method is called after the reflection cycle completes
        and can send the reflection results to external services.
        
        Args:
            result: Results from the reflection cycle
            memories: Memories that were reflected upon
            schema: Updated schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        if "POST_REFLECTION" not in self.config.get("enabled_hooks", []):
            return {"skipped": True, "reason": "Hook not enabled in configuration"}
        
        self.logger.info("Post-reflection: Sending reflection data to external services")
        
        # Prepare data payload
        payload = self._prepare_reflection_payload(result, memories, schema)
        
        # Call API endpoint if configured
        api_result = self._call_api("reflection", payload)
        
        # Call webhook if configured
        webhook_result = self._call_webhook("reflection", payload)
        
        return {
            "api_result": api_result,
            "webhook_result": webhook_result
        }
    
    def post_memory_add_handler(
        self,
        memory: Memory,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-memory-add events.
        
        This method is called after a memory is added to the memory store
        and can send the memory data to external services.
        
        Args:
            memory: Memory that was added
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        if "POST_MEMORY_ADD" not in self.config.get("enabled_hooks", []):
            return {"skipped": True, "reason": "Hook not enabled in configuration"}
        
        self.logger.info("Post-memory-add: Sending memory data to external services")
        
        # Prepare data payload
        payload = self._prepare_memory_payload(memory)
        
        # Call API endpoint if configured
        api_result = self._call_api("memory", payload)
        
        # Call webhook if configured
        webhook_result = self._call_webhook("memory", payload)
        
        return {
            "api_result": api_result,
            "webhook_result": webhook_result
        }
    
    def post_schema_update_handler(
        self,
        schema: SchemaGraph,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-schema-update events.
        
        This method is called after the schema graph is updated
        and can send the schema data to external services.
        
        Args:
            schema: Updated schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        if "POST_SCHEMA_UPDATE" not in self.config.get("enabled_hooks", []):
            return {"skipped": True, "reason": "Hook not enabled in configuration"}
        
        self.logger.info("Post-schema-update: Sending schema data to external services")
        
        # Prepare data payload
        payload = self._prepare_schema_payload(schema)
        
        # Call API endpoint if configured
        api_result = self._call_api("schema", payload)
        
        # Call webhook if configured
        webhook_result = self._call_webhook("schema", payload)
        
        return {
            "api_result": api_result,
            "webhook_result": webhook_result
        }
    
    def runtime_monitoring_handler(
        self,
        metrics: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for runtime-monitoring events.
        
        This method is called during runtime monitoring
        and can send monitoring data to external services.
        
        Args:
            metrics: Runtime metrics
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        if "RUNTIME_MONITORING" not in self.config.get("enabled_hooks", []):
            return {"skipped": True, "reason": "Hook not enabled in configuration"}
        
        self.logger.info("Runtime-monitoring: Sending monitoring data to external services")
        
        # Prepare data payload
        payload = self._prepare_monitoring_payload(metrics)
        
        # Call API endpoint if configured
        api_result = self._call_api("monitoring", payload)
        
        # Call webhook if configured
        webhook_result = self._call_webhook("monitoring", payload)
        
        return {
            "api_result": api_result,
            "webhook_result": webhook_result
        }
    
    def _prepare_reflection_payload(
        self,
        result: Dict[str, Any],
        memories: Optional[List[Memory]] = None,
        schema: Optional[SchemaGraph] = None
    ) -> Dict[str, Any]:
        """
        Prepare payload for reflection data.
        
        Args:
            result: Reflection results
            memories: Reflected memories
            schema: Schema graph
            
        Returns:
            Dictionary with payload data
        """
        payload = {
            "timestamp": self._get_timestamp(),
            "event_type": "reflection",
            "results": result,
        }
        
        # Add memory summary if available
        if memories:
            payload["memory_count"] = len(memories)
            payload["memory_summary"] = self._summarize_memories(memories)
        
        # Add schema summary if available
        if schema:
            schema_summary = self._summarize_schema(schema)
            if schema_summary:
                payload["schema_summary"] = schema_summary
        
        return payload
    
    def _prepare_memory_payload(self, memory: Memory) -> Dict[str, Any]:
        """
        Prepare payload for memory data.
        
        Args:
            memory: Memory object
            
        Returns:
            Dictionary with payload data
        """
        payload = {
            "timestamp": self._get_timestamp(),
            "event_type": "memory_added",
            "memory_id": getattr(memory, "id", str(id(memory))),
        }
        
        # Add memory attributes
        for attr in ["content", "importance", "memory_type", "created_at", "tags"]:
            if hasattr(memory, attr):
                payload[attr] = getattr(memory, attr)
        
        return payload
    
    def _prepare_schema_payload(self, schema: SchemaGraph) -> Dict[str, Any]:
        """
        Prepare payload for schema data.
        
        Args:
            schema: Schema graph
            
        Returns:
            Dictionary with payload data
        """
        payload = {
            "timestamp": self._get_timestamp(),
            "event_type": "schema_updated",
            "schema_summary": self._summarize_schema(schema)
        }
        
        return payload
    
    def _prepare_monitoring_payload(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare payload for monitoring data.
        
        Args:
            metrics: Runtime metrics
            
        Returns:
            Dictionary with payload data
        """
        payload = {
            "timestamp": self._get_timestamp(),
            "event_type": "monitoring",
            "metrics": metrics
        }
        
        return payload
    
    def _call_api(self, endpoint_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an API endpoint with the given payload.
        
        Args:
            endpoint_type: Type of endpoint to call (memory, schema, reflection, monitoring)
            payload: Data payload to send
            
        Returns:
            Dictionary with API call results
        """
        endpoint_url = self.config.get("api_endpoints", {}).get(endpoint_type)
        if not endpoint_url:
            return {"success": False, "reason": f"No API endpoint configured for {endpoint_type}"}
        
        # Apply rate limiting
        self._rate_limit()
        
        # Prepare request headers
        headers = {"Content-Type": "application/json"}
        
        # Add authentication if configured
        auth_type = self.config.get("api_auth", {}).get("type")
        if auth_type == "basic":
            username = self.config["api_auth"].get("username")
            password = self.config["api_auth"].get("password")
            auth = (username, password) if username and password else None
        elif auth_type == "bearer":
            token = self.config["api_auth"].get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            auth = None
        elif auth_type == "api_key":
            key_name = self.config["api_auth"].get("key_name")
            key_value = self.config["api_auth"].get("key_value")
            if key_name and key_value:
                headers[key_name] = key_value
            auth = None
        else:
            auth = None
        
        # Make the request with retry logic
        result = {"success": False, "status_code": None, "response": None}
        retry_count = self.config.get("retry_count", 3)
        timeout = self.config.get("timeout", 10)
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    endpoint_url,
                    json=payload,
                    headers=headers,
                    auth=auth,
                    timeout=timeout
                )
                
                result["status_code"] = response.status_code
                result["success"] = 200 <= response.status_code < 300
                
                try:
                    result["response"] = response.json()
                except ValueError:
                    result["response"] = response.text
                
                # Update stats
                self.stats["api_calls"] += 1
                self.stats["last_api_call"] = self._get_timestamp()
                
                if result["success"]:
                    break
                
            except requests.RequestException as e:
                result["error"] = str(e)
                self.stats["errors"] += 1
            
            # Wait before retrying
            if attempt < retry_count - 1:
                time.sleep(0.5 * (attempt + 1))
        
        return result
    
    def _call_webhook(self, webhook_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a webhook with the given payload.
        
        Args:
            webhook_type: Type of webhook to call (memory, schema, reflection, monitoring)
            payload: Data payload to send
            
        Returns:
            Dictionary with webhook call results
        """
        webhook_url = self.config.get("webhooks", {}).get(webhook_type)
        if not webhook_url:
            return {"success": False, "reason": f"No webhook configured for {webhook_type}"}
        
        # Apply rate limiting
        self._rate_limit()
        
        # Make the request with retry logic
        result = {"success": False, "status_code": None, "response": None}
        retry_count = self.config.get("retry_count", 3)
        timeout = self.config.get("timeout", 10)
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout
                )
                
                result["status_code"] = response.status_code
                result["success"] = 200 <= response.status_code < 300
                
                try:
                    result["response"] = response.json()
                except ValueError:
                    result["response"] = response.text
                
                # Update stats
                self.stats["webhook_calls"] += 1
                
                if result["success"]:
                    break
                
            except requests.RequestException as e:
                result["error"] = str(e)
                self.stats["errors"] += 1
            
            # Wait before retrying
            if attempt < retry_count - 1:
                time.sleep(0.5 * (attempt + 1))
        
        return result
    
    def _rate_limit(self) -> None:
        """
        Apply rate limiting to API and webhook calls.
        """
        rate_limit = self.config.get("rate_limit", 60)  # requests per minute
        if rate_limit <= 0:
            return
        
        # Calculate minimum time between requests in seconds
        min_interval = 60.0 / rate_limit
        
        # Check if we need to wait
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < min_interval:
            # Wait for the remaining time
            wait_time = min_interval - elapsed
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
    
    def _summarize_memories(self, memories: List[Memory]) -> List[Dict[str, Any]]:
        """
        Create a summary of memory objects for external services.
        
        Args:
            memories: List of memory objects
            
        Returns:
            List of memory summaries
        """
        summary = []
        for memory in memories:
            memory_summary = {
                "id": getattr(memory, "id", str(id(memory)))
            }
            
            # Add basic attributes if available
            for attr in ["content", "importance", "memory_type", "created_at"]:
                if hasattr(memory, attr):
                    memory_summary[attr] = getattr(memory, attr)
            
            # Add tags if available
            if hasattr(memory, "tags") and getattr(memory, "tags"):
                memory_summary["tags"] = getattr(memory, "tags")
            
            summary.append(memory_summary)
        
        return summary
    
    def _summarize_schema(self, schema: SchemaGraph) -> Dict[str, Any]:
        """
        Create a summary of a schema graph for external services.
        
        Args:
            schema: Schema graph
            
        Returns:
            Dictionary with schema summary
        """
        try:
            # Get the graph from the schema
            G = schema.get_graph() if hasattr(schema, 'get_graph') else None
            
            if not G:
                return {}
            
            # Get node and edge counts
            node_count = len(G.nodes())
            edge_count = len(G.edges())
            
            # Calculate average degree
            if node_count > 0:
                avg_degree = 2 * edge_count / node_count
            else:
                avg_degree = 0
            
            # Calculate average edge weight (coherence)
            weights = [data.get('weight', 0) for _, _, data in G.edges(data=True)]
            avg_weight = sum(weights) / len(weights) if weights else 0
            
            # Get node types
            node_types = {}
            for _, data in G.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "avg_degree": avg_degree,
                "avg_coherence": avg_weight,
                "node_types": node_types,
                "density": 2 * edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing schema: {str(e)}")
            return {}
    
    def _get_timestamp(self) -> str:
        """Generate an ISO format timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# This allows the plugin to be loaded directly
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the plugin
    plugin = IntegrationPlugin()
    plugin.initialize()
    
    # Create a mock config with test webhook (comment out in production)
    # plugin.config["webhooks"]["reflection"] = "https://httpbin.org/post"
    # plugin.config["enabled_hooks"] = ["POST_REFLECTION"]
    
    # Test the reflection handler with a simple payload
    test_result = plugin.post_reflection_handler(
        result={"coherence_before": 0.5, "coherence_after": 0.7},
        memories=[],
        schema=None
    )
    
    print(f"Integration test result: {test_result}")
    
    plugin.shutdown() 