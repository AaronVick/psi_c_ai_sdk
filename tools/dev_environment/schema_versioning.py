#!/usr/bin/env python3
"""
Schema Versioning and Health Monitoring System

This module provides tools for versioning schema snapshots, monitoring schema health,
and managing schema evolution over time. It ensures deployment readiness by tracking
schema stability, drift, and maintaining version history.

The versioning system implements best practices for schema management, including
semantic versioning, health metrics tracking, and automated migration capabilities.
"""

import os
import sys
import json
import hashlib
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
import networkx as nx

logger = logging.getLogger(__name__)

class SchemaVersionManager:
    """
    Schema Version Manager for tracking and versioning schema evolution.
    
    This class provides tools for maintaining schema version history,
    calculating schema fingerprints, and monitoring schema health
    and stability over time.
    
    Attributes:
        version_history: History of schema versions
        current_version: Current schema version
        schema_dir: Directory for storing schema versions
        health_metrics: Historical schema health metrics
    """
    
    def __init__(self, schema_dir: str = None):
        """
        Initialize the schema version manager.
        
        Args:
            schema_dir: Directory for storing schema versions
        """
        self.schema_dir = schema_dir or "schema_versions"
        self.version_history = []
        self.current_version = {"major": 0, "minor": 0, "patch": 0}
        self.health_metrics = []
        
        # Create schema directory if it doesn't exist
        if not os.path.exists(self.schema_dir):
            os.makedirs(self.schema_dir)
            
        # Load version history if available
        self._load_version_history()
        
    def _load_version_history(self):
        """Load version history from disk."""
        history_path = os.path.join(self.schema_dir, "version_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    self.version_history = data.get("versions", [])
                    self.current_version = data.get("current_version", {"major": 0, "minor": 0, "patch": 0})
                    self.health_metrics = data.get("health_metrics", [])
            except Exception as e:
                logger.error(f"Error loading version history: {e}")
                
    def _save_version_history(self):
        """Save version history to disk."""
        history_path = os.path.join(self.schema_dir, "version_history.json")
        try:
            with open(history_path, 'w') as f:
                json.dump({
                    "versions": self.version_history,
                    "current_version": self.current_version,
                    "health_metrics": self.health_metrics
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version history: {e}")
            
    def calculate_schema_fingerprint(self, schema_graph):
        """
        Calculate a fingerprint (hash) for a schema graph.
        
        Args:
            schema_graph: NetworkX graph representing the schema
            
        Returns:
            str: Schema fingerprint
        """
        # Create a deterministic representation of the graph
        nodes = sorted(list(schema_graph.nodes()))
        edges = sorted([(u, v) for u, v in schema_graph.edges()])
        
        # Create a string representation of the graph structure
        graph_str = f"Nodes:{nodes}|Edges:{edges}"
        
        # Calculate hash
        hash_obj = hashlib.sha256(graph_str.encode())
        
        return hash_obj.hexdigest()
    
    def create_version(self, schema_graph, schema_integration, version_type="patch", commit_message=""):
        """
        Create a new schema version.
        
        Args:
            schema_graph: NetworkX graph representing the schema
            schema_integration: MemorySchemaIntegration instance
            version_type: Type of version increment (major, minor, patch)
            commit_message: Message describing the changes
            
        Returns:
            str: New version string
        """
        # Calculate schema fingerprint
        fingerprint = self.calculate_schema_fingerprint(schema_graph)
        
        # Check if schema has changed
        if self.version_history and self.version_history[-1]["fingerprint"] == fingerprint:
            logger.info("Schema has not changed, no new version created")
            return self.get_version_string()
            
        # Calculate health metrics
        try:
            health_metrics = {
                "schema_health": schema_integration.calculate_schema_health(),
                "cognitive_debt": schema_integration.calculate_cognitive_debt(),
                "complexity_budget": schema_integration.calculate_complexity_budget(),
                "stats": schema_integration.calculate_schema_statistics()
            }
        except:
            health_metrics = {}
            
        # Increment version
        new_version = self.current_version.copy()
        if version_type == "major":
            new_version["major"] += 1
            new_version["minor"] = 0
            new_version["patch"] = 0
        elif version_type == "minor":
            new_version["minor"] += 1
            new_version["patch"] = 0
        else:  # patch
            new_version["patch"] += 1
            
        # Create version entry
        version_entry = {
            "version": new_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "fingerprint": fingerprint,
            "message": commit_message,
            "health_metrics": health_metrics
        }
        
        # Add to version history
        self.version_history.append(version_entry)
        self.current_version = new_version
        
        # Track health metrics
        self.health_metrics.append({
            "version": self.get_version_string(),
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": health_metrics
        })
        
        # Save version history
        self._save_version_history()
        
        # Save schema snapshot
        self._save_schema_snapshot(schema_graph, self.get_version_string())
        
        return self.get_version_string()
    
    def _save_schema_snapshot(self, schema_graph, version_str):
        """
        Save a schema snapshot for the given version.
        
        Args:
            schema_graph: NetworkX graph representing the schema
            version_str: Version string
        """
        snapshot_path = os.path.join(self.schema_dir, f"schema_v{version_str}.json")
        
        # Convert graph to serializable format
        nodes = []
        for node, attrs in schema_graph.nodes(data=True):
            node_data = {"id": node}
            for key, value in attrs.items():
                # Skip large objects like embeddings
                if key == "embedding":
                    continue
                node_data[key] = value
            nodes.append(node_data)
            
        edges = []
        for source, target, attrs in schema_graph.edges(data=True):
            edge_data = {"source": source, "target": target}
            for key, value in attrs.items():
                edge_data[key] = value
            edges.append(edge_data)
            
        # Create snapshot data
        snapshot = {
            "version": version_str,
            "timestamp": datetime.datetime.now().isoformat(),
            "nodes": nodes,
            "edges": edges
        }
        
        # Save to file
        try:
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving schema snapshot: {e}")
    
    def get_version_string(self):
        """
        Get the current version as a string.
        
        Returns:
            str: Version string
        """
        return f"{self.current_version['major']}.{self.current_version['minor']}.{self.current_version['patch']}"
    
    def load_schema_version(self, version_str=None):
        """
        Load a schema snapshot for the given version.
        
        Args:
            version_str: Version string (defaults to current version)
            
        Returns:
            dict: Schema snapshot data
        """
        if version_str is None:
            version_str = self.get_version_string()
            
        snapshot_path = os.path.join(self.schema_dir, f"schema_v{version_str}.json")
        
        if not os.path.exists(snapshot_path):
            logger.error(f"Schema snapshot for version {version_str} not found")
            return None
            
        try:
            with open(snapshot_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading schema snapshot: {e}")
            return None
    
    def create_migration(self, from_version, to_version):
        """
        Create a migration script between schema versions.
        
        Args:
            from_version: Source version string
            to_version: Target version string
            
        Returns:
            dict: Migration steps
        """
        # Load schema snapshots
        source_schema = self.load_schema_version(from_version)
        target_schema = self.load_schema_version(to_version)
        
        if not source_schema or not target_schema:
            return {"status": "error", "message": "Schema versions not found"}
            
        # Create source and target graphs
        source_graph = nx.Graph()
        target_graph = nx.Graph()
        
        # Add nodes to source graph
        for node in source_schema.get("nodes", []):
            node_id = node.pop("id")
            source_graph.add_node(node_id, **node)
            
        # Add edges to source graph
        for edge in source_schema.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            source_graph.add_edge(source, target, **edge)
            
        # Add nodes to target graph
        for node in target_schema.get("nodes", []):
            node_id = node.pop("id")
            target_graph.add_node(node_id, **node)
            
        # Add edges to target graph
        for edge in target_schema.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            target_graph.add_edge(source, target, **edge)
            
        # Calculate differences
        source_nodes = set(source_graph.nodes())
        target_nodes = set(target_graph.nodes())
        
        added_nodes = target_nodes - source_nodes
        removed_nodes = source_nodes - target_nodes
        common_nodes = source_nodes.intersection(target_nodes)
        
        source_edges = set(source_graph.edges())
        target_edges = set(target_graph.edges())
        
        added_edges = target_edges - source_edges
        removed_edges = source_edges - target_edges
        
        # Create migration steps
        migration = {
            "from_version": from_version,
            "to_version": to_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "steps": []
        }
        
        # Add node removal steps
        for node in removed_nodes:
            migration["steps"].append({
                "action": "remove_node",
                "node_id": node
            })
            
        # Add node addition steps
        for node in added_nodes:
            migration["steps"].append({
                "action": "add_node",
                "node_id": node,
                "attributes": target_graph.nodes[node]
            })
            
        # Add node modification steps
        for node in common_nodes:
            source_attrs = source_graph.nodes[node]
            target_attrs = target_graph.nodes[node]
            
            # Check if attributes have changed
            if source_attrs != target_attrs:
                migration["steps"].append({
                    "action": "update_node",
                    "node_id": node,
                    "attributes": target_attrs
                })
                
        # Add edge removal steps
        for source, target in removed_edges:
            migration["steps"].append({
                "action": "remove_edge",
                "source": source,
                "target": target
            })
            
        # Add edge addition steps
        for source, target in added_edges:
            migration["steps"].append({
                "action": "add_edge",
                "source": source,
                "target": target,
                "attributes": target_graph.edges[source, target]
            })
            
        return migration
    
    def get_health_trend(self, metric_name, window=10):
        """
        Get trend data for a health metric.
        
        Args:
            metric_name: Name of the health metric
            window: Number of versions to include
            
        Returns:
            dict: Trend data for the metric
        """
        if not self.health_metrics:
            return {"status": "no_data"}
            
        # Get metrics for the window
        metrics = self.health_metrics[-window:]
        
        # Extract metric values
        values = []
        versions = []
        timestamps = []
        
        for entry in metrics:
            metric_value = None
            
            # Handle nested metrics
            if "." in metric_name:
                parts = metric_name.split(".")
                current = entry.get("metrics", {})
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                metric_value = current
            else:
                metric_value = entry.get("metrics", {}).get(metric_name)
                
            if metric_value is not None:
                values.append(metric_value)
                versions.append(entry.get("version"))
                timestamps.append(entry.get("timestamp"))
                
        if not values:
            return {"status": "no_data_for_metric"}
            
        # Calculate trend statistics
        avg_value = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)
        
        # Calculate trend direction
        trend = "stable"
        if len(values) > 1:
            if values[-1] > values[0]:
                trend = "increasing"
            elif values[-1] < values[0]:
                trend = "decreasing"
                
        return {
            "status": "success",
            "metric": metric_name,
            "versions": versions,
            "values": values,
            "timestamps": timestamps,
            "statistics": {
                "average": avg_value,
                "minimum": min_value,
                "maximum": max_value,
                "current": values[-1] if values else None,
                "trend": trend
            }
        }
    
    def get_deployment_readiness(self):
        """
        Assess schema deployment readiness.
        
        Returns:
            dict: Deployment readiness assessment
        """
        if not self.version_history:
            return {
                "status": "not_ready",
                "message": "No schema versions available",
                "readiness_score": 0.0
            }
            
        # Get latest health metrics
        latest_metrics = self.health_metrics[-1]["metrics"] if self.health_metrics else {}
        
        # Calculate readiness score
        readiness_score = 0.0
        checklist = []
        
        # Check schema health
        schema_health = latest_metrics.get("schema_health", 0)
        if schema_health > 0.7:
            readiness_score += 0.3
            checklist.append({"check": "schema_health", "status": "pass", "value": schema_health})
        else:
            checklist.append({"check": "schema_health", "status": "fail", "value": schema_health})
            
        # Check cognitive debt
        cognitive_debt = latest_metrics.get("cognitive_debt", 10)
        if cognitive_debt < 5.0:
            readiness_score += 0.2
            checklist.append({"check": "cognitive_debt", "status": "pass", "value": cognitive_debt})
        else:
            checklist.append({"check": "cognitive_debt", "status": "fail", "value": cognitive_debt})
            
        # Check schema stats
        stats = latest_metrics.get("stats", {})
        
        # Check node count
        node_count = stats.get("node_count", 0)
        if node_count > 0:
            readiness_score += 0.1
            checklist.append({"check": "node_count", "status": "pass", "value": node_count})
        else:
            checklist.append({"check": "node_count", "status": "fail", "value": node_count})
            
        # Check edge count
        edge_count = stats.get("edge_count", 0)
        if edge_count > 0:
            readiness_score += 0.1
            checklist.append({"check": "edge_count", "status": "pass", "value": edge_count})
        else:
            checklist.append({"check": "edge_count", "status": "fail", "value": edge_count})
            
        # Check connectivity
        connectivity = stats.get("connectivity", {})
        avg_degree = connectivity.get("average_degree", 0)
        if avg_degree > 1.0:
            readiness_score += 0.1
            checklist.append({"check": "average_degree", "status": "pass", "value": avg_degree})
        else:
            checklist.append({"check": "average_degree", "status": "fail", "value": avg_degree})
            
        # Check version count
        if len(self.version_history) > 2:
            readiness_score += 0.1
            checklist.append({"check": "version_count", "status": "pass", "value": len(self.version_history)})
        else:
            checklist.append({"check": "version_count", "status": "fail", "value": len(self.version_history)})
            
        # Check version stability
        if len(self.version_history) >= 2:
            latest_fingerprint = self.version_history[-1]["fingerprint"]
            previous_fingerprint = self.version_history[-2]["fingerprint"]
            
            if latest_fingerprint != previous_fingerprint:
                # Schema is still changing
                readiness_score -= 0.1
                checklist.append({"check": "schema_stability", "status": "fail", "value": "changing"})
            else:
                readiness_score += 0.1
                checklist.append({"check": "schema_stability", "status": "pass", "value": "stable"})
        
        # Determine overall readiness
        status = "not_ready"
        message = "Schema is not ready for deployment"
        
        if readiness_score >= 0.8:
            status = "ready"
            message = "Schema is ready for deployment"
        elif readiness_score >= 0.6:
            status = "partially_ready"
            message = "Schema is partially ready, but some checks failed"
            
        return {
            "status": status,
            "message": message,
            "readiness_score": readiness_score,
            "version": self.get_version_string(),
            "checklist": checklist
        }
    
    def create_deployment_report(self, output_path=None):
        """
        Create a comprehensive deployment report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            dict: Deployment report
        """
        # Generate report data
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "schema_version": self.get_version_string(),
            "version_history": self.version_history[-5:],  # Last 5 versions
            "readiness": self.get_deployment_readiness(),
            "health_trends": {
                "schema_health": self.get_health_trend("schema_health"),
                "cognitive_debt": self.get_health_trend("cognitive_debt"),
                "complexity_budget": self.get_health_trend("complexity_budget")
            }
        }
        
        # Save report to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report


class SchemaDeploymentManager:
    """
    Schema Deployment Manager for ensuring schema readiness and deployment.
    
    This class provides tools for managing schema deployment across
    environments, validating schema health, and monitoring schema
    performance in production.
    
    Attributes:
        version_manager: Schema version manager
        environments: Dictionary of environments and their schema versions
    """
    
    def __init__(self, version_manager):
        """
        Initialize the schema deployment manager.
        
        Args:
            version_manager: SchemaVersionManager instance
        """
        self.version_manager = version_manager
        self.environments = {
            "development": version_manager.get_version_string(),
            "staging": None,
            "production": None
        }
        
        # Load environment data if available
        self._load_environments()
        
    def _load_environments(self):
        """Load environment data from disk."""
        env_path = os.path.join(self.version_manager.schema_dir, "environments.json")
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    self.environments = json.load(f)
            except Exception as e:
                logger.error(f"Error loading environment data: {e}")
                
    def _save_environments(self):
        """Save environment data to disk."""
        env_path = os.path.join(self.version_manager.schema_dir, "environments.json")
        try:
            with open(env_path, 'w') as f:
                json.dump(self.environments, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving environment data: {e}")
            
    def deploy_to_environment(self, environment, version=None):
        """
        Deploy a schema version to an environment.
        
        Args:
            environment: Target environment (staging, production)
            version: Schema version to deploy (defaults to current version)
            
        Returns:
            dict: Deployment result
        """
        if environment not in ["staging", "production"]:
            return {"status": "error", "message": f"Invalid environment: {environment}"}
            
        if version is None:
            version = self.version_manager.get_version_string()
            
        # Check if version exists
        schema_data = self.version_manager.load_schema_version(version)
        if not schema_data:
            return {"status": "error", "message": f"Version {version} not found"}
            
        # Check deployment readiness for production
        if environment == "production":
            readiness = self.version_manager.get_deployment_readiness()
            if readiness["status"] != "ready":
                return {
                    "status": "error",
                    "message": f"Schema not ready for production deployment. Readiness: {readiness['readiness_score']}",
                    "readiness": readiness
                }
                
        # Update environment version
        self.environments[environment] = version
        self._save_environments()
        
        return {
            "status": "success",
            "message": f"Deployed version {version} to {environment}",
            "environment": environment,
            "version": version
        }
    
    def get_environment_status(self):
        """
        Get status of all environments.
        
        Returns:
            dict: Environment status
        """
        return {
            "environments": self.environments,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def create_deployment_plan(self, target_version=None):
        """
        Create a deployment plan for rolling out schema changes.
        
        Args:
            target_version: Target schema version (defaults to current version)
            
        Returns:
            dict: Deployment plan
        """
        if target_version is None:
            target_version = self.version_manager.get_version_string()
            
        # Check if version exists
        schema_data = self.version_manager.load_schema_version(target_version)
        if not schema_data:
            return {"status": "error", "message": f"Version {target_version} not found"}
            
        # Get current versions in staging and production
        staging_version = self.environments.get("staging")
        production_version = self.environments.get("production")
        
        # Create migration plans
        migrations = {}
        
        if staging_version and staging_version != target_version:
            migrations["staging"] = self.version_manager.create_migration(staging_version, target_version)
            
        if production_version and production_version != target_version:
            migrations["production"] = self.version_manager.create_migration(production_version, target_version)
            
        # Get deployment readiness
        readiness = self.version_manager.get_deployment_readiness()
        
        # Create deployment plan
        plan = {
            "target_version": target_version,
            "current_environments": self.environments,
            "readiness": readiness,
            "migrations": migrations,
            "steps": []
        }
        
        # Generate deployment steps
        if staging_version != target_version:
            plan["steps"].append({
                "action": "deploy_to_staging",
                "current_version": staging_version,
                "target_version": target_version,
                "migration_size": len(migrations.get("staging", {}).get("steps", []))
            })
            
        if production_version != target_version:
            # Only recommend production deployment if schema is ready
            if readiness["status"] == "ready":
                plan["steps"].append({
                    "action": "deploy_to_production",
                    "current_version": production_version,
                    "target_version": target_version,
                    "migration_size": len(migrations.get("production", {}).get("steps", []))
                })
            else:
                plan["steps"].append({
                    "action": "improve_readiness",
                    "current_score": readiness["readiness_score"],
                    "target_score": 0.8,
                    "failing_checks": [c for c in readiness["checklist"] if c["status"] == "fail"]
                })
                
        return plan 