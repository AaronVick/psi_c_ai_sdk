"""
Runtime Policy Layer for ΨC-AI SDK

This module provides a policy management system to toggle agent behaviors
dynamically, enabling adaptation to safety constraints, research scenarios,
and external requirements. The policy layer acts as a global control system
that can modify how the ΨC agent operates across various dimensions.
"""

import logging
import json
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PolicyMode(Enum):
    """Operating modes for the ΨC agent system."""
    
    NORMAL = auto()              # Standard operation with all features
    SAFE_MODE = auto()           # Restricted operation with safety limits
    SHADOWBOX = auto()           # Enables adversarial inputs and testing
    INTERACTIVE_REFLECTION = auto()  # Allows external questions to trigger reflection
    NO_EXTERNAL_INPUT = auto()   # Isolates the agent's memory loop
    ISOLATED = auto()            # Full isolation, no external communication
    PASSIVE = auto()             # Read-only mode, no memory modification
    EXPERIMENTAL = auto()        # Enables experimental features


@dataclass
class PolicyConstraint:
    """A specific constraint applied to agent behavior."""
    
    name: str
    description: str
    enabled: bool = True
    severity: float = 1.0  # 0.0 to 1.0, how strictly to enforce
    parameters: Dict[str, Any] = field(default_factory=dict)


class RuntimePolicy:
    """
    Policy management system for ΨC agent runtime behavior.
    
    This class maintains a set of policies that control how the agent
    operates, including safety constraints, interaction patterns, and
    resource usage limits.
    """
    
    def __init__(
        self,
        mode: PolicyMode = PolicyMode.NORMAL,
        config_path: Optional[str] = None
    ):
        """
        Initialize the runtime policy system.
        
        Args:
            mode: Initial policy mode
            config_path: Optional path to policy configuration file
        """
        self.current_mode = mode
        self.active_constraints: Dict[str, PolicyConstraint] = {}
        self.policy_history: List[Dict[str, Any]] = []
        
        # Boundaries that can be enforced
        self.available_constraints = {
            # Input/output constraints
            "restrict_external_data": PolicyConstraint(
                name="restrict_external_data",
                description="Limits or blocks external data sources",
                enabled=False
            ),
            "block_schema_merging": PolicyConstraint(
                name="block_schema_merging",
                description="Prevents merging of external schemas",
                enabled=False
            ),
            
            # Reflection constraints
            "limit_reflection_depth": PolicyConstraint(
                name="limit_reflection_depth",
                description="Caps recursive reflection depth",
                enabled=False,
                parameters={"max_depth": 3}
            ),
            "throttle_reflection_rate": PolicyConstraint(
                name="throttle_reflection_rate",
                description="Limits reflection frequency",
                enabled=False,
                parameters={"min_interval_seconds": 5.0}
            ),
            
            # Schema constraints
            "prevent_axiom_changes": PolicyConstraint(
                name="prevent_axiom_changes",
                description="Prevents modifications to core axioms",
                enabled=False
            ),
            "entropy_shutdown": PolicyConstraint(
                name="entropy_shutdown",
                description="Forces shutdown at high entropy levels",
                enabled=False,
                parameters={"threshold": 0.9}
            ),
            
            # AGI safety constraints
            "agi_boundary_enforcement": PolicyConstraint(
                name="agi_boundary_enforcement",
                description="Enforces separation from AGI systems",
                enabled=False,
                parameters={"min_distance": 0.7}
            ),
            "external_influence_throttling": PolicyConstraint(
                name="external_influence_throttling",
                description="Limits rate of external influence on beliefs",
                enabled=False,
                parameters={"max_influence_rate": 0.2}
            ),
            
            # Resource constraints
            "memory_limit": PolicyConstraint(
                name="memory_limit",
                description="Limits total memory usage",
                enabled=False,
                parameters={"max_memories": 1000}
            ),
            "compute_budget": PolicyConstraint(
                name="compute_budget",
                description="Limits computational resources",
                enabled=False,
                parameters={"max_operations_per_minute": 1000}
            )
        }
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Apply initial mode
        self.set_mode(mode)
    
    def set_mode(self, mode: PolicyMode) -> None:
        """
        Switch to a different policy mode.
        
        Args:
            mode: New policy mode to apply
        """
        old_mode = self.current_mode
        self.current_mode = mode
        
        # Reset all constraints
        for constraint in self.available_constraints.values():
            constraint.enabled = False
        
        # Apply constraints based on mode
        if mode == PolicyMode.SAFE_MODE:
            self._apply_safe_mode()
        elif mode == PolicyMode.SHADOWBOX:
            self._apply_shadowbox_mode()
        elif mode == PolicyMode.INTERACTIVE_REFLECTION:
            self._apply_interactive_reflection_mode()
        elif mode == PolicyMode.NO_EXTERNAL_INPUT:
            self._apply_no_external_input_mode()
        elif mode == PolicyMode.ISOLATED:
            self._apply_isolated_mode()
        elif mode == PolicyMode.PASSIVE:
            self._apply_passive_mode()
        elif mode == PolicyMode.EXPERIMENTAL:
            self._apply_experimental_mode()
        
        # Record policy change
        self.policy_history.append({
            "timestamp": import_module("time").time(),
            "old_mode": old_mode.name,
            "new_mode": mode.name
        })
        
        logger.info(f"Policy mode changed: {old_mode.name} -> {mode.name}")
    
    def enable_constraint(self, constraint_name: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Enable a specific constraint.
        
        Args:
            constraint_name: Name of constraint to enable
            parameters: Optional parameters to override defaults
            
        Returns:
            Whether constraint was successfully enabled
        """
        if constraint_name not in self.available_constraints:
            logger.warning(f"Unknown constraint: {constraint_name}")
            return False
        
        constraint = self.available_constraints[constraint_name]
        constraint.enabled = True
        
        # Update parameters if provided
        if parameters:
            constraint.parameters.update(parameters)
        
        # Add to active constraints
        self.active_constraints[constraint_name] = constraint
        
        logger.info(f"Enabled constraint: {constraint_name}")
        return True
    
    def disable_constraint(self, constraint_name: str) -> bool:
        """
        Disable a specific constraint.
        
        Args:
            constraint_name: Name of constraint to disable
            
        Returns:
            Whether constraint was successfully disabled
        """
        if constraint_name not in self.available_constraints:
            logger.warning(f"Unknown constraint: {constraint_name}")
            return False
        
        constraint = self.available_constraints[constraint_name]
        constraint.enabled = False
        
        # Remove from active constraints
        if constraint_name in self.active_constraints:
            del self.active_constraints[constraint_name]
        
        logger.info(f"Disabled constraint: {constraint_name}")
        return True
    
    def is_allowed(self, action: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if an action is allowed under current policy.
        
        Args:
            action: Action to check
            context: Optional context for policy decision
            
        Returns:
            Whether action is allowed
        """
        context = context or {}
        
        # Map actions to constraints
        action_constraints = {
            "external_data_access": ["restrict_external_data"],
            "schema_merge": ["block_schema_merging"],
            "reflection": ["limit_reflection_depth", "throttle_reflection_rate"],
            "axiom_modification": ["prevent_axiom_changes"],
            "memory_creation": ["memory_limit"],
            "belief_update": ["external_influence_throttling"],
            "agi_interaction": ["agi_boundary_enforcement"]
        }
        
        # Check if action has any active constraints
        if action in action_constraints:
            relevant_constraints = action_constraints[action]
            for constraint_name in relevant_constraints:
                constraint = self.available_constraints.get(constraint_name)
                if constraint and constraint.enabled:
                    # Specific constraint logic could be implemented here
                    # For now, just check if enabled
                    return False
        
        return True
    
    def is_constraint_active(self, constraint_name: str) -> bool:
        """
        Check if a specific constraint is active.
        
        Args:
            constraint_name: Name of constraint to check
            
        Returns:
            Whether constraint is active
        """
        if constraint_name not in self.available_constraints:
            return False
        
        return self.available_constraints[constraint_name].enabled
    
    def get_active_constraints(self) -> Dict[str, PolicyConstraint]:
        """
        Get all currently active constraints.
        
        Returns:
            Dictionary of active constraints
        """
        return {
            name: constraint for name, constraint in self.available_constraints.items()
            if constraint.enabled
        }
    
    def load_config(self, config_path: str) -> bool:
        """
        Load policy configuration from a file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Whether configuration was loaded successfully
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply mode if specified
            if "mode" in config:
                try:
                    mode = PolicyMode[config["mode"]]
                    self.set_mode(mode)
                except (KeyError, ValueError):
                    logger.warning(f"Invalid policy mode in config: {config['mode']}")
            
            # Apply individual constraints
            if "constraints" in config:
                for constraint_config in config["constraints"]:
                    name = constraint_config.get("name")
                    if not name or name not in self.available_constraints:
                        logger.warning(f"Unknown constraint in config: {name}")
                        continue
                    
                    enabled = constraint_config.get("enabled", False)
                    parameters = constraint_config.get("parameters", {})
                    
                    if enabled:
                        self.enable_constraint(name, parameters)
                    else:
                        self.disable_constraint(name)
            
            logger.info(f"Loaded policy configuration from {config_path}")
            return True
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load policy configuration: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Save current policy configuration to a file.
        
        Args:
            config_path: Path to save configuration file
            
        Returns:
            Whether configuration was saved successfully
        """
        config = {
            "mode": self.current_mode.name,
            "constraints": []
        }
        
        # Add all constraints
        for name, constraint in self.available_constraints.items():
            config["constraints"].append({
                "name": name,
                "enabled": constraint.enabled,
                "parameters": constraint.parameters
            })
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved policy configuration to {config_path}")
            return True
            
        except IOError as e:
            logger.error(f"Failed to save policy configuration: {e}")
            return False
    
    def _apply_safe_mode(self) -> None:
        """Apply Safe Mode constraints."""
        self.enable_constraint("restrict_external_data")
        self.enable_constraint("block_schema_merging")
        self.enable_constraint("prevent_axiom_changes")
        self.enable_constraint("entropy_shutdown")
        self.enable_constraint("agi_boundary_enforcement")
        self.enable_constraint("external_influence_throttling")
    
    def _apply_shadowbox_mode(self) -> None:
        """Apply Shadowbox Mode constraints."""
        # Allow adversarial inputs but maintain some safety
        self.enable_constraint("prevent_axiom_changes")
        self.enable_constraint("entropy_shutdown", {"threshold": 0.95})  # Higher threshold
    
    def _apply_interactive_reflection_mode(self) -> None:
        """Apply Interactive Reflection Mode constraints."""
        # Focus on reflection capabilities
        self.enable_constraint("throttle_reflection_rate", {"min_interval_seconds": 1.0})
        self.enable_constraint("prevent_axiom_changes")
    
    def _apply_no_external_input_mode(self) -> None:
        """Apply No External Input Mode constraints."""
        self.enable_constraint("restrict_external_data")
        self.enable_constraint("block_schema_merging")
    
    def _apply_isolated_mode(self) -> None:
        """Apply Isolated Mode constraints."""
        # Complete isolation
        self.enable_constraint("restrict_external_data")
        self.enable_constraint("block_schema_merging")
        self.enable_constraint("prevent_axiom_changes")
        self.enable_constraint("agi_boundary_enforcement", {"min_distance": 1.0})  # Maximum distance
    
    def _apply_passive_mode(self) -> None:
        """Apply Passive Mode constraints."""
        # Read-only mode
        self.enable_constraint("memory_limit", {"max_memories": 0})  # No new memories
        self.enable_constraint("prevent_axiom_changes")
        self.enable_constraint("block_schema_merging")
    
    def _apply_experimental_mode(self) -> None:
        """Apply Experimental Mode constraints."""
        # Minimal constraints for maximum flexibility
        self.enable_constraint("entropy_shutdown", {"threshold": 0.99})  # Emergency shutdown only


# Module-level instance for global policy access
from importlib import import_module
global_policy = RuntimePolicy() 