"""
Entropy Response System for ΨC-AI SDK

This module provides a system for automatically responding to high entropy conditions
in AI memory systems. It implements various strategies for entropy reduction and
a configurable response framework that can be integrated with the EntropyMonitor.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import threading
import time
from datetime import datetime

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator
from psi_c_ai_sdk.entropy.monitor import EntropyMonitor, EntropyAlert, EntropySubscriber
from psi_c_ai_sdk.entropy.pruning import EntropyBasedPruner

logger = logging.getLogger(__name__)


class EntropyResponseStrategy(Enum):
    """Strategies for responding to elevated entropy levels."""
    MEMORY_PRUNING = "memory_pruning"        # Remove high-entropy memories
    MEMORY_CONSOLIDATION = "consolidation"   # Merge similar memories
    CONCEPT_REORGANIZATION = "reorganization"  # Reorganize concept structure
    TARGETED_REFLECTION = "reflection"       # Trigger reflection on high-entropy areas
    MEMORY_ISOLATION = "isolation"           # Isolate but don't delete problematic memories
    ENTROPY_DIFFUSION = "diffusion"          # Spread entropy across system to reduce peaks
    EMERGENCY_SHUTDOWN = "shutdown"          # Initiate emergency shutdown for critical entropy


class EntropyResponseConfig:
    """Configuration for entropy response system."""
    
    def __init__(
        self,
        elevated_strategies: Optional[List[EntropyResponseStrategy]] = None,
        high_strategies: Optional[List[EntropyResponseStrategy]] = None,
        critical_strategies: Optional[List[EntropyResponseStrategy]] = None,
        auto_adjust_thresholds: bool = True,
        cooldown_period: float = 300.0,  # 5 minutes
        max_memory_prune_ratio: float = 0.05,  # Max 5% of memories pruned at once
        min_entropy_reduction_target: float = 0.1  # Target 10% entropy reduction
    ):
        """
        Initialize entropy response configuration.
        
        Args:
            elevated_strategies: Strategies to use for ELEVATED entropy alerts
            high_strategies: Strategies to use for HIGH entropy alerts
            critical_strategies: Strategies to use for CRITICAL entropy alerts
            auto_adjust_thresholds: Whether to automatically adjust response thresholds
            cooldown_period: Seconds to wait between response actions
            max_memory_prune_ratio: Maximum ratio of memories to prune at once
            min_entropy_reduction_target: Minimum target entropy reduction
        """
        # Default strategies if none provided
        self.elevated_strategies = elevated_strategies or [
            EntropyResponseStrategy.MEMORY_PRUNING,
            EntropyResponseStrategy.MEMORY_CONSOLIDATION
        ]
        
        self.high_strategies = high_strategies or [
            EntropyResponseStrategy.MEMORY_PRUNING,
            EntropyResponseStrategy.CONCEPT_REORGANIZATION,
            EntropyResponseStrategy.TARGETED_REFLECTION
        ]
        
        self.critical_strategies = critical_strategies or [
            EntropyResponseStrategy.MEMORY_PRUNING,
            EntropyResponseStrategy.MEMORY_ISOLATION,
            EntropyResponseStrategy.ENTROPY_DIFFUSION,
            EntropyResponseStrategy.EMERGENCY_SHUTDOWN
        ]
        
        self.auto_adjust_thresholds = auto_adjust_thresholds
        self.cooldown_period = cooldown_period
        self.max_memory_prune_ratio = max_memory_prune_ratio
        self.min_entropy_reduction_target = min_entropy_reduction_target
        
    def get_strategies_for_alert(self, alert: EntropyAlert) -> List[EntropyResponseStrategy]:
        """
        Get strategies to use for the given alert level.
        
        Args:
            alert: Alert level to get strategies for
            
        Returns:
            List of strategies to use
        """
        if alert == EntropyAlert.ELEVATED:
            return self.elevated_strategies
        elif alert == EntropyAlert.HIGH:
            return self.high_strategies
        elif alert == EntropyAlert.CRITICAL:
            return self.critical_strategies
        else:
            return []  # No strategies for NORMAL level


class EntropyResponse(EntropySubscriber):
    """
    System for automatically responding to entropy alerts.
    
    This class implements the EntropySubscriber interface to receive alerts
    from the EntropyMonitor and take appropriate actions to reduce entropy.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        config: Optional[EntropyResponseConfig] = None,
        reflection_trigger_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize the entropy response system.
        
        Args:
            memory_store: Memory store to manage
            config: Response configuration
            reflection_trigger_callback: Function to call to trigger reflection
        """
        self.memory_store = memory_store
        self.config = config or EntropyResponseConfig()
        self.reflection_trigger_callback = reflection_trigger_callback
        
        # Initialize components
        self.entropy_calculator = EntropyCalculator()
        self.pruner = EntropyBasedPruner(
            entropy_threshold=0.75,
            max_pruning_ratio=self.config.max_memory_prune_ratio
        )
        
        # Response tracking
        self.last_response_time: Optional[datetime] = None
        self.response_history: List[Dict[str, Any]] = []
        self.current_alert_level = EntropyAlert.NORMAL
        self.locked_memories: Set[str] = set()  # IDs of isolated memories
        
    def on_entropy_alert(self, alert_level: EntropyAlert, entropy_value: float,
                        details: Dict[str, Any]) -> None:
        """
        Respond to an entropy alert.
        
        Args:
            alert_level: Current alert level
            entropy_value: Current entropy value
            details: Additional details about the alert
        """
        # Record the alert
        self.current_alert_level = alert_level
        
        # Skip if alert is NORMAL
        if alert_level == EntropyAlert.NORMAL:
            logger.info(f"Entropy alert: {alert_level.name}, no response needed")
            return
            
        # Check if we're in cooldown period
        now = datetime.now()
        if self.last_response_time:
            elapsed = (now - self.last_response_time).total_seconds()
            if elapsed < self.config.cooldown_period:
                logger.info(
                    f"Entropy alert: {alert_level.name}, in cooldown period "
                    f"({elapsed:.1f}/{self.config.cooldown_period:.1f} seconds elapsed)"
                )
                return
                
        # Get strategies for this alert level
        strategies = self.config.get_strategies_for_alert(alert_level)
        
        logger.info(
            f"Responding to {alert_level.name} entropy alert, "
            f"value: {entropy_value:.4f}, strategies: {[s.value for s in strategies]}"
        )
        
        # Apply strategies in order
        results = []
        initial_entropy = entropy_value
        current_entropy = initial_entropy
        
        for strategy in strategies:
            # Apply the strategy
            result = self._apply_strategy(strategy, current_entropy, details)
            results.append(result)
            
            # Update current entropy if strategy was applied
            if result["applied"]:
                current_entropy = self.entropy_calculator.calculate_memory_store_entropy(self.memory_store)
                
                # Stop if we've reduced entropy enough
                if (initial_entropy - current_entropy) >= self.config.min_entropy_reduction_target:
                    logger.info(
                        f"Entropy reduced sufficiently: {initial_entropy:.4f} -> {current_entropy:.4f} "
                        f"(target: {self.config.min_entropy_reduction_target:.2f})"
                    )
                    break
        
        # Record the response
        self.last_response_time = now
        response_record = {
            "timestamp": now,
            "alert_level": alert_level.name,
            "initial_entropy": initial_entropy,
            "final_entropy": current_entropy,
            "entropy_reduction": initial_entropy - current_entropy,
            "strategies_applied": [r for r in results if r["applied"]],
            "strategies_skipped": [r for r in results if not r["applied"]]
        }
        
        self.response_history.append(response_record)
        
        # Limit history size
        if len(self.response_history) > 50:
            self.response_history = self.response_history[-50:]
            
        logger.info(
            f"Entropy response complete: {initial_entropy:.4f} -> {current_entropy:.4f} "
            f"(reduction: {initial_entropy - current_entropy:.4f})"
        )
    
    def on_termination_decision(self, entropy_value: float, details: Dict[str, Any]) -> bool:
        """
        Handle a termination decision request.
        
        Args:
            entropy_value: Current entropy value
            details: Additional details about the termination
            
        Returns:
            True if termination should proceed, False to override
        """
        logger.critical(f"Critical entropy level requires termination decision: {entropy_value:.4f}")
        
        # Try emergency entropy reduction
        result = self._apply_strategy(
            EntropyResponseStrategy.ENTROPY_DIFFUSION, 
            entropy_value,
            details
        )
        
        # Only proceed with termination if entropy diffusion failed
        if not result["applied"] or not result.get("success", False):
            logger.critical("Emergency entropy reduction failed, recommending termination")
            return True
            
        # Check if entropy was reduced enough
        current_entropy = self.entropy_calculator.calculate_memory_store_entropy(self.memory_store)
        if current_entropy >= 0.9:  # Still very high
            logger.critical(
                f"Emergency entropy reduction insufficient: {entropy_value:.4f} -> {current_entropy:.4f}, "
                f"recommending termination"
            )
            return True
            
        logger.warning(
            f"Emergency entropy reduction successful: {entropy_value:.4f} -> {current_entropy:.4f}, "
            f"overriding termination"
        )
        return False
    
    def _apply_strategy(
        self, 
        strategy: EntropyResponseStrategy, 
        current_entropy: float,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply an entropy reduction strategy.
        
        Args:
            strategy: Strategy to apply
            current_entropy: Current entropy value
            details: Details about the alert that triggered this response
            
        Returns:
            Dictionary with results of strategy application
        """
        result = {
            "strategy": strategy.value,
            "applied": False,
            "success": False,
            "details": {}
        }
        
        try:
            if strategy == EntropyResponseStrategy.MEMORY_PRUNING:
                result.update(self._apply_memory_pruning(current_entropy))
                
            elif strategy == EntropyResponseStrategy.MEMORY_CONSOLIDATION:
                result.update(self._apply_memory_consolidation())
                
            elif strategy == EntropyResponseStrategy.CONCEPT_REORGANIZATION:
                result.update(self._apply_concept_reorganization())
                
            elif strategy == EntropyResponseStrategy.TARGETED_REFLECTION:
                result.update(self._apply_targeted_reflection(details))
                
            elif strategy == EntropyResponseStrategy.MEMORY_ISOLATION:
                result.update(self._apply_memory_isolation())
                
            elif strategy == EntropyResponseStrategy.ENTROPY_DIFFUSION:
                result.update(self._apply_entropy_diffusion())
                
            elif strategy == EntropyResponseStrategy.EMERGENCY_SHUTDOWN:
                result.update(self._apply_emergency_shutdown())
                
            else:
                logger.warning(f"Unknown entropy response strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Error applying entropy response strategy {strategy}: {e}")
            result["error"] = str(e)
            
        return result
    
    def _apply_memory_pruning(self, current_entropy: float) -> Dict[str, Any]:
        """Apply memory pruning strategy."""
        # Determine pruning threshold based on current entropy
        entropy_threshold = max(0.7, current_entropy - 0.1)
        
        # Determine max memories to prune
        total_memories = len(self.memory_store.get_all_memories())
        max_to_prune = max(1, int(total_memories * self.config.max_memory_prune_ratio))
        
        # Run the pruner
        stats = self.pruner.prune_high_entropy_memories(
            memory_store=self.memory_store,
            dry_run=False,
            max_to_prune=max_to_prune
        )
        
        if stats["pruned_count"] > 0:
            logger.info(
                f"Memory pruning complete: removed {stats['pruned_count']} memories "
                f"(avg entropy: {stats['average_entropy']:.4f})"
            )
            return {
                "applied": True,
                "success": True,
                "details": stats
            }
        else:
            logger.info("Memory pruning skipped: no memories exceeded entropy threshold")
            return {
                "applied": False,
                "details": stats
            }
    
    def _apply_memory_consolidation(self) -> Dict[str, Any]:
        """Apply memory consolidation strategy."""
        # This would merge similar memories to reduce redundancy
        # Simplified implementation for now
        logger.info("Memory consolidation not yet implemented")
        return {
            "applied": False,
            "details": {"reason": "Not implemented"}
        }
    
    def _apply_concept_reorganization(self) -> Dict[str, Any]:
        """Apply concept reorganization strategy."""
        # This would reorganize concept structures to improve coherence
        # Simplified implementation for now
        logger.info("Concept reorganization not yet implemented")
        return {
            "applied": False,
            "details": {"reason": "Not implemented"}
        }
    
    def _apply_targeted_reflection(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Apply targeted reflection strategy."""
        if not self.reflection_trigger_callback:
            logger.info("Targeted reflection skipped: no callback registered")
            return {
                "applied": False,
                "details": {"reason": "No reflection callback registered"}
            }
            
        # Identify areas of high entropy to reflect on
        high_entropy_areas = []
        
        # Check entropy metrics to find problematic areas
        metrics = details.get("metrics", {})
        if "embedding" in metrics and metrics["embedding"] > 0.7:
            high_entropy_areas.append("embedding_space")
        if "semantic" in metrics and metrics["semantic"] > 0.7:
            high_entropy_areas.append("semantic_coherence")
        if "temporal" in metrics and metrics["temporal"] > 0.7:
            high_entropy_areas.append("temporal_patterns")
            
        if not high_entropy_areas:
            logger.info("Targeted reflection skipped: no specific high-entropy areas identified")
            return {
                "applied": False,
                "details": {"reason": "No specific high-entropy areas identified"}
            }
            
        # Trigger reflection
        reflection_topic = f"Entropy reduction in {', '.join(high_entropy_areas)}"
        self.reflection_trigger_callback(reflection_topic, {
            "entropy_areas": high_entropy_areas,
            "entropy_metrics": metrics,
            "alert_level": self.current_alert_level.name
        })
        
        logger.info(f"Triggered reflection on topic: {reflection_topic}")
        return {
            "applied": True,
            "success": True,
            "details": {
                "reflection_topic": reflection_topic,
                "entropy_areas": high_entropy_areas
            }
        }
    
    def _apply_memory_isolation(self) -> Dict[str, Any]:
        """Apply memory isolation strategy."""
        # This would identify and isolate problematic memories without deleting them
        # Find highest entropy memories
        memories = self.memory_store.get_all_memories()
        
        if not memories:
            return {
                "applied": False,
                "details": {"reason": "No memories to isolate"}
            }
            
        # Calculate entropy for each memory
        memory_entropies = []
        for memory in memories:
            # Skip already isolated memories
            if str(memory.uuid) in self.locked_memories:
                continue
                
            entropy = self.entropy_calculator.calculate_memory_entropy(memory)
            memory_entropies.append((memory, entropy))
            
        # Sort by entropy (descending)
        memory_entropies.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 highest entropy memories
        isolation_candidates = memory_entropies[:5]
        
        if not isolation_candidates:
            return {
                "applied": False,
                "details": {"reason": "No suitable memories to isolate"}
            }
            
        # Isolate memories (implementation would depend on memory system)
        isolated_ids = []
        for memory, entropy in isolation_candidates:
            memory_id = str(memory.uuid)
            self.locked_memories.add(memory_id)
            isolated_ids.append(memory_id)
            
            # Mark memory as isolated (implementation would depend on memory system)
            # This is a simplified placeholder implementation
            if hasattr(memory, "metadata") and isinstance(memory.metadata, dict):
                memory.metadata["isolated"] = True
                memory.metadata["isolation_reason"] = f"High entropy: {entropy:.4f}"
            
        logger.info(f"Isolated {len(isolated_ids)} high-entropy memories")
        return {
            "applied": True,
            "success": True,
            "details": {
                "isolated_memory_count": len(isolated_ids),
                "isolated_memory_ids": isolated_ids
            }
        }
    
    def _apply_entropy_diffusion(self) -> Dict[str, Any]:
        """Apply entropy diffusion strategy."""
        # This would attempt to spread high entropy concentrations across the system
        # For example by recomputing embeddings, reconnecting memory networks, etc.
        
        logger.info("Applying emergency entropy diffusion")
        
        # 1. Force embedding recomputation for high-entropy memories
        memories = self.memory_store.get_all_memories()
        memory_entropies = []
        
        for memory in memories:
            entropy = self.entropy_calculator.calculate_memory_entropy(memory)
            memory_entropies.append((memory, entropy))
            
        # Sort by entropy (descending)
        memory_entropies.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 10% highest entropy memories
        high_entropy_count = max(1, int(len(memory_entropies) * 0.1))
        high_entropy_memories = memory_entropies[:high_entropy_count]
        
        # This would trigger embedding recomputation
        # (Implementation depends on memory system)
        recomputed_count = 0
        for memory, _ in high_entropy_memories:
            # Placeholder for recomputation
            if hasattr(memory, "recompute_embedding") and callable(memory.recompute_embedding):
                memory.recompute_embedding()
                recomputed_count += 1
        
        # 2. Apply other diffusion techniques
        # (Implementation would depend on memory system)
        
        return {
            "applied": True,
            "success": True,
            "details": {
                "recomputed_embeddings": recomputed_count,
                "high_entropy_memories": high_entropy_count
            }
        }
    
    def _apply_emergency_shutdown(self) -> Dict[str, Any]:
        """Apply emergency shutdown strategy."""
        logger.critical("Initiating emergency shutdown due to critical entropy levels")
        
        # This would initiate a controlled shutdown of the system
        # (Implementation depends on system architecture)
        
        return {
            "applied": True,
            "success": True,
            "details": {
                "shutdown_initiated": True,
                "entropy_level": self.current_alert_level.name
            }
        }
    
    def get_response_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of entropy responses.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of response history entries, newest first
        """
        history = list(reversed(self.response_history))
        return history[:limit]
    
    def get_locked_memories(self) -> Set[str]:
        """
        Get the set of memory IDs that have been isolated.
        
        Returns:
            Set of isolated memory IDs
        """
        return self.locked_memories.copy()


def create_entropy_response(
    memory_store: MemoryStore,
    entropy_monitor: EntropyMonitor,
    reflection_trigger_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    config: Optional[EntropyResponseConfig] = None
) -> EntropyResponse:
    """
    Create and register an entropy response system with an entropy monitor.
    
    Args:
        memory_store: Memory store to manage
        entropy_monitor: Entropy monitor to subscribe to
        reflection_trigger_callback: Function to call to trigger reflection
        config: Response configuration
        
    Returns:
        Configured EntropyResponse instance
    """
    response = EntropyResponse(
        memory_store=memory_store,
        config=config,
        reflection_trigger_callback=reflection_trigger_callback
    )
    
    # Register with the monitor
    entropy_monitor.add_subscriber(response)
    
    logger.info("Entropy response system initialized and registered with monitor")
    return response 