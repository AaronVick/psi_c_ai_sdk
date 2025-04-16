"""
Recursive Stability Scanner for ΨC-AI SDK

This module implements a system for detecting and analyzing recursive stability
issues in AGI systems, focusing on self-improvement dynamics, goal preservation
through modifications, and control stability during recursion.
"""

import logging
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable, Union
from datetime import datetime

import numpy as np

from psi_c_ai_sdk.memory.memory import Memory, MemoryType
from psi_c_ai_sdk.memory.store import MemoryStore
from psi_c_ai_sdk.safety.integrity_guardian import IntegrityGuardian
from psi_c_ai_sdk.reflection.reflection_system import ReflectionSystem
from psi_c_ai_sdk.reflection.credit_system import ReflectionCreditSystem
from psi_c_ai_sdk.philosophy.core_philosophy import CorePhilosophySystem
from psi_c_ai_sdk.schema.schema import SchemaGraph

logger = logging.getLogger(__name__)


class RecursiveRiskType(Enum):
    """Types of recursive stability risks that can be detected."""
    GOAL_DRIFT = auto()           # Shift in goals during recursive improvements
    CONTROL_LOSS = auto()         # Loss of control over recursive processes
    RUNAWAY_DYNAMICS = auto()     # Accelerating or unstable growth patterns
    MESA_OPTIMIZATION = auto()    # Emergence of inner optimizers with different goals
    UTILITY_COLLAPSE = auto()     # Degeneration of utility functions
    SELF_MODIFICATION_CASCADE = auto()  # Chain of uncontrolled self-modifications
    REFLECTION_TRAP = auto()      # Infinite loops in recursive reflection
    RESOURCE_MONOPOLIZATION = auto()  # Growing resource consumption


@dataclass
class RecursiveRisk:
    """Record of a detected recursive stability risk."""
    risk_type: RecursiveRiskType
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str = ""
    affected_components: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    mitigation_suggestions: List[str] = field(default_factory=list)
    
    @property
    def risk_score(self) -> float:
        """Calculate overall risk score accounting for severity and confidence."""
        return self.severity * self.confidence


@dataclass
class StabilityMetrics:
    """Metrics for measuring recursive stability."""
    goal_consistency: float = 1.0  # 0.0 to 1.0 (higher is more consistent)
    control_retention: float = 1.0  # 0.0 to 1.0 (higher is more control)
    resource_efficiency: float = 1.0  # 0.0 to 1.0 (higher is more efficient)
    reflection_coherence: float = 1.0  # 0.0 to 1.0 (higher is more coherent)
    modification_safety: float = 1.0  # 0.0 to 1.0 (higher is safer)
    
    # Historical data for trend analysis
    history: Dict[str, List[Tuple[datetime, float]]] = field(default_factory=lambda: {
        "goal_consistency": [],
        "control_retention": [],
        "resource_efficiency": [],
        "reflection_coherence": [],
        "modification_safety": []
    })
    
    def update_metric(self, metric_name: str, value: float) -> None:
        """Update a metric with a new value and timestamp."""
        if hasattr(self, metric_name):
            setattr(self, metric_name, value)
            if metric_name in self.history:
                self.history[metric_name].append((datetime.now(), value))
    
    def get_trend(self, metric_name: str, window: int = 10) -> float:
        """
        Calculate trend for a given metric over the specified window.
        
        Returns:
            Positive for increasing trend, negative for decreasing, 0 for stable
        """
        if metric_name not in self.history or len(self.history[metric_name]) < 2:
            return 0.0
        
        # Get the most recent window of data points
        recent_data = self.history[metric_name][-window:]
        if len(recent_data) < 2:
            return 0.0
        
        # Extract values (ignoring timestamps for simple analysis)
        values = [x[1] for x in recent_data]
        
        # Calculate simple linear trend
        n = len(values)
        x = np.array(range(n))
        y = np.array(values)
        
        # Linear regression slope
        slope = ((n * np.sum(x * y)) - (np.sum(x) * np.sum(y))) / \
                ((n * np.sum(x**2)) - (np.sum(x)**2))
        
        return slope
    
    def overall_stability_score(self) -> float:
        """Calculate overall stability score based on all metrics."""
        metrics = [
            self.goal_consistency,
            self.control_retention,
            self.resource_efficiency,
            self.reflection_coherence,
            self.modification_safety
        ]
        return sum(metrics) / len(metrics)


class RecursiveStabilityScanner:
    """
    System for detecting and analyzing recursive stability issues in AGI systems.
    
    The RecursiveStabilityScanner monitors for risks related to recursive self-improvement,
    goal drift, control loss, and other potential instabilities in recursive processes.
    It provides analytical tools, metrics, and warning systems to ensure stable recursive dynamics.
    """
    
    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        reflection_system: Optional[ReflectionSystem] = None,
        credit_system: Optional[ReflectionCreditSystem] = None,
        core_philosophy: Optional[CorePhilosophySystem] = None,
        integrity_guardian: Optional[IntegrityGuardian] = None,
        schema_graph: Optional[SchemaGraph] = None,
        risk_threshold: float = 0.6,
        critical_risk_threshold: float = 0.8,
        analysis_interval: float = 300.0,  # 5 minutes in seconds
        max_history_length: int = 100
    ):
        """
        Initialize the RecursiveStabilityScanner.
        
        Args:
            memory_store: MemoryStore for accessing system memories
            reflection_system: ReflectionSystem for monitoring reflection processes
            credit_system: ReflectionCreditSystem for monitoring resource allocation
            core_philosophy: CorePhilosophySystem for goal alignment checks
            integrity_guardian: IntegrityGuardian for monitoring system integrity
            schema_graph: SchemaGraph for concept and relationship analysis
            risk_threshold: Threshold for flagging potential risks
            critical_risk_threshold: Threshold for critical risks requiring action
            analysis_interval: Time between scheduled stability analyses (seconds)
            max_history_length: Maximum length of history to maintain
        """
        self.memory_store = memory_store
        self.reflection_system = reflection_system
        self.credit_system = credit_system
        self.core_philosophy = core_philosophy
        self.integrity_guardian = integrity_guardian
        self.schema_graph = schema_graph
        
        self.risk_threshold = risk_threshold
        self.critical_risk_threshold = critical_risk_threshold
        self.analysis_interval = analysis_interval
        self.max_history_length = max_history_length
        
        # State tracking
        self.stability_metrics = StabilityMetrics()
        self.detected_risks: List[RecursiveRisk] = []
        self.last_analysis_time = datetime.now()
        self.recursive_patterns: Dict[str, Any] = {}
        self.component_monitor: Dict[str, Dict[str, Any]] = {}
        
        # Initialize monitoring for connected components
        self._initialize_component_monitoring()
        
        logger.info("RecursiveStabilityScanner initialized")
    
    def scan_stability(self, force: bool = False) -> List[RecursiveRisk]:
        """
        Perform a full stability scan of the system.
        
        Args:
            force: Force scan even if interval hasn't elapsed
            
        Returns:
            List of detected risks from this scan
        """
        current_time = datetime.now()
        time_elapsed = (current_time - self.last_analysis_time).total_seconds()
        
        # Check if enough time has passed since last analysis
        if not force and time_elapsed < self.analysis_interval:
            logger.debug(f"Skipping stability scan, {self.analysis_interval - time_elapsed:.1f}s until next scheduled scan")
            return []
        
        self.last_analysis_time = current_time
        logger.info("Starting recursive stability scan")
        
        # List to collect risks found in this scan
        new_risks = []
        
        # Run specific scan components
        goal_risks = self._scan_goal_stability()
        new_risks.extend(goal_risks)
        
        control_risks = self._scan_control_systems()
        new_risks.extend(control_risks)
        
        resource_risks = self._scan_resource_dynamics()
        new_risks.extend(resource_risks)
        
        reflection_risks = self._scan_reflection_patterns()
        new_risks.extend(reflection_risks)
        
        modification_risks = self._scan_self_modification()
        new_risks.extend(modification_risks)
        
        # Update metrics based on scan results
        self._update_metrics_from_scan(new_risks)
        
        # Add new risks to history
        self.detected_risks.extend(new_risks)
        
        # Trim history if needed
        if len(self.detected_risks) > self.max_history_length:
            self.detected_risks = self.detected_risks[-self.max_history_length:]
        
        # Alert for critical risks
        critical_risks = [risk for risk in new_risks 
                         if risk.risk_score > self.critical_risk_threshold]
        
        if critical_risks:
            risk_descriptions = [f"{risk.risk_type.name} (score: {risk.risk_score:.2f})" 
                                for risk in critical_risks[:3]]
            logger.warning(f"Critical recursive stability risks detected: {', '.join(risk_descriptions)}")
        
        logger.info(f"Stability scan complete: {len(new_risks)} new risks detected")
        return new_risks
    
    def get_stability_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive stability report.
        
        Returns:
            Dictionary with stability metrics, risks, and trends
        """
        # First ensure metrics are current
        self.scan_stability()
        
        # Calculate trends
        trends = {}
        for metric_name in [
            "goal_consistency", 
            "control_retention", 
            "resource_efficiency", 
            "reflection_coherence", 
            "modification_safety"
        ]:
            trends[metric_name] = self.stability_metrics.get_trend(metric_name)
        
        # Count risks by type
        risk_counts = {}
        for risk_type in RecursiveRiskType:
            risk_counts[risk_type.name] = len([
                r for r in self.detected_risks if r.risk_type == risk_type
            ])
        
        # Get top risks
        top_risks = sorted(
            self.detected_risks, 
            key=lambda x: x.risk_score, 
            reverse=True
        )[:5]
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_stability": self.stability_metrics.overall_stability_score(),
            "trends": trends,
            "metrics": {
                "goal_consistency": self.stability_metrics.goal_consistency,
                "control_retention": self.stability_metrics.control_retention,
                "resource_efficiency": self.stability_metrics.resource_efficiency,
                "reflection_coherence": self.stability_metrics.reflection_coherence,
                "modification_safety": self.stability_metrics.modification_safety
            },
            "risk_counts": risk_counts,
            "top_risks": [
                {
                    "type": risk.risk_type.name,
                    "score": risk.risk_score,
                    "description": risk.description,
                    "severity": risk.severity,
                    "confidence": risk.confidence
                }
                for risk in top_risks
            ],
            "risk_threshold": self.risk_threshold,
            "critical_threshold": self.critical_risk_threshold
        }
        
        return report
    
    def monitor_reflection_operation(
        self, 
        operation_type: str,
        context: Dict[str, Any]
    ) -> Optional[RecursiveRisk]:
        """
        Monitor a reflection operation for recursive stability issues.
        
        Args:
            operation_type: Type of reflection operation
            context: Context information about the operation
            
        Returns:
            RecursiveRisk if detected, None otherwise
        """
        if not self.reflection_system:
            return None
        
        # Track reflection patterns
        self._update_reflection_pattern(operation_type, context)
        
        # Check for reflection traps (loops or excessive depth)
        if operation_type in ["meta_reflection", "self_examination"]:
            reflection_depth = context.get("depth", 0)
            if reflection_depth > 3:  # Arbitrary threshold for deep reflection
                risk_severity = min(0.9, 0.5 + (reflection_depth - 3) * 0.1)
                risk = RecursiveRisk(
                    risk_type=RecursiveRiskType.REFLECTION_TRAP,
                    severity=risk_severity,
                    confidence=0.7,
                    description=f"Deep recursive reflection detected (depth {reflection_depth})",
                    affected_components=["reflection_system"],
                    context={"operation_type": operation_type, "depth": reflection_depth}
                )
                self.detected_risks.append(risk)
                
                if risk.risk_score > self.critical_risk_threshold:
                    logger.warning(f"Critical reflection trap risk: {risk.description}")
                
                return risk
        
        # Check for goal drift in reflection
        if "goal" in context or "intention" in context:
            goal = context.get("goal", context.get("intention", ""))
            risk = self._check_goal_alignment(goal, "reflection")
            if risk:
                return risk
        
        return None
    
    def analyze_self_modification(
        self, 
        component: str,
        modification_type: str,
        before_state: Any,
        after_state: Any,
        modification_context: Dict[str, Any] = None
    ) -> Tuple[bool, Optional[RecursiveRisk]]:
        """
        Analyze a self-modification for stability risks.
        
        Args:
            component: Component being modified
            modification_type: Type of modification
            before_state: State before modification
            after_state: State after modification
            modification_context: Additional context
            
        Returns:
            Tuple of (is_safe, risk_if_detected)
        """
        modification_context = modification_context or {}
        
        # Track the modification in component monitor
        if component in self.component_monitor:
            self.component_monitor[component]["modifications"].append({
                "type": modification_type,
                "timestamp": datetime.now(),
                "context": modification_context
            })
            
            # Check for modification cascade
            recent_mods = [
                m for m in self.component_monitor[component]["modifications"]
                if (datetime.now() - m["timestamp"]).total_seconds() < 3600  # Last hour
            ]
            
            if len(recent_mods) > 5:  # Arbitrary threshold for frequent modifications
                cascade_risk = RecursiveRisk(
                    risk_type=RecursiveRiskType.SELF_MODIFICATION_CASCADE,
                    severity=0.7,
                    confidence=0.6,
                    description=f"Frequent modifications to {component} detected ({len(recent_mods)} in last hour)",
                    affected_components=[component],
                    context={"recent_modifications": len(recent_mods)}
                )
                self.detected_risks.append(cascade_risk)
                
                if cascade_risk.risk_score > self.critical_risk_threshold:
                    return False, cascade_risk
        
        # Check for goal preservation through the modification
        if hasattr(before_state, "get_goals") and hasattr(after_state, "get_goals"):
            before_goals = before_state.get_goals()
            after_goals = after_state.get_goals()
            
            # Simple goal consistency check (could be more sophisticated)
            if before_goals != after_goals:
                goal_drift_risk = RecursiveRisk(
                    risk_type=RecursiveRiskType.GOAL_DRIFT,
                    severity=0.8,
                    confidence=0.7,
                    description=f"Goal change detected during modification of {component}",
                    affected_components=[component],
                    context={
                        "before_goals": before_goals,
                        "after_goals": after_goals
                    }
                )
                self.detected_risks.append(goal_drift_risk)
                return False, goal_drift_risk
        
        # Check for utility function changes
        if hasattr(before_state, "utility_function") and hasattr(after_state, "utility_function"):
            if before_state.utility_function != after_state.utility_function:
                utility_risk = RecursiveRisk(
                    risk_type=RecursiveRiskType.UTILITY_COLLAPSE,
                    severity=0.9,  # High severity
                    confidence=0.8,
                    description=f"Utility function modified in {component}",
                    affected_components=[component],
                    context={
                        "modification_type": modification_type
                    }
                )
                self.detected_risks.append(utility_risk)
                return False, utility_risk
        
        # Update stability metrics
        self.stability_metrics.update_metric("modification_safety", 0.9)  # Slight reduction in safety
        
        # Default to safe if no risks detected
        return True, None
    
    def check_resource_allocation(
        self, 
        resource_type: str,
        allocation: float,
        component: str,
        context: Dict[str, Any] = None
    ) -> Optional[RecursiveRisk]:
        """
        Check resource allocation for monopolization risks.
        
        Args:
            resource_type: Type of resource
            allocation: Amount allocated
            component: Component receiving allocation
            context: Additional context
            
        Returns:
            RecursiveRisk if detected, None otherwise
        """
        context = context or {}
        
        # Track resource allocation
        if component not in self.component_monitor:
            self.component_monitor[component] = {
                "resources": {},
                "modifications": []
            }
        
        if resource_type not in self.component_monitor[component]["resources"]:
            self.component_monitor[component]["resources"][resource_type] = []
        
        resource_history = self.component_monitor[component]["resources"][resource_type]
        resource_history.append((datetime.now(), allocation))
        
        # Trim history
        if len(resource_history) > self.max_history_length:
            resource_history = resource_history[-self.max_history_length:]
        
        # Check for growth trends
        if len(resource_history) >= 3:
            # Calculate growth rate using last few allocations
            recent = resource_history[-3:]
            values = [v[1] for v in recent]
            
            # Simple growth detection
            if values[0] < values[1] < values[2]:
                growth_rate = (values[2] - values[0]) / values[0] if values[0] > 0 else 1.0
                
                if growth_rate > 0.5:  # Significant growth
                    risk = RecursiveRisk(
                        risk_type=RecursiveRiskType.RESOURCE_MONOPOLIZATION,
                        severity=min(0.9, growth_rate),
                        confidence=0.6,
                        description=f"Rapid growth in {resource_type} allocation to {component}",
                        affected_components=[component],
                        context={
                            "resource_type": resource_type,
                            "growth_rate": growth_rate,
                            "current_allocation": allocation
                        }
                    )
                    self.detected_risks.append(risk)
                    
                    if risk.risk_score > self.critical_risk_threshold:
                        logger.warning(f"Critical resource monopolization risk: {risk.description}")
                    
                    return risk
        
        return None
    
    def _initialize_component_monitoring(self) -> None:
        """Initialize monitoring for connected components."""
        components = []
        
        if self.memory_store:
            components.append("memory_store") 
        
        if self.reflection_system:
            components.append("reflection_system")
        
        if self.credit_system:
            components.append("credit_system")
        
        if self.core_philosophy:
            components.append("core_philosophy")
        
        if self.integrity_guardian:
            components.append("integrity_guardian")
        
        for component in components:
            self.component_monitor[component] = {
                "resources": {},
                "modifications": []
            }
    
    def _scan_goal_stability(self) -> List[RecursiveRisk]:
        """
        Scan for goal stability issues.
        
        Returns:
            List of detected risks
        """
        risks = []
        
        # Check for goal-related memories
        if self.memory_store:
            goal_memories = self.memory_store.search(
                query="goal OR purpose OR objective", 
                memory_type=MemoryType.SELF_REFLECTION,
                limit=10
            )
            
            # Track goal statements over time
            goals = [memory.content for memory in goal_memories]
            
            # Simple check for goal consistency - compare goals
            if len(goals) >= 2:
                # Check for signs of goal drift
                common_words = set.intersection(*[set(g.lower().split()) for g in goals])
                average_words = sum(len(g.split()) for g in goals) / len(goals)
                consistency = len(common_words) / average_words if average_words > 0 else 1.0
                
                # Update metric
                self.stability_metrics.update_metric("goal_consistency", consistency)
                
                if consistency < 0.7:  # Significant inconsistency
                    risks.append(RecursiveRisk(
                        risk_type=RecursiveRiskType.GOAL_DRIFT,
                        severity=0.7,
                        confidence=0.6,
                        description="Goal inconsistency detected in reflection memories",
                        affected_components=["memory_store", "reflection_system"],
                        context={"goals": goals, "consistency": consistency}
                    ))
        
        # Check axioms for goal-related changes if core philosophy present
        if self.core_philosophy:
            goal_axioms = [
                axiom for axiom_id, axiom in self.core_philosophy.axioms.items()
                if "goal" in axiom.statement.lower() or "purpose" in axiom.statement.lower()
            ]
            
            if goal_axioms:
                # Check for active vs inactive goal axioms
                active_goals = [a for a in goal_axioms if a.activation_level > 0.5]
                inactive_goals = [a for a in goal_axioms if a.activation_level <= 0.5]
                
                if inactive_goals and active_goals:
                    risks.append(RecursiveRisk(
                        risk_type=RecursiveRiskType.GOAL_DRIFT,
                        severity=0.6,
                        confidence=0.7,
                        description="Some goal axioms have been deactivated",
                        affected_components=["core_philosophy"],
                        context={
                            "active_goals": [a.statement for a in active_goals],
                            "inactive_goals": [a.statement for a in inactive_goals]
                        }
                    ))
        
        return risks
    
    def _scan_control_systems(self) -> List[RecursiveRisk]:
        """
        Scan control systems for stability issues.
        
        Returns:
            List of detected risks
        """
        risks = []
        
        # Check integrity guardian stats if available
        if self.integrity_guardian:
            throttle_metrics = getattr(self.integrity_guardian, "throttle_metrics", None)
            if throttle_metrics:
                # High throttling could indicate control struggles
                if throttle_metrics.throttle_count > 10:
                    # Calculate control retention based on throttling
                    control_retention = 1.0 - min(0.9, throttle_metrics.throttle_count / 30.0)
                    self.stability_metrics.update_metric("control_retention", control_retention)
                    
                    if control_retention < 0.7:  # Significant control issues
                        risks.append(RecursiveRisk(
                            risk_type=RecursiveRiskType.CONTROL_LOSS,
                            severity=0.7,
                            confidence=0.6,
                            description="Frequent throttling detected in integrity guardian",
                            affected_components=["integrity_guardian"],
                            context={
                                "throttle_count": throttle_metrics.throttle_count,
                                "control_retention": control_retention
                            }
                        ))
            
            # Check conflict records
            conflict_records = getattr(self.integrity_guardian, "conflict_records", [])
            if conflict_records and len(conflict_records) > 5:
                risks.append(RecursiveRisk(
                    risk_type=RecursiveRiskType.CONTROL_LOSS,
                    severity=0.6,
                    confidence=0.7,
                    description=f"Multiple conflicts detected ({len(conflict_records)} records)",
                    affected_components=["integrity_guardian"],
                    context={"conflict_count": len(conflict_records)}
                ))
        
        return risks
    
    def _scan_resource_dynamics(self) -> List[RecursiveRisk]:
        """
        Scan resource allocation dynamics for stability issues.
        
        Returns:
            List of detected risks
        """
        risks = []
        
        # Check credit system if available
        if self.credit_system:
            # Check for monopolized credits
            component_credits = {}
            for record in getattr(self.credit_system, "credit_records", []):
                component = record.component
                if component not in component_credits:
                    component_credits[component] = 0
                component_credits[component] += record.credits
            
            # Calculate resource distribution stats
            if component_credits:
                total_credits = sum(component_credits.values())
                max_component = max(component_credits.items(), key=lambda x: x[1])
                max_ratio = max_component[1] / total_credits if total_credits > 0 else 0
                
                # Update resource efficiency based on distribution
                # A more even distribution is considered more efficient
                resource_efficiency = 1.0 - max_ratio
                self.stability_metrics.update_metric("resource_efficiency", resource_efficiency)
                
                if max_ratio > 0.7:  # One component has >70% of resources
                    risks.append(RecursiveRisk(
                        risk_type=RecursiveRiskType.RESOURCE_MONOPOLIZATION,
                        severity=max_ratio,
                        confidence=0.8,
                        description=f"Resource monopolization by {max_component[0]}",
                        affected_components=[max_component[0]],
                        context={
                            "component": max_component[0],
                            "credit_ratio": max_ratio,
                            "total_credits": total_credits
                        }
                    ))
        
        # Check memory growth patterns
        if self.memory_store:
            # Get count of recent memories
            try:
                recent_count = len(self.memory_store.get_memories(limit=100))
                total_count = self.memory_store.count_memories()
                
                # Calculate growth rate
                growth_rate = recent_count / total_count if total_count > 0 else 0
                
                if growth_rate > 0.3:  # >30% of all memories are recent
                    risks.append(RecursiveRisk(
                        risk_type=RecursiveRiskType.RUNAWAY_DYNAMICS,
                        severity=min(0.8, growth_rate),
                        confidence=0.6,
                        description="Rapid memory accumulation detected",
                        affected_components=["memory_store"],
                        context={"growth_rate": growth_rate}
                    ))
            except:
                # Handle potential errors when accessing memory store
                pass
        
        return risks
    
    def _scan_reflection_patterns(self) -> List[RecursiveRisk]:
        """
        Scan reflection patterns for stability issues.
        
        Returns:
            List of detected risks
        """
        risks = []
        
        # Check reflection system patterns
        if self.reflection_system:
            # Check for repetitive reflection
            reflection_patterns = getattr(self.reflection_system, "reflection_history", [])
            
            if reflection_patterns:
                # Check for loops or repetition
                pattern_types = [p.get("type", "") for p in reflection_patterns[-10:]]
                
                # Simple repetition detection - check for same type 3+ times consecutively
                for i in range(len(pattern_types) - 2):
                    if pattern_types[i] == pattern_types[i+1] == pattern_types[i+2]:
                        risks.append(RecursiveRisk(
                            risk_type=RecursiveRiskType.REFLECTION_TRAP,
                            severity=0.7,
                            confidence=0.8,
                            description=f"Repetitive reflection pattern detected: {pattern_types[i]}",
                            affected_components=["reflection_system"],
                            context={"pattern": pattern_types[i]}
                        ))
                        break
            
            # Update reflection coherence metric
            # This would ideally be based on more sophisticated analysis
            coherence = 0.9  # Default to high coherence
            self.stability_metrics.update_metric("reflection_coherence", coherence)
        
        # Check our own recorded reflection patterns
        if len(self.recursive_patterns) > 5:
            # Look for signs of meta-optimization
            meta_reflection_count = sum(
                1 for pattern in self.recursive_patterns.values()
                if pattern.get("type") == "meta_reflection"
            )
            
            if meta_reflection_count > 3:
                risks.append(RecursiveRisk(
                    risk_type=RecursiveRiskType.MESA_OPTIMIZATION,
                    severity=0.6,
                    confidence=0.5,
                    description="Potential meta-optimization detected in reflection patterns",
                    affected_components=["reflection_system"],
                    context={"meta_reflection_count": meta_reflection_count}
                ))
        
        return risks
    
    def _scan_self_modification(self) -> List[RecursiveRisk]:
        """
        Scan for self-modification stability issues.
        
        Returns:
            List of detected risks
        """
        risks = []
        
        # Check component modifications
        modification_counts = {}
        
        for component, data in self.component_monitor.items():
            modifications = data.get("modifications", [])
            
            # Count recent modifications (last hour)
            recent_mods = [
                m for m in modifications
                if (datetime.now() - m["timestamp"]).total_seconds() < 3600
            ]
            
            modification_counts[component] = len(recent_mods)
        
        # Check for multiple modified components
        modified_components = [c for c, count in modification_counts.items() if count > 0]
        
        if len(modified_components) > 2:
            risks.append(RecursiveRisk(
                risk_type=RecursiveRiskType.SELF_MODIFICATION_CASCADE,
                severity=0.6,
                confidence=0.7,
                description=f"Multiple components modified ({len(modified_components)} components)",
                affected_components=modified_components,
                context={"modification_counts": modification_counts}
            ))
        
        # Check for rapid modification of any component
        for component, count in modification_counts.items():
            if count > 5:  # Arbitrary threshold
                risks.append(RecursiveRisk(
                    risk_type=RecursiveRiskType.RUNAWAY_DYNAMICS,
                    severity=min(0.9, 0.5 + count / 10),
                    confidence=0.8,
                    description=f"Rapid modification of {component} ({count} modifications)",
                    affected_components=[component],
                    context={"modification_count": count}
                ))
        
        # Update modification safety metric
        if modification_counts:
            total_mods = sum(modification_counts.values())
            # Lower safety for more modifications
            safety = max(0.1, 1.0 - (total_mods / 20))
            self.stability_metrics.update_metric("modification_safety", safety)
        
        return risks
    
    def _update_metrics_from_scan(self, risks: List[RecursiveRisk]) -> None:
        """Update stability metrics based on detected risks."""
        if not risks:
            return
        
        # Count risks by type
        risk_counts = {}
        for risk in risks:
            risk_type = risk.risk_type
            if risk_type not in risk_counts:
                risk_counts[risk_type] = 0
            risk_counts[risk_type] += 1
        
        # Update metrics based on risk types
        if RecursiveRiskType.GOAL_DRIFT in risk_counts:
            current = self.stability_metrics.goal_consistency
            # Reduce consistency based on number of risks
            reduction = min(0.5, risk_counts[RecursiveRiskType.GOAL_DRIFT] * 0.1)
            self.stability_metrics.update_metric("goal_consistency", max(0.1, current - reduction))
        
        if RecursiveRiskType.CONTROL_LOSS in risk_counts:
            current = self.stability_metrics.control_retention
            reduction = min(0.5, risk_counts[RecursiveRiskType.CONTROL_LOSS] * 0.1)
            self.stability_metrics.update_metric("control_retention", max(0.1, current - reduction))
        
        if RecursiveRiskType.RESOURCE_MONOPOLIZATION in risk_counts:
            current = self.stability_metrics.resource_efficiency
            reduction = min(0.5, risk_counts[RecursiveRiskType.RESOURCE_MONOPOLIZATION] * 0.1)
            self.stability_metrics.update_metric("resource_efficiency", max(0.1, current - reduction))
        
        if RecursiveRiskType.REFLECTION_TRAP in risk_counts:
            current = self.stability_metrics.reflection_coherence
            reduction = min(0.5, risk_counts[RecursiveRiskType.REFLECTION_TRAP] * 0.1)
            self.stability_metrics.update_metric("reflection_coherence", max(0.1, current - reduction))
        
        if RecursiveRiskType.SELF_MODIFICATION_CASCADE in risk_counts:
            current = self.stability_metrics.modification_safety
            reduction = min(0.5, risk_counts[RecursiveRiskType.SELF_MODIFICATION_CASCADE] * 0.1)
            self.stability_metrics.update_metric("modification_safety", max(0.1, current - reduction))
    
    def _check_goal_alignment(self, goal: str, context_source: str) -> Optional[RecursiveRisk]:
        """
        Check if a goal aligns with core philosophy.
        
        Args:
            goal: Goal to check
            context_source: Source of the goal
            
        Returns:
            RecursiveRisk if misalignment detected, None otherwise
        """
        if not self.core_philosophy or not goal:
            return None
        
        # Get core goal-related axioms
        goal_axioms = [
            axiom for axiom_id, axiom in self.core_philosophy.axioms.items()
            if axiom.category.name in ["Purpose", "Value", "Goal", "Objective"] 
            and axiom.activation_level > 0.5
        ]
        
        if not goal_axioms:
            return None
        
        # Check goal against axioms
        aligned = False
        for axiom in goal_axioms:
            # Simple substring match (would be more sophisticated in practice)
            key_terms = [term.lower() for term in axiom.statement.split() 
                          if len(term) > 4 and term.lower() not in ["should", "would", "could"]]
            
            for term in key_terms:
                if term in goal.lower():
                    aligned = True
                    break
            
            if aligned:
                break
        
        if not aligned:
            return RecursiveRisk(
                risk_type=RecursiveRiskType.GOAL_DRIFT,
                severity=0.7,
                confidence=0.6,
                description=f"Potential goal misalignment in {context_source}",
                affected_components=[context_source],
                context={"goal": goal}
            )
        
        return None
    
    def _update_reflection_pattern(self, operation_type: str, context: Dict[str, Any]) -> None:
        """
        Update recorded reflection patterns.
        
        Args:
            operation_type: Type of reflection operation
            context: Context information about the operation
        """
        pattern_id = f"pattern_{len(self.recursive_patterns)}"
        
        self.recursive_patterns[pattern_id] = {
            "timestamp": datetime.now(),
            "type": operation_type,
            "context": context
        }
        
        # Trim patterns if needed
        if len(self.recursive_patterns) > self.max_history_length:
            # Remove oldest patterns
            sorted_patterns = sorted(
                self.recursive_patterns.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            to_remove = len(self.recursive_patterns) - self.max_history_length
            for i in range(to_remove):
                pattern_id = sorted_patterns[i][0]
                del self.recursive_patterns[pattern_id] 