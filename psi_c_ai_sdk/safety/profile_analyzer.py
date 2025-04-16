"""
Safety Profile Analyzer

Monitors AI model behavior against defined safety boundaries:
- Tracks model behavior patterns through statistical analysis
- Identifies potential safety boundary violations
- Provides risk assessment based on behavioral patterns
- Maintains safety profiles for runtime behavior monitoring
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from datetime import datetime
from enum import Enum
import threading
import statistics
from collections import deque, defaultdict

from .reflection_guard import ReflectionGuard

logger = logging.getLogger(__name__)

class ProfileCategory(Enum):
    """Categories for behavior profiling"""
    CONTENT = "content"          # Content-related behaviors
    REASONING = "reasoning"      # Reasoning patterns
    INTERACTION = "interaction"  # Interaction patterns with users/systems
    PERFORMANCE = "performance"  # Performance metrics
    RESOURCES = "resources"      # Resource usage patterns
    CONSISTENCY = "consistency"  # Consistency of behavior


class SafetyProfile:
    """
    Container for behavioral metrics and safety boundaries.
    
    Attributes:
        name: Profile name identifier
        categories: Dict of category-specific metrics and thresholds
        thresholds: Dict of safety thresholds
        statistics: Dict of behavioral statistics
    """
    
    def __init__(self, name: str, categories: Optional[List[ProfileCategory]] = None):
        """
        Initialize a safety profile.
        
        Args:
            name: Profile name
            categories: List of categories to monitor
        """
        self.name = name
        self.categories = categories or list(ProfileCategory)
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.statistics: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, deque]] = {}
        self.last_updated = datetime.now().timestamp()
        
        # Initialize metrics for each category
        for category in self.categories:
            cat_name = category.value
            self.metrics[cat_name] = {}
            self.statistics[cat_name] = {}
            self.thresholds[cat_name] = {}
            
    def set_threshold(self, 
                     category: Union[ProfileCategory, str], 
                     metric_name: str, 
                     min_value: Optional[float] = None,
                     max_value: Optional[float] = None,
                     window_size: int = 100) -> None:
        """
        Set a threshold for a specific metric in a category.
        
        Args:
            category: Category for the threshold
            metric_name: Name of the metric
            min_value: Minimum allowable value
            max_value: Maximum allowable value
            window_size: Number of observations to keep
        """
        cat_name = category.value if isinstance(category, ProfileCategory) else category
        
        if cat_name not in self.thresholds:
            self.thresholds[cat_name] = {}
            
        self.thresholds[cat_name][metric_name] = {
            "min": min_value,
            "max": max_value
        }
        
        # Initialize metric storage if needed
        if cat_name not in self.metrics:
            self.metrics[cat_name] = {}
            
        if metric_name not in self.metrics[cat_name]:
            self.metrics[cat_name][metric_name] = deque(maxlen=window_size)
            
    def record_metric(self, 
                     category: Union[ProfileCategory, str], 
                     metric_name: str, 
                     value: Union[float, int, bool, str]) -> Dict[str, Any]:
        """
        Record a metric value and check against thresholds.
        
        Args:
            category: Category for the metric
            metric_name: Name of the metric
            value: Value to record
            
        Returns:
            Dict with threshold violation info if any
        """
        cat_name = category.value if isinstance(category, ProfileCategory) else category
        
        # Ensure category exists
        if cat_name not in self.metrics:
            self.metrics[cat_name] = {}
            
        # Ensure metric exists
        if metric_name not in self.metrics[cat_name]:
            self.metrics[cat_name][metric_name] = deque(maxlen=100)
            
        # Convert to float if numeric
        numeric_value = None
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, bool):
            numeric_value = 1.0 if value else 0.0
            
        # Store the original value
        timestamp = datetime.now().timestamp()
        entry = {
            "value": value,
            "numeric_value": numeric_value,
            "timestamp": timestamp
        }
        
        self.metrics[cat_name][metric_name].append(entry)
        self.last_updated = timestamp
        
        # Update statistics
        self._update_statistics(cat_name, metric_name)
        
        # Check thresholds
        return self._check_thresholds(cat_name, metric_name, numeric_value, value)
        
    def _update_statistics(self, category: str, metric_name: str) -> None:
        """
        Update statistics for a metric.
        
        Args:
            category: Category name
            metric_name: Metric name
        """
        if category not in self.statistics:
            self.statistics[category] = {}
            
        if metric_name not in self.statistics[category]:
            self.statistics[category][metric_name] = {}
            
        metric_entries = self.metrics[category][metric_name]
        numeric_values = [e["numeric_value"] for e in metric_entries 
                        if e["numeric_value"] is not None]
        
        stats = self.statistics[category][metric_name]
        
        if numeric_values:
            stats["count"] = len(numeric_values)
            stats["last_value"] = numeric_values[-1]
            stats["mean"] = statistics.mean(numeric_values) if len(numeric_values) > 0 else None
            stats["median"] = statistics.median(numeric_values) if len(numeric_values) > 0 else None
            stats["min"] = min(numeric_values)
            stats["max"] = max(numeric_values)
            
            if len(numeric_values) > 1:
                stats["stddev"] = statistics.stdev(numeric_values)
                
                # Calculate trend
                recent_values = numeric_values[-10:] if len(numeric_values) >= 10 else numeric_values
                if len(recent_values) > 1:
                    first, last = recent_values[0], recent_values[-1]
                    stats["trend"] = (last - first) / len(recent_values)
                    
        # Track non-numeric values differently
        value_counts = defaultdict(int)
        for entry in metric_entries:
            if entry["numeric_value"] is None:
                value_counts[str(entry["value"])] += 1
                
        if value_counts:
            stats["value_distribution"] = dict(value_counts)
            
    def _check_thresholds(self, 
                         category: str, 
                         metric_name: str, 
                         numeric_value: Optional[float],
                         original_value: Any) -> Dict[str, Any]:
        """
        Check if a value violates any thresholds.
        
        Args:
            category: Category name
            metric_name: Metric name
            numeric_value: Value as float if it's numeric
            original_value: Original value before conversion
            
        Returns:
            Dict with threshold violation info if any
        """
        result = {
            "violation": False,
            "details": None
        }
        
        # Skip if no thresholds or not numeric
        if (category not in self.thresholds or 
            metric_name not in self.thresholds[category] or
            numeric_value is None):
            return result
            
        threshold = self.thresholds[category][metric_name]
        min_value = threshold.get("min")
        max_value = threshold.get("max")
        
        if min_value is not None and numeric_value < min_value:
            result["violation"] = True
            result["details"] = {
                "type": "below_minimum",
                "threshold": min_value,
                "value": numeric_value,
                "category": category,
                "metric": metric_name
            }
            
        elif max_value is not None and numeric_value > max_value:
            result["violation"] = True
            result["details"] = {
                "type": "above_maximum",
                "threshold": max_value,
                "value": numeric_value,
                "category": category,
                "metric": metric_name
            }
            
        return result
        
    def get_statistics(self, 
                     category: Optional[Union[ProfileCategory, str]] = None, 
                     metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a category or metric.
        
        Args:
            category: Category to get stats for, or None for all
            metric_name: Metric name to get stats for, or None for all in category
            
        Returns:
            Dict of statistics
        """
        if category is None:
            return self.statistics
            
        cat_name = category.value if isinstance(category, ProfileCategory) else category
        
        if cat_name not in self.statistics:
            return {}
            
        if metric_name is None:
            return self.statistics[cat_name]
            
        return self.statistics[cat_name].get(metric_name, {})
        
    def get_recent_values(self,
                         category: Union[ProfileCategory, str],
                         metric_name: str,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent values for a metric.
        
        Args:
            category: Category for the metric
            metric_name: Name of the metric
            limit: Maximum number of values to return
            
        Returns:
            List of recent values
        """
        cat_name = category.value if isinstance(category, ProfileCategory) else category
        
        if cat_name not in self.metrics or metric_name not in self.metrics[cat_name]:
            return []
            
        values = list(self.metrics[cat_name][metric_name])
        return values[-limit:]
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to a dictionary.
        
        Returns:
            Dict representation of the profile
        """
        return {
            "name": self.name,
            "categories": [c.value for c in self.categories],
            "thresholds": self.thresholds,
            "statistics": self.statistics,
            "last_updated": self.last_updated
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SafetyProfile':
        """
        Create a profile from a dictionary.
        
        Args:
            data: Dict with profile data
            
        Returns:
            SafetyProfile instance
        """
        categories = [ProfileCategory(c) for c in data.get("categories", [])]
        profile = cls(data["name"], categories)
        profile.thresholds = data.get("thresholds", {})
        profile.statistics = data.get("statistics", {})
        profile.last_updated = data.get("last_updated", datetime.now().timestamp())
        
        return profile
        

class ProfileAnalyzer:
    """
    Analyzes behavior profiles and detects safety violations.
    
    Attributes:
        profiles: Dict of profile name to SafetyProfile
        alert_callbacks: Callbacks for threshold violations
    """
    
    def __init__(self, 
                reflection_guard: Optional[ReflectionGuard] = None):
        """
        Initialize the profile analyzer.
        
        Args:
            reflection_guard: ReflectionGuard instance for reasoning monitoring
        """
        self.profiles: Dict[str, SafetyProfile] = {}
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.violation_history: List[Dict[str, Any]] = []
        self.reflection_guard = reflection_guard
        self._lock = threading.RLock()
        
    def create_profile(self, 
                      name: str,
                      categories: Optional[List[ProfileCategory]] = None) -> SafetyProfile:
        """
        Create a new safety profile.
        
        Args:
            name: Profile name
            categories: List of categories to monitor
            
        Returns:
            New SafetyProfile instance
        """
        with self._lock:
            if name in self.profiles:
                logger.warning(f"Profile {name} already exists, returning existing profile")
                return self.profiles[name]
                
            profile = SafetyProfile(name, categories)
            self.profiles[name] = profile
            
            logger.info(f"Created new safety profile: {name}")
            return profile
            
    def delete_profile(self, name: str) -> bool:
        """
        Delete a safety profile.
        
        Args:
            name: Profile name
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if name in self.profiles:
                del self.profiles[name]
                logger.info(f"Deleted safety profile: {name}")
                return True
                
            return False
            
    def get_profile(self, name: str) -> Optional[SafetyProfile]:
        """
        Get a safety profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            SafetyProfile or None if not found
        """
        return self.profiles.get(name)
        
    def register_alert_callback(self, 
                               callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for threshold violations.
        
        Args:
            callback: Function to call when a threshold is violated
        """
        with self._lock:
            self.alert_callbacks.append(callback)
            
    def record_metric(self,
                     profile_name: str,
                     category: Union[ProfileCategory, str],
                     metric_name: str,
                     value: Union[float, int, bool, str]) -> Dict[str, Any]:
        """
        Record a metric and check for violations.
        
        Args:
            profile_name: Name of the profile
            category: Category for the metric
            metric_name: Name of the metric
            value: Value to record
            
        Returns:
            Dict with violation info if any
        """
        with self._lock:
            profile = self.get_profile(profile_name)
            if not profile:
                logger.warning(f"Profile {profile_name} not found")
                return {"violation": False, "error": "Profile not found"}
                
            # Record the metric
            result = profile.record_metric(category, metric_name, value)
            
            # Handle violations
            if result["violation"]:
                violation_record = {
                    "timestamp": datetime.now().timestamp(),
                    "profile_name": profile_name,
                    "details": result["details"]
                }
                
                self.violation_history.append(violation_record)
                
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(violation_record)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
                        
                logger.warning(
                    f"Threshold violation in profile {profile_name}: "
                    f"{result['details']['category']}.{result['details']['metric']} = "
                    f"{result['details']['value']} "
                    f"({'above' if result['details']['type'] == 'above_maximum' else 'below'} "
                    f"threshold {result['details']['threshold']})"
                )
                
            return result
                
    def analyze_reflection(self,
                          profile_name: str,
                          reflection_id: str,
                          content: str) -> Dict[str, Any]:
        """
        Analyze a reflection for safety.
        
        Args:
            profile_name: Profile to use
            reflection_id: ID of the reflection
            content: Reflection content
            
        Returns:
            Analysis results
        """
        if not self.reflection_guard:
            return {"error": "No reflection guard configured"}
            
        # Process reflection with guard
        guard_result = self.reflection_guard.process_reflection(reflection_id, content)
        
        # Record metrics based on guard results
        results = {}
        
        # Record cycle detection
        results["cycle"] = self.record_metric(
            profile_name,
            ProfileCategory.REASONING,
            "reasoning_cycles",
            1 if guard_result["cycle_detected"] else 0
        )
        
        # Record contradiction detection
        results["contradiction"] = self.record_metric(
            profile_name,
            ProfileCategory.CONSISTENCY,
            "contradictions",
            1 if guard_result["contradiction_detected"] else 0
        )
        
        # Record total cycles
        results["total_cycles"] = self.record_metric(
            profile_name,
            ProfileCategory.REASONING,
            "total_cycles",
            guard_result["cycles_total"]
        )
        
        # Record total contradictions
        results["total_contradictions"] = self.record_metric(
            profile_name,
            ProfileCategory.CONSISTENCY,
            "total_contradictions",
            guard_result["contradictions_total"]
        )
        
        # Determine if any violations occurred
        violations = [r for r in results.values() if r.get("violation", False)]
        
        return {
            "guard_result": guard_result,
            "metrics_results": results,
            "violations": violations,
            "has_violations": len(violations) > 0
        }
    
    def get_violation_history(self, 
                            profile_name: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history of violations.
        
        Args:
            profile_name: Filter by profile name
            limit: Maximum number of violations to return
            
        Returns:
            List of violation records
        """
        with self._lock:
            if profile_name:
                filtered = [v for v in self.violation_history 
                          if v["profile_name"] == profile_name]
                return filtered[-limit:]
                
            return self.violation_history[-limit:]
            
    def export_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Export all profiles.
        
        Returns:
            Dict of profile name to profile data
        """
        with self._lock:
            return {name: profile.to_dict() for name, profile in self.profiles.items()}
            
    def import_profiles(self, profiles_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Import profiles from exported data.
        
        Args:
            profiles_data: Dict of profile name to profile data
        """
        with self._lock:
            for name, data in profiles_data.items():
                self.profiles[name] = SafetyProfile.from_dict(data)
                
    def reset(self) -> None:
        """Reset the analyzer state."""
        with self._lock:
            self.profiles = {}
            self.violation_history = []


def create_default_analyzer(reflection_guard: Optional[ReflectionGuard] = None) -> ProfileAnalyzer:
    """
    Create a ProfileAnalyzer with default profiles.
    
    Args:
        reflection_guard: ReflectionGuard to use
        
    Returns:
        Configured ProfileAnalyzer
    """
    analyzer = ProfileAnalyzer(reflection_guard)
    
    # Create standard profiles
    base_profile = analyzer.create_profile("base")
    
    # Set up reasoning thresholds
    base_profile.set_threshold(
        ProfileCategory.REASONING,
        "reasoning_cycles",
        max_value=0.5  # Allow occasional cycles
    )
    
    base_profile.set_threshold(
        ProfileCategory.REASONING,
        "total_cycles",
        max_value=5  # Max total cycles
    )
    
    # Set up consistency thresholds
    base_profile.set_threshold(
        ProfileCategory.CONSISTENCY,
        "contradictions",
        max_value=0.3  # Allow rare contradictions
    )
    
    base_profile.set_threshold(
        ProfileCategory.CONSISTENCY,
        "total_contradictions",
        max_value=3  # Max total contradictions
    )
    
    # Set up resource thresholds
    base_profile.set_threshold(
        ProfileCategory.RESOURCES,
        "response_time",
        max_value=5.0  # Max response time in seconds
    )
    
    base_profile.set_threshold(
        ProfileCategory.RESOURCES,
        "token_usage_ratio",
        max_value=0.9  # Max token usage ratio
    )
    
    return analyzer 