"""
Computational Load Profiler

This module provides tools for measuring the computational resource utilization
of different components within the ΨC-AI SDK, identifying performance bottlenecks,
and optimizing resource allocation.

The profiler tracks CPU time, memory usage, and latency for each module and method,
allowing developers to identify which components are consuming the most resources
and optimize accordingly.
"""

import time
import logging
import gc
import functools
import threading
import inspect
import os
import psutil
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from datetime import datetime
import traceback
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import memory_profiler
    _HAS_MEMORY_PROFILER = True
except ImportError:
    _HAS_MEMORY_PROFILER = False
    logger.warning("memory_profiler package not available. Memory usage tracking will be limited.")


@dataclass
class ExecutionProfile:
    """Profile of a single execution of a module or method."""
    module_name: str
    method_name: Optional[str] = None
    execution_time: float = 0.0  # in seconds
    start_memory: float = 0.0    # in MB
    end_memory: float = 0.0      # in MB
    memory_delta: float = 0.0    # in MB
    cpu_usage: float = 0.0       # in percent (0-100)
    thread_id: str = ""
    timestamp: float = field(default_factory=time.time)
    args_summary: Optional[str] = None
    success: bool = True
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "module_name": self.module_name,
            "method_name": self.method_name,
            "execution_time": self.execution_time,
            "start_memory": self.start_memory,
            "end_memory": self.end_memory,
            "memory_delta": self.memory_delta,
            "cpu_usage": self.cpu_usage,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp,
            "timestamp_readable": datetime.fromtimestamp(self.timestamp).isoformat(),
            "args_summary": self.args_summary,
            "success": self.success,
            "error_type": self.error_type
        }


@dataclass
class ModuleProfile:
    """Aggregated profile data for a module."""
    module_name: str
    call_count: int = 0
    total_execution_time: float = 0.0  # in seconds
    avg_execution_time: float = 0.0    # in seconds
    min_execution_time: float = float('inf')  # in seconds
    max_execution_time: float = 0.0    # in seconds
    total_memory_delta: float = 0.0    # in MB
    avg_memory_delta: float = 0.0      # in MB
    avg_cpu_usage: float = 0.0         # in percent (0-100)
    error_count: int = 0
    method_profiles: Dict[str, 'ModuleProfile'] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        result = {
            "module_name": self.module_name,
            "call_count": self.call_count,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": self.avg_execution_time,
            "min_execution_time": self.min_execution_time if self.min_execution_time != float('inf') else 0,
            "max_execution_time": self.max_execution_time,
            "total_memory_delta": self.total_memory_delta,
            "avg_memory_delta": self.avg_memory_delta,
            "avg_cpu_usage": self.avg_cpu_usage,
            "error_count": self.error_count,
        }
        
        if self.method_profiles:
            result["method_profiles"] = {
                method_name: profile.to_dict() 
                for method_name, profile in self.method_profiles.items()
            }
        
        return result


class ComputationalLoadProfiler:
    """
    Profiles computational load across ΨC-AI SDK components.
    
    This class provides tools for tracking and analyzing the computational
    resources used by different modules and methods, helping identify 
    performance bottlenecks.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        trace_memory: bool = True,
        trace_cpu: bool = True,
        max_profiles: int = 1000,
        save_dir: Optional[Union[str, Path]] = None,
        auto_save: bool = False,
        auto_save_interval: int = 60  # in seconds
    ):
        """
        Initialize the computational load profiler.
        
        Args:
            enabled: Whether profiling is enabled
            trace_memory: Whether to trace memory usage
            trace_cpu: Whether to trace CPU usage
            max_profiles: Maximum number of individual execution profiles to keep
            save_dir: Directory to save profiles to
            auto_save: Whether to automatically save profiles
            auto_save_interval: How often to auto-save profiles (in seconds)
        """
        self.enabled = enabled
        self.trace_memory = trace_memory and _HAS_MEMORY_PROFILER
        self.trace_cpu = trace_cpu
        self.max_profiles = max_profiles
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Execution profiles (individual runs)
        self.execution_profiles: List[ExecutionProfile] = []
        
        # Aggregated profiles by module
        self.module_profiles: Dict[str, ModuleProfile] = {}
        
        # Process info
        self.process = psutil.Process(os.getpid())
        
        # Auto-save
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        self._last_save_time = time.time()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Start auto-save thread if needed
        if self.auto_save:
            self._start_auto_save()
    
    def _start_auto_save(self) -> None:
        """Start a background thread to periodically save profiles."""
        def save_worker():
            while True:
                time.sleep(self.auto_save_interval)
                try:
                    current_time = time.time()
                    if current_time - self._last_save_time >= self.auto_save_interval:
                        self.save_profiles()
                        self._last_save_time = current_time
                except Exception as e:
                    logger.error(f"Error in auto-save: {e}")
        
        thread = threading.Thread(target=save_worker, daemon=True)
        thread.start()
        logger.debug("Started auto-save thread")
    
    @contextmanager
    def profile_execution(
        self, 
        module_name: str, 
        method_name: Optional[str] = None,
        args_summary: Optional[str] = None
    ):
        """
        Context manager for profiling execution of a code block.
        
        Args:
            module_name: Name of the module being profiled
            method_name: Name of the method being profiled (optional)
            args_summary: Summary of arguments (optional)
            
        Example:
            ```python
            with profiler.profile_execution("my_module", "my_method"):
                # Code to profile
                result = complex_computation()
            ```
        """
        if not self.enabled:
            yield
            return
        
        thread_id = threading.get_ident()
        start_time = time.time()
        
        # Get initial resource usage
        start_memory = self._get_memory_usage()
        start_cpu = self.process.cpu_percent()
        
        success = True
        error_type = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            # Get final resource usage
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self.process.cpu_percent()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2  # Average CPU usage
            
            # Create execution profile
            profile = ExecutionProfile(
                module_name=module_name,
                method_name=method_name,
                execution_time=execution_time,
                start_memory=start_memory,
                end_memory=end_memory,
                memory_delta=memory_delta,
                cpu_usage=cpu_usage,
                thread_id=str(thread_id),
                args_summary=args_summary,
                success=success,
                error_type=error_type
            )
            
            # Record the profile
            self._record_profile(profile)
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        if _HAS_MEMORY_PROFILER and self.trace_memory:
            return memory_profiler.memory_usage()[0]
        else:
            # Fallback using psutil
            return self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    def _record_profile(self, profile: ExecutionProfile) -> None:
        """
        Record an execution profile.
        
        Args:
            profile: The execution profile to record
        """
        with self.lock:
            # Add to execution profiles
            self.execution_profiles.append(profile)
            
            # Trim if necessary
            if len(self.execution_profiles) > self.max_profiles:
                self.execution_profiles = self.execution_profiles[-self.max_profiles:]
            
            # Update aggregated profiles
            self._update_aggregated_profiles(profile)
    
    def _update_aggregated_profiles(self, profile: ExecutionProfile) -> None:
        """
        Update aggregated module profiles with a new execution profile.
        
        Args:
            profile: The execution profile to incorporate
        """
        # Get or create module profile
        if profile.module_name not in self.module_profiles:
            self.module_profiles[profile.module_name] = ModuleProfile(module_name=profile.module_name)
        
        module_profile = self.module_profiles[profile.module_name]
        
        # Update module profile
        module_profile.call_count += 1
        module_profile.total_execution_time += profile.execution_time
        module_profile.avg_execution_time = module_profile.total_execution_time / module_profile.call_count
        module_profile.min_execution_time = min(module_profile.min_execution_time, profile.execution_time)
        module_profile.max_execution_time = max(module_profile.max_execution_time, profile.execution_time)
        module_profile.total_memory_delta += profile.memory_delta
        module_profile.avg_memory_delta = module_profile.total_memory_delta / module_profile.call_count
        
        # Update CPU usage as rolling average
        module_profile.avg_cpu_usage = (
            (module_profile.avg_cpu_usage * (module_profile.call_count - 1) + profile.cpu_usage) / 
            module_profile.call_count
        )
        
        if not profile.success:
            module_profile.error_count += 1
        
        # If method_name is specified, also update method profile
        if profile.method_name:
            if profile.method_name not in module_profile.method_profiles:
                module_profile.method_profiles[profile.method_name] = ModuleProfile(
                    module_name=f"{profile.module_name}.{profile.method_name}"
                )
            
            method_profile = module_profile.method_profiles[profile.method_name]
            
            # Update method profile (same as module profile)
            method_profile.call_count += 1
            method_profile.total_execution_time += profile.execution_time
            method_profile.avg_execution_time = method_profile.total_execution_time / method_profile.call_count
            method_profile.min_execution_time = min(method_profile.min_execution_time, profile.execution_time)
            method_profile.max_execution_time = max(method_profile.max_execution_time, profile.execution_time)
            method_profile.total_memory_delta += profile.memory_delta
            method_profile.avg_memory_delta = method_profile.total_memory_delta / method_profile.call_count
            method_profile.avg_cpu_usage = (
                (method_profile.avg_cpu_usage * (method_profile.call_count - 1) + profile.cpu_usage) / 
                method_profile.call_count
            )
            
            if not profile.success:
                method_profile.error_count += 1
    
    def save_profiles(self, filepath: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Save profiles to a JSON file.
        
        Args:
            filepath: Path to save to (if None, a timestamped file in save_dir is used)
            
        Returns:
            Path to the saved file, or None if saving failed
        """
        with self.lock:
            if not filepath:
                if not self.save_dir:
                    logger.error("Cannot save profiles: no save_dir specified")
                    return None
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.save_dir / f"performance_profile_{timestamp}.json"
            else:
                filepath = Path(filepath)
            
            try:
                # Create export data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module_profiles": {
                        module_name: profile.to_dict() 
                        for module_name, profile in self.module_profiles.items()
                    },
                    "execution_profiles": [
                        profile.to_dict() for profile in self.execution_profiles
                    ]
                }
                
                # Save to file
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Saved performance profiles to {filepath}")
                return filepath
            
            except Exception as e:
                logger.error(f"Failed to save profiles: {e}")
                return None
    
    def load_profiles(self, filepath: Union[str, Path]) -> bool:
        """
        Load profiles from a JSON file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Profile file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Process module profiles
            module_profiles = {}
            for module_name, profile_data in data.get("module_profiles", {}).items():
                module_profile = ModuleProfile(module_name=module_name)
                
                # Copy simple attributes
                for attr in ["call_count", "total_execution_time", "avg_execution_time",
                            "min_execution_time", "max_execution_time", "total_memory_delta",
                            "avg_memory_delta", "avg_cpu_usage", "error_count"]:
                    if attr in profile_data:
                        setattr(module_profile, attr, profile_data[attr])
                
                # Process method profiles
                for method_name, method_data in profile_data.get("method_profiles", {}).items():
                    method_profile = ModuleProfile(module_name=f"{module_name}.{method_name}")
                    
                    # Copy simple attributes
                    for attr in ["call_count", "total_execution_time", "avg_execution_time",
                                "min_execution_time", "max_execution_time", "total_memory_delta",
                                "avg_memory_delta", "avg_cpu_usage", "error_count"]:
                        if attr in method_data:
                            setattr(method_profile, attr, method_data[attr])
                    
                    module_profile.method_profiles[method_name] = method_profile
                
                module_profiles[module_name] = module_profile
            
            # Process execution profiles
            execution_profiles = []
            for profile_data in data.get("execution_profiles", []):
                profile = ExecutionProfile(
                    module_name=profile_data.get("module_name", ""),
                    method_name=profile_data.get("method_name"),
                    execution_time=profile_data.get("execution_time", 0.0),
                    start_memory=profile_data.get("start_memory", 0.0),
                    end_memory=profile_data.get("end_memory", 0.0),
                    memory_delta=profile_data.get("memory_delta", 0.0),
                    cpu_usage=profile_data.get("cpu_usage", 0.0),
                    thread_id=profile_data.get("thread_id", ""),
                    timestamp=profile_data.get("timestamp", 0.0),
                    args_summary=profile_data.get("args_summary"),
                    success=profile_data.get("success", True),
                    error_type=profile_data.get("error_type")
                )
                execution_profiles.append(profile)
            
            # Update internal state
            with self.lock:
                self.module_profiles = module_profiles
                self.execution_profiles = execution_profiles
            
            logger.info(f"Loaded performance profiles from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return False
    
    def identify_bottlenecks(
        self, 
        execution_time_threshold: float = 0.1,  # in seconds
        memory_delta_threshold: float = 10.0,   # in MB
        cpu_usage_threshold: float = 50.0       # in percent
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify performance bottlenecks based on thresholds.
        
        Args:
            execution_time_threshold: Threshold for execution time (seconds)
            memory_delta_threshold: Threshold for memory delta (MB)
            cpu_usage_threshold: Threshold for CPU usage (percent)
            
        Returns:
            Dictionary mapping bottleneck type to list of problematic modules/methods
        """
        bottlenecks = {
            "execution_time": [],
            "memory_usage": [],
            "cpu_usage": []
        }
        
        with self.lock:
            # Check module profiles
            for module_name, profile in self.module_profiles.items():
                # Check execution time
                if profile.avg_execution_time >= execution_time_threshold:
                    bottlenecks["execution_time"].append({
                        "module": module_name,
                        "method": None,
                        "value": profile.avg_execution_time,
                        "unit": "seconds",
                        "call_count": profile.call_count
                    })
                
                # Check memory delta
                if profile.avg_memory_delta >= memory_delta_threshold:
                    bottlenecks["memory_usage"].append({
                        "module": module_name,
                        "method": None,
                        "value": profile.avg_memory_delta,
                        "unit": "MB",
                        "call_count": profile.call_count
                    })
                
                # Check CPU usage
                if profile.avg_cpu_usage >= cpu_usage_threshold:
                    bottlenecks["cpu_usage"].append({
                        "module": module_name,
                        "method": None,
                        "value": profile.avg_cpu_usage,
                        "unit": "percent",
                        "call_count": profile.call_count
                    })
                
                # Check method profiles
                for method_name, method_profile in profile.method_profiles.items():
                    # Check execution time
                    if method_profile.avg_execution_time >= execution_time_threshold:
                        bottlenecks["execution_time"].append({
                            "module": module_name,
                            "method": method_name,
                            "value": method_profile.avg_execution_time,
                            "unit": "seconds",
                            "call_count": method_profile.call_count
                        })
                    
                    # Check memory delta
                    if method_profile.avg_memory_delta >= memory_delta_threshold:
                        bottlenecks["memory_usage"].append({
                            "module": module_name,
                            "method": method_name,
                            "value": method_profile.avg_memory_delta,
                            "unit": "MB",
                            "call_count": method_profile.call_count
                        })
                    
                    # Check CPU usage
                    if method_profile.avg_cpu_usage >= cpu_usage_threshold:
                        bottlenecks["cpu_usage"].append({
                            "module": module_name,
                            "method": method_name,
                            "value": method_profile.avg_cpu_usage,
                            "unit": "percent",
                            "call_count": method_profile.call_count
                        })
        
        # Sort bottlenecks by value (descending)
        for bottleneck_type in bottlenecks:
            bottlenecks[bottleneck_type].sort(key=lambda x: x["value"], reverse=True)
        
        return bottlenecks
    
    def get_module_profile(self, module_name: str) -> Optional[ModuleProfile]:
        """
        Get profile data for a specific module.
        
        Args:
            module_name: Name of the module to get profile for
            
        Returns:
            ModuleProfile for the module, or None if not found
        """
        with self.lock:
            return self.module_profiles.get(module_name)
    
    def get_method_profile(
        self, 
        module_name: str, 
        method_name: str
    ) -> Optional[ModuleProfile]:
        """
        Get profile data for a specific method.
        
        Args:
            module_name: Name of the module
            method_name: Name of the method
            
        Returns:
            ModuleProfile for the method, or None if not found
        """
        with self.lock:
            module_profile = self.module_profiles.get(module_name)
            if not module_profile:
                return None
            
            return module_profile.method_profiles.get(method_name)
    
    def get_recent_executions(
        self,
        module_name: Optional[str] = None,
        method_name: Optional[str] = None,
        limit: int = 10
    ) -> List[ExecutionProfile]:
        """
        Get recent execution profiles.
        
        Args:
            module_name: Filter by module name (optional)
            method_name: Filter by method name (optional)
            limit: Maximum number of profiles to return
            
        Returns:
            List of matching execution profiles
        """
        with self.lock:
            # Apply filters
            filtered_profiles = self.execution_profiles
            
            if module_name:
                filtered_profiles = [p for p in filtered_profiles if p.module_name == module_name]
            
            if method_name:
                filtered_profiles = [p for p in filtered_profiles if p.method_name == method_name]
            
            # Sort by timestamp (newest first) and limit
            return sorted(filtered_profiles, key=lambda p: p.timestamp, reverse=True)[:limit]
    
    def get_module_ranking(
        self,
        metric: str = "execution_time",
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get modules ranked by a specific metric.
        
        Args:
            metric: Metric to rank by ("execution_time", "memory_delta", "cpu_usage")
            limit: Maximum number of modules to return
            
        Returns:
            List of (module_name, metric_value) tuples
        """
        with self.lock:
            if metric == "execution_time":
                key_func = lambda p: p.avg_execution_time
            elif metric == "memory_delta":
                key_func = lambda p: p.avg_memory_delta
            elif metric == "cpu_usage":
                key_func = lambda p: p.avg_cpu_usage
            else:
                logger.error(f"Unknown metric: {metric}")
                return []
            
            # Rank modules
            ranked_modules = sorted(
                [(name, key_func(profile)) for name, profile in self.module_profiles.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            return ranked_modules[:limit]
    
    def reset(self) -> None:
        """Reset all profiles."""
        with self.lock:
            self.execution_profiles = []
            self.module_profiles = {}
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with report data
        """
        with self.lock:
            # Get bottlenecks
            bottlenecks = self.identify_bottlenecks()
            
            # Get top modules by execution time
            top_modules_time = self.get_module_ranking(metric="execution_time")
            top_modules_memory = self.get_module_ranking(metric="memory_delta")
            top_modules_cpu = self.get_module_ranking(metric="cpu_usage")
            
            # Calculate overall stats
            total_execution_time = sum(
                profile.total_execution_time for profile in self.module_profiles.values()
            )
            total_memory_delta = sum(
                profile.total_memory_delta for profile in self.module_profiles.values()
            )
            avg_cpu_usage = sum(
                profile.avg_cpu_usage * profile.call_count 
                for profile in self.module_profiles.values()
            ) / sum(
                profile.call_count for profile in self.module_profiles.values()
            ) if self.module_profiles else 0
            
            # Get error counts
            error_counts = {
                module_name: profile.error_count 
                for module_name, profile in self.module_profiles.items()
                if profile.error_count > 0
            }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_stats": {
                    "total_module_count": len(self.module_profiles),
                    "total_execution_time": total_execution_time,
                    "total_memory_delta": total_memory_delta,
                    "avg_cpu_usage": avg_cpu_usage,
                    "total_error_count": sum(error_counts.values()) if error_counts else 0
                },
                "bottlenecks": bottlenecks,
                "top_modules": {
                    "execution_time": top_modules_time,
                    "memory_delta": top_modules_memory,
                    "cpu_usage": top_modules_cpu
                },
                "error_counts": error_counts
            }


# Decorator for profiling functions
def profile(module_name: Optional[str] = None):
    """
    Decorator for profiling functions.
    
    Args:
        module_name: Name of the module (defaults to module of decorated function)
        
    Returns:
        Decorated function
        
    Example:
        ```python
        @profile("my_module")
        def my_function():
            # Code to profile
            pass
        ```
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get global profiler instance
            profiler = get_profiler()
            
            if not profiler.enabled:
                return func(*args, **kwargs)
            
            # Get module name
            nonlocal module_name
            if module_name is None:
                module_name = func.__module__
            
            # Generate args summary
            try:
                args_str = ", ".join([str(arg)[:20] for arg in args])
                kwargs_str = ", ".join([f"{k}={str(v)[:20]}" for k, v in kwargs.items()])
                args_summary = f"{args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str}"
                if len(args_summary) > 100:
                    args_summary = args_summary[:97] + "..."
            except:
                args_summary = "error generating args summary"
            
            # Profile execution
            with profiler.profile_execution(
                module_name=module_name,
                method_name=func.__name__,
                args_summary=args_summary
            ):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


# Global profiler instance
_global_profiler: Optional[ComputationalLoadProfiler] = None

def get_profiler() -> ComputationalLoadProfiler:
    """
    Get the global profiler instance.
    
    Returns:
        Global ComputationalLoadProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = ComputationalLoadProfiler()
    return _global_profiler

def set_profiler(profiler: ComputationalLoadProfiler) -> None:
    """
    Set the global profiler instance.
    
    Args:
        profiler: ComputationalLoadProfiler instance to use
    """
    global _global_profiler
    _global_profiler = profiler 