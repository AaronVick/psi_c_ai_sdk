"""
Reflection Scheduler for ΨC-AI SDK

This module provides advanced scheduling logic for triggering reflection
cycles based on system events, cognitive debt, and epistemic uncertainty.
"""

import logging
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
import uuid
import heapq
import threading

from psi_c_ai_sdk.reflection.reflection_engine import (
    ReflectionEngine, ReflectionState, ReflectionOutcome
)
from psi_c_ai_sdk.epistemic.epistemic_status import (
    EpistemicStatus, EpistemicState, ReflectionTrigger
)

# Configure logging
logger = logging.getLogger(__name__)


class SchedulePolicy(Enum):
    """Policy for scheduling reflection."""
    
    ADAPTIVE = auto()       # Adapt based on system load and importance
    IMMEDIATE = auto()      # Schedule immediately
    BACKGROUND = auto()     # Schedule in background when resources available
    PERIODIC = auto()       # Schedule at fixed intervals
    THRESHOLD = auto()      # Schedule when threshold reached


class ReflectionPriority(Enum):
    """Priority levels for reflection tasks."""
    
    CRITICAL = 10     # Must be processed immediately
    HIGH = 7          # Process as soon as possible
    MEDIUM = 5        # Standard priority
    LOW = 3           # Process when resources available
    BACKGROUND = 1    # Process only when system is idle


class ScheduledReflection:
    """A scheduled reflection task."""
    
    def __init__(
        self,
        id: str,
        trigger: ReflectionTrigger,
        priority: ReflectionPriority,
        scheduled_time: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a scheduled reflection.
        
        Args:
            id: Unique identifier
            trigger: What triggered the reflection
            priority: Priority level
            scheduled_time: When to execute
            metadata: Additional metadata
        """
        self.id = id
        self.trigger = trigger
        self.priority = priority
        self.scheduled_time = scheduled_time
        self.metadata = metadata or {}
        self.executed = False
        self.execution_time: Optional[datetime] = None
        self.outcome_id: Optional[str] = None
    
    def __lt__(self, other):
        """
        Compare two scheduled reflections for priority queue.
        
        The comparison is first by priority, then by scheduled time.
        """
        # First compare by priority (higher number = higher priority)
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        
        # Then compare by scheduled time (earlier = higher priority)
        return self.scheduled_time < other.scheduled_time


class AdvancedReflectionScheduler:
    """
    Advanced scheduler for reflection cycles with priority-based scheduling.
    
    This scheduler manages reflection cycles based on priority, system load,
    and scheduled times. It supports various scheduling policies and can
    adapt to changing system conditions.
    """
    
    def __init__(
        self,
        reflection_engine: ReflectionEngine,
        epistemic_status: EpistemicStatus,
        default_policy: SchedulePolicy = SchedulePolicy.ADAPTIVE,
        max_scheduled: int = 100,
        check_interval: timedelta = timedelta(seconds=30),
        background_interval: timedelta = timedelta(minutes=30),
        system_load_threshold: float = 0.7,
        enable_background_thread: bool = False
    ):
        """
        Initialize the advanced reflection scheduler.
        
        Args:
            reflection_engine: Reflection engine to schedule
            epistemic_status: Epistemic status to monitor
            default_policy: Default scheduling policy
            max_scheduled: Maximum number of scheduled reflections
            check_interval: How often to check for scheduled reflections
            background_interval: Interval for background reflections
            system_load_threshold: Threshold for system load (0-1)
            enable_background_thread: Whether to run in background thread
        """
        self.reflection_engine = reflection_engine
        self.epistemic_status = epistemic_status
        self.default_policy = default_policy
        self.max_scheduled = max_scheduled
        self.check_interval = check_interval
        self.background_interval = background_interval
        self.system_load_threshold = system_load_threshold
        
        # Priority queue for scheduled reflections
        self.scheduled_queue: List[ScheduledReflection] = []
        
        # Track executed reflections
        self.executed: Dict[str, ScheduledReflection] = {}
        
        # Track statistics
        self.total_scheduled = 0
        self.total_executed = 0
        self.total_missed = 0
        self.last_check_time = datetime.now()
        self.last_execution_time: Optional[datetime] = None
        
        # Background thread
        self.enable_background_thread = enable_background_thread
        self.background_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.pre_execution_callbacks: List[Callable[[ScheduledReflection], None]] = []
        self.post_execution_callbacks: List[Callable[[ScheduledReflection, Optional[str]], None]] = []
        
        # Start background thread if enabled
        if self.enable_background_thread:
            self._start_background_thread()
    
    def _get_system_load(self) -> float:
        """
        Get current system load.
        
        Returns:
            System load value between 0-1
        """
        # This is a placeholder - in a real implementation,
        # this would check CPU, memory, and other resources
        # For now, return a fixed value
        return 0.3
    
    def _can_execute(self, priority: ReflectionPriority) -> bool:
        """
        Check if a reflection with given priority can be executed.
        
        Args:
            priority: Priority level to check
            
        Returns:
            Whether execution is allowed
        """
        # Critical priority always executes
        if priority == ReflectionPriority.CRITICAL:
            return True
            
        # Check if engine is already reflecting
        if self.reflection_engine.current_state != ReflectionState.IDLE:
            return False
            
        # Check system load for lower priorities
        system_load = self._get_system_load()
        
        if priority == ReflectionPriority.HIGH:
            # Allow HIGH priority unless system is very loaded
            return system_load < 0.9
        elif priority == ReflectionPriority.MEDIUM:
            # Medium priority needs moderate load
            return system_load < self.system_load_threshold
        elif priority == ReflectionPriority.LOW:
            # Low priority needs low load
            return system_load < 0.4
        elif priority == ReflectionPriority.BACKGROUND:
            # Background priority needs very low load
            return system_load < 0.2
            
        return False
    
    def _get_priority_for_trigger(self, trigger: ReflectionTrigger) -> ReflectionPriority:
        """
        Determine priority level for a trigger.
        
        Args:
            trigger: Reflection trigger
            
        Returns:
            Appropriate priority level
        """
        if trigger == ReflectionTrigger.CRITICAL_CONTRADICTION:
            return ReflectionPriority.CRITICAL
        elif trigger == ReflectionTrigger.MAJOR_CONTRADICTION:
            return ReflectionPriority.HIGH
        elif trigger == ReflectionTrigger.CERTAINTY_THRESHOLD:
            return ReflectionPriority.MEDIUM
        elif trigger == ReflectionTrigger.STABILITY_THRESHOLD:
            return ReflectionPriority.MEDIUM
        elif trigger == ReflectionTrigger.PERIODIC:
            return ReflectionPriority.LOW
        elif trigger == ReflectionTrigger.MANUAL:
            return ReflectionPriority.HIGH  # Manual is usually important
        else:
            return ReflectionPriority.MEDIUM
    
    def _get_delay_for_priority(self, priority: ReflectionPriority) -> timedelta:
        """
        Get delay time for a priority level.
        
        Args:
            priority: Priority level
            
        Returns:
            Time to delay execution
        """
        if priority == ReflectionPriority.CRITICAL:
            return timedelta(seconds=0)  # Immediate
        elif priority == ReflectionPriority.HIGH:
            return timedelta(seconds=10)
        elif priority == ReflectionPriority.MEDIUM:
            return timedelta(minutes=1)
        elif priority == ReflectionPriority.LOW:
            return timedelta(minutes=5)
        elif priority == ReflectionPriority.BACKGROUND:
            return timedelta(minutes=15)
        
        # Default
        return timedelta(minutes=1)
    
    def schedule_reflection(
        self,
        trigger: ReflectionTrigger,
        policy: Optional[SchedulePolicy] = None,
        priority: Optional[ReflectionPriority] = None,
        scheduled_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Schedule a reflection cycle.
        
        Args:
            trigger: What triggered the reflection
            policy: Scheduling policy to use
            priority: Priority level (if None, determined from trigger)
            scheduled_time: When to execute (if None, determined from policy)
            metadata: Additional metadata
            
        Returns:
            ID of scheduled reflection, or None if scheduling failed
        """
        # Use default policy if not specified
        if policy is None:
            policy = self.default_policy
            
        # Determine priority if not specified
        if priority is None:
            priority = self._get_priority_for_trigger(trigger)
            
        # Determine scheduled time based on policy and priority
        if scheduled_time is None:
            if policy == SchedulePolicy.IMMEDIATE:
                scheduled_time = datetime.now()
            elif policy == SchedulePolicy.BACKGROUND:
                # Add a longer delay for background tasks
                scheduled_time = datetime.now() + timedelta(minutes=5)
            elif policy == SchedulePolicy.PERIODIC:
                # Schedule for next interval
                if self.last_execution_time:
                    scheduled_time = self.last_execution_time + self.background_interval
                else:
                    scheduled_time = datetime.now() + self.background_interval
            else:  # ADAPTIVE or THRESHOLD
                # Calculate delay based on priority
                delay = self._get_delay_for_priority(priority)
                scheduled_time = datetime.now() + delay
                
        # Create scheduled reflection
        reflection_id = f"scheduled_{int(time.time())}_{self.total_scheduled}"
        scheduled = ScheduledReflection(
            id=reflection_id,
            trigger=trigger,
            priority=priority,
            scheduled_time=scheduled_time,
            metadata=metadata
        )
        
        # Add to priority queue
        heapq.heappush(self.scheduled_queue, scheduled)
        
        # Update statistics
        self.total_scheduled += 1
        
        # Check if immediate execution is needed
        if policy == SchedulePolicy.IMMEDIATE and self._can_execute(priority):
            self._execute_next_scheduled()
            
        # Limit queue size if needed
        if len(self.scheduled_queue) > self.max_scheduled:
            self._prune_scheduled()
            
        logger.info(f"Scheduled reflection {reflection_id} for {scheduled_time.isoformat()} with priority {priority.name}")
        
        return reflection_id
    
    def check_and_execute(self) -> Optional[str]:
        """
        Check scheduled reflections and execute if appropriate.
        
        Returns:
            ID of executed reflection outcome, or None if no execution
        """
        # Update last check time
        self.last_check_time = datetime.now()
        
        # Check if there are any scheduled reflections
        if not self.scheduled_queue:
            return None
            
        # Check if next scheduled reflection is due
        next_scheduled = self.scheduled_queue[0]
        
        if next_scheduled.scheduled_time <= datetime.now():
            # Check if execution is allowed
            if self._can_execute(next_scheduled.priority):
                return self._execute_next_scheduled()
            else:
                # Skip if higher priority or system load prevents execution
                if next_scheduled.priority == ReflectionPriority.CRITICAL:
                    # Critical reflections should never be skipped, log warning
                    logger.warning(f"Critical reflection {next_scheduled.id} delayed due to system state")
                    
                # Skip low priority reflections that are significantly past due
                if (next_scheduled.priority in 
                    [ReflectionPriority.LOW, ReflectionPriority.BACKGROUND] and
                    datetime.now() - next_scheduled.scheduled_time > timedelta(hours=1)):
                    
                    # Remove from queue
                    heapq.heappop(self.scheduled_queue)
                    self.total_missed += 1
                    logger.info(f"Skipped low priority reflection {next_scheduled.id} due to being past due")
                
        return None
    
    def _execute_next_scheduled(self) -> Optional[str]:
        """
        Execute the next scheduled reflection.
        
        Returns:
            ID of executed reflection outcome, or None if execution failed
        """
        # Get next scheduled reflection
        if not self.scheduled_queue:
            return None
            
        scheduled = heapq.heappop(self.scheduled_queue)
        
        # Execute callbacks before execution
        for callback in self.pre_execution_callbacks:
            try:
                callback(scheduled)
            except Exception as e:
                logger.error(f"Error in pre-execution callback: {str(e)}", exc_info=True)
        
        # Execute reflection
        logger.info(f"Executing scheduled reflection {scheduled.id} with priority {scheduled.priority.name}")
        
        # Trigger reflection in engine
        outcome_id = self.reflection_engine.trigger_reflection(scheduled.trigger)
        
        if outcome_id:
            # Run reflection cycle
            outcome = self.reflection_engine.run_reflection_cycle()
            
            # Update statistics
            scheduled.executed = True
            scheduled.execution_time = datetime.now()
            scheduled.outcome_id = outcome_id if outcome else None
            
            self.executed[scheduled.id] = scheduled
            self.total_executed += 1
            self.last_execution_time = datetime.now()
            
            # Execute callbacks after execution
            for callback in self.post_execution_callbacks:
                try:
                    callback(scheduled, outcome_id)
                except Exception as e:
                    logger.error(f"Error in post-execution callback: {str(e)}", exc_info=True)
            
            return outcome_id
        else:
            logger.error(f"Failed to trigger reflection for scheduled {scheduled.id}")
            self.total_missed += 1
            return None
    
    def _prune_scheduled(self) -> None:
        """
        Prune the scheduled reflection queue to maintain size limit.
        
        This removes the lowest priority, furthest scheduled reflections.
        """
        if len(self.scheduled_queue) <= self.max_scheduled:
            return
            
        # Find reflections to remove (keep the ones with highest priority,
        # then earliest scheduled)
        to_remove = len(self.scheduled_queue) - self.max_scheduled
        
        # Need to rebuild the heap without the items to remove
        temp = sorted(self.scheduled_queue)  # Sorted by priority, then time
        self.scheduled_queue = temp[:len(temp) - to_remove]
        heapq.heapify(self.scheduled_queue)
        
        logger.info(f"Pruned {to_remove} scheduled reflections to maintain queue size")
        self.total_missed += to_remove
    
    def get_scheduled_by_id(self, scheduled_id: str) -> Optional[ScheduledReflection]:
        """
        Get a scheduled reflection by ID.
        
        Args:
            scheduled_id: ID of the scheduled reflection
            
        Returns:
            ScheduledReflection if found, None otherwise
        """
        # Check executed first
        if scheduled_id in self.executed:
            return self.executed[scheduled_id]
            
        # Check scheduled queue
        for scheduled in self.scheduled_queue:
            if scheduled.id == scheduled_id:
                return scheduled
                
        return None
    
    def cancel_scheduled(self, scheduled_id: str) -> bool:
        """
        Cancel a scheduled reflection.
        
        Args:
            scheduled_id: ID of the scheduled reflection
            
        Returns:
            Whether cancellation was successful
        """
        # Find in scheduled queue
        for i, scheduled in enumerate(self.scheduled_queue):
            if scheduled.id == scheduled_id:
                # Remove from queue (need to rebuild heap)
                self.scheduled_queue.pop(i)
                heapq.heapify(self.scheduled_queue)
                logger.info(f"Cancelled scheduled reflection {scheduled_id}")
                return True
                
        logger.warning(f"Could not find scheduled reflection {scheduled_id} to cancel")
        return False
    
    def reschedule(
        self,
        scheduled_id: str,
        new_time: datetime,
        new_priority: Optional[ReflectionPriority] = None
    ) -> bool:
        """
        Reschedule a reflection to a new time.
        
        Args:
            scheduled_id: ID of the scheduled reflection
            new_time: New scheduled time
            new_priority: New priority (if None, keep existing)
            
        Returns:
            Whether rescheduling was successful
        """
        # Find in scheduled queue
        for i, scheduled in enumerate(self.scheduled_queue):
            if scheduled.id == scheduled_id:
                # Update scheduled time
                scheduled.scheduled_time = new_time
                
                # Update priority if specified
                if new_priority is not None:
                    scheduled.priority = new_priority
                    
                # Rebuild heap
                heapq.heapify(self.scheduled_queue)
                
                logger.info(f"Rescheduled reflection {scheduled_id} to {new_time.isoformat()}")
                return True
                
        logger.warning(f"Could not find scheduled reflection {scheduled_id} to reschedule")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the scheduler.
        
        Returns:
            Dictionary with scheduler statistics
        """
        return {
            "total_scheduled": self.total_scheduled,
            "total_executed": self.total_executed,
            "total_missed": self.total_missed,
            "queue_size": len(self.scheduled_queue),
            "last_check_time": self.last_check_time.isoformat(),
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "next_scheduled": self.scheduled_queue[0].scheduled_time.isoformat() if self.scheduled_queue else None,
            "next_trigger": self.scheduled_queue[0].trigger.name if self.scheduled_queue else None,
            "next_priority": self.scheduled_queue[0].priority.name if self.scheduled_queue else None
        }
    
    def add_pre_execution_callback(self, callback: Callable[[ScheduledReflection], None]) -> None:
        """
        Add a callback to execute before reflection execution.
        
        Args:
            callback: Callback function that takes a ScheduledReflection
        """
        self.pre_execution_callbacks.append(callback)
    
    def add_post_execution_callback(
        self,
        callback: Callable[[ScheduledReflection, Optional[str]], None]
    ) -> None:
        """
        Add a callback to execute after reflection execution.
        
        Args:
            callback: Callback function that takes a ScheduledReflection and outcome ID
        """
        self.post_execution_callbacks.append(callback)
    
    def _background_thread_func(self) -> None:
        """
        Background thread function for continuous scheduling.
        """
        logger.info("Starting background reflection scheduler thread")
        
        while not self.stop_event.is_set():
            try:
                # Check for scheduled reflections
                self.check_and_execute()
                
                # Check for new needed reflections
                needed, trigger = self.reflection_engine.check_reflection_needed()
                if needed and trigger:
                    self.schedule_reflection(trigger)
                    
                # Sleep for check interval
                self.stop_event.wait(self.check_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Error in background scheduler thread: {str(e)}", exc_info=True)
                # Sleep to avoid tight loop on errors
                self.stop_event.wait(10)
                
        logger.info("Background reflection scheduler thread stopped")
    
    def _start_background_thread(self) -> None:
        """
        Start the background scheduler thread.
        """
        if self.background_thread is not None and self.background_thread.is_alive():
            logger.warning("Background thread already running")
            return
            
        # Reset stop event
        self.stop_event.clear()
        
        # Create and start thread
        self.background_thread = threading.Thread(
            target=self._background_thread_func,
            daemon=True
        )
        self.background_thread.start()
    
    def stop_background_thread(self) -> None:
        """
        Stop the background scheduler thread.
        """
        if self.background_thread is None or not self.background_thread.is_alive():
            return
            
        # Set stop event
        self.stop_event.set()
        
        # Wait for thread to terminate
        self.background_thread.join(timeout=5.0)
        
        if self.background_thread.is_alive():
            logger.warning("Background thread did not terminate in time")
            
        self.background_thread = None 