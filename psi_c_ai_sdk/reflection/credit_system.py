#!/usr/bin/env python3
"""
Reflection Credit System for ΨC-AI SDK

This module implements a credit-based system for managing reflective resources
and tracking cognitive debt. It provides mechanisms to:
1. Allocate credits for reflection operations
2. Track cognitive debt accumulation
3. Prioritize reflection activities based on importance
4. Balance immediate vs. long-term cognitive needs
"""

import logging
import time
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
import math

# Configure logging
logger = logging.getLogger(__name__)


class CreditPriority(Enum):
    """Priority levels for credit allocation."""
    
    CRITICAL = auto()  # Highest priority, address immediately
    HIGH = auto()      # Important, address soon
    MEDIUM = auto()    # Normal priority
    LOW = auto()       # Can be delayed if necessary
    MAINTENANCE = auto()  # Regular maintenance activities


@dataclass
class ReflectionCreditRecord:
    """Record of credit allocation for reflection activities."""
    
    id: str                          # Unique identifier
    timestamp: float                 # Time of allocation
    credits_allocated: float         # Amount of credits allocated
    credits_used: float = 0.0        # Amount of credits used
    priority: CreditPriority         # Priority level
    reason: str                      # Why credits were allocated
    status: str = "pending"          # Status: pending, active, completed, expired
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    due_by: Optional[float] = None   # Optional deadline
    
    @property
    def is_complete(self) -> bool:
        """Check if the activity is complete."""
        return self.status == "completed"
    
    @property
    def is_active(self) -> bool:
        """Check if the activity is currently active."""
        return self.status == "active"
    
    @property
    def is_overdue(self) -> bool:
        """Check if the activity is overdue."""
        if self.due_by is None:
            return False
        return time.time() > self.due_by and self.status != "completed"
    
    @property
    def unused_credits(self) -> float:
        """Get unused credits."""
        return max(0.0, self.credits_allocated - self.credits_used)
    
    def mark_active(self) -> None:
        """Mark the activity as active."""
        self.status = "active"
        self.metadata["activated_at"] = time.time()
    
    def mark_complete(self, credits_used: Optional[float] = None) -> float:
        """
        Mark the activity as complete.
        
        Args:
            credits_used: Override for credits used (default: all allocated)
            
        Returns:
            Unused credits
        """
        if credits_used is not None:
            self.credits_used = min(self.credits_allocated, credits_used)
        else:
            self.credits_used = self.credits_allocated
            
        self.status = "completed"
        self.metadata["completed_at"] = time.time()
        
        return self.unused_credits
    
    def mark_expired(self) -> float:
        """
        Mark the activity as expired.
        
        Returns:
            Unused credits
        """
        unused = self.unused_credits
        self.status = "expired"
        self.metadata["expired_at"] = time.time()
        
        return unused


class ReflectionCreditSystem:
    """
    System for managing reflection credits and cognitive debt.
    
    The credit system manages the allocation and tracking of cognitive
    resources for reflection activities. It helps ensure the system
    doesn't over-allocate resources and maintains a balance between
    immediate needs and long-term cognitive health.
    """
    
    def __init__(
        self,
        initial_credits: float = 100.0,
        max_credits: float = 200.0,
        credit_regeneration_rate: float = 1.0,  # Credits per minute
        debt_threshold: float = 50.0,
        critical_debt_threshold: float = 100.0,
        debt_interest_rate: float = 0.05,  # Per hour
        max_record_history: int = 1000
    ):
        """
        Initialize the reflection credit system.
        
        Args:
            initial_credits: Initial credit balance
            max_credits: Maximum credit balance
            credit_regeneration_rate: Rate at which credits regenerate (per minute)
            debt_threshold: Threshold for concerning cognitive debt
            critical_debt_threshold: Threshold for critical cognitive debt
            debt_interest_rate: Rate at which debt increases over time (per hour)
            max_record_history: Maximum number of records to maintain
        """
        self.available_credits = initial_credits
        self.max_credits = max_credits
        self.credit_regeneration_rate = credit_regeneration_rate
        self.debt_threshold = debt_threshold
        self.critical_debt_threshold = critical_debt_threshold
        self.debt_interest_rate = debt_interest_rate
        self.max_record_history = max_record_history
        
        # Tracking
        self.cognitive_debt = 0.0
        self.last_update_time = time.time()
        self.credit_records: Dict[str, ReflectionCreditRecord] = {}
        self.completed_records: List[ReflectionCreditRecord] = []
        
        # Usage metrics
        self.total_credits_allocated = 0.0
        self.total_credits_used = 0.0
        self.total_credits_expired = 0.0
        self.debt_paid = 0.0
        
        # Priority quotas (percentage of credits reserved for each priority)
        self.priority_quotas = {
            CreditPriority.CRITICAL: 0.4,    # 40% for critical
            CreditPriority.HIGH: 0.3,        # 30% for high
            CreditPriority.MEDIUM: 0.2,      # 20% for medium
            CreditPriority.LOW: 0.05,        # 5% for low
            CreditPriority.MAINTENANCE: 0.05  # 5% for maintenance
        }
    
    def update(self) -> Dict[str, Any]:
        """
        Update credit balance and debt based on time passed.
        
        Returns:
            Updated system status
        """
        current_time = time.time()
        time_diff_minutes = (current_time - self.last_update_time) / 60.0
        
        # Regenerate credits
        if self.available_credits < self.max_credits:
            new_credits = self.credit_regeneration_rate * time_diff_minutes
            self.available_credits = min(self.max_credits, self.available_credits + new_credits)
        
        # Apply debt interest (converted from per-hour to per-minute)
        if self.cognitive_debt > 0:
            hourly_interest = self.debt_interest_rate * (time_diff_minutes / 60.0)
            self.cognitive_debt *= (1 + hourly_interest)
        
        # Process expired records
        for record_id, record in list(self.credit_records.items()):
            if record.due_by and current_time > record.due_by and not record.is_complete:
                unused_credits = record.mark_expired()
                self.available_credits += unused_credits
                self.total_credits_expired += unused_credits
                
                # Move to completed records
                self.completed_records.append(record)
                del self.credit_records[record_id]
        
        # Prune completed records if needed
        if len(self.completed_records) > self.max_record_history:
            self.completed_records = self.completed_records[-self.max_record_history:]
        
        # Update time
        self.last_update_time = current_time
        
        return self.get_status()
    
    def allocate_credits(
        self,
        amount: float,
        priority: CreditPriority,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        due_by: Optional[float] = None
    ) -> Optional[str]:
        """
        Allocate credits for a reflection activity.
        
        Args:
            amount: Amount of credits to allocate
            priority: Priority level
            reason: Reason for allocation
            metadata: Additional metadata
            due_by: Optional deadline
            
        Returns:
            Record ID if successful, None if insufficient credits
        """
        # First update the system
        self.update()
        
        # Check if we have enough credits
        if amount > self.available_credits:
            logger.warning(
                f"Insufficient credits for allocation: requested={amount}, "
                f"available={self.available_credits}"
            )
            
            # Check if we should accumulate debt
            if priority == CreditPriority.CRITICAL:
                # Allow debt for critical activities
                self.cognitive_debt += (amount - self.available_credits)
                logger.warning(
                    f"Accumulating cognitive debt for critical activity: "
                    f"new_debt={amount - self.available_credits}, "
                    f"total_debt={self.cognitive_debt}"
                )
            else:
                # For non-critical, try to allocate a reduced amount
                available_amount = self.get_credit_allocation(priority, amount)
                if available_amount <= 0:
                    return None
                amount = available_amount
        
        # Create credit record
        record_id = str(uuid.uuid4())
        record = ReflectionCreditRecord(
            id=record_id,
            timestamp=time.time(),
            credits_allocated=amount,
            priority=priority,
            reason=reason,
            metadata=metadata or {},
            due_by=due_by
        )
        
        # Update system state
        self.credit_records[record_id] = record
        self.available_credits -= amount
        self.total_credits_allocated += amount
        
        logger.info(
            f"Credits allocated: amount={amount}, priority={priority.name}, "
            f"reason='{reason}', record_id={record_id}"
        )
        
        return record_id
    
    def get_credit_record(self, record_id: str) -> Optional[ReflectionCreditRecord]:
        """
        Get credit record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Credit record if found, None otherwise
        """
        if record_id in self.credit_records:
            return self.credit_records[record_id]
        
        # Try in completed records
        for record in self.completed_records:
            if record.id == record_id:
                return record
                
        return None
    
    def start_activity(self, record_id: str) -> bool:
        """
        Mark a credit record as active.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if successful, False otherwise
        """
        record = self.get_credit_record(record_id)
        if not record or record.status != "pending":
            if not record:
                logger.warning(f"Cannot start activity: record not found: {record_id}")
            else:
                logger.warning(
                    f"Cannot start activity: invalid state: {record.status}, "
                    f"record_id={record_id}"
                )
            return False
        
        record.mark_active()
        logger.info(f"Activity started: record_id={record_id}")
        return True
    
    def complete_activity(
        self,
        record_id: str,
        credits_used: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Mark a credit record as complete.
        
        Args:
            record_id: Record ID
            credits_used: Override for credits used (default: all allocated)
            
        Returns:
            Tuple of (success, unused_credits)
        """
        record = self.get_credit_record(record_id)
        if not record or record.status not in ["pending", "active"]:
            if not record:
                logger.warning(f"Cannot complete activity: record not found: {record_id}")
            else:
                logger.warning(
                    f"Cannot complete activity: invalid state: {record.status}, "
                    f"record_id={record_id}"
                )
            return False, 0.0
        
        # Mark complete and get unused credits
        unused_credits = record.mark_complete(credits_used)
        
        # Update system state
        self.available_credits += unused_credits
        self.total_credits_used += record.credits_used
        
        # Move to completed records
        if record_id in self.credit_records:
            self.completed_records.append(record)
            del self.credit_records[record_id]
        
        logger.info(
            f"Activity completed: record_id={record_id}, "
            f"credits_used={record.credits_used}, unused={unused_credits}"
        )
        
        return True, unused_credits
    
    def pay_cognitive_debt(self, amount: float) -> float:
        """
        Pay down cognitive debt.
        
        Args:
            amount: Amount to pay
            
        Returns:
            Amount actually paid
        """
        if self.cognitive_debt <= 0:
            return 0.0
        
        # Limit payment to available credits
        payment = min(amount, self.available_credits)
        
        # Limit payment to current debt
        payment = min(payment, self.cognitive_debt)
        
        if payment <= 0:
            return 0.0
        
        # Update system state
        self.available_credits -= payment
        self.cognitive_debt -= payment
        self.debt_paid += payment
        
        logger.info(
            f"Cognitive debt payment: amount={payment}, "
            f"remaining_debt={self.cognitive_debt}"
        )
        
        return payment
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status
        """
        return {
            "available_credits": self.available_credits,
            "max_credits": self.max_credits,
            "cognitive_debt": self.cognitive_debt,
            "debt_threshold": self.debt_threshold,
            "critical_debt_threshold": self.critical_debt_threshold,
            "debt_status": self.get_debt_status(),
            "active_activities": len(self.credit_records),
            "completed_activities": len(self.completed_records)
        }
    
    def get_debt_status(self) -> str:
        """
        Get debt status classification.
        
        Returns:
            Debt status: "healthy", "concerning", or "critical"
        """
        if self.cognitive_debt >= self.critical_debt_threshold:
            return "critical"
        elif self.cognitive_debt >= self.debt_threshold:
            return "concerning"
        else:
            return "healthy"
    
    def get_credit_allocation(
        self,
        priority: CreditPriority,
        requested_amount: float
    ) -> float:
        """
        Calculate how many credits can be allocated for a priority level.
        
        Args:
            priority: Priority level
            requested_amount: Requested amount
            
        Returns:
            Amount that can be allocated
        """
        # For critical, allow debt accumulation
        if priority == CreditPriority.CRITICAL:
            return requested_amount
        
        # Calculate quota for this priority
        priority_quota = self.priority_quotas.get(priority, 0.05)
        max_allowed = self.max_credits * priority_quota
        
        # Calculate how much is already allocated for this priority
        allocated_for_priority = sum(
            record.credits_allocated
            for record in self.credit_records.values()
            if record.priority == priority
        )
        
        # Calculate available amount for this priority
        available_for_priority = max(0, max_allowed - allocated_for_priority)
        
        # Return minimum of requested and available
        return min(requested_amount, available_for_priority, self.available_credits)
    
    def get_active_records_by_priority(self, priority: CreditPriority) -> List[ReflectionCreditRecord]:
        """
        Get active records for a priority level.
        
        Args:
            priority: Priority level
            
        Returns:
            List of active records for the priority level
        """
        return [
            record for record in self.credit_records.values()
            if record.priority == priority
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.
        
        Returns:
            System metrics
        """
        return {
            "total_credits_allocated": self.total_credits_allocated,
            "total_credits_used": self.total_credits_used,
            "total_credits_expired": self.total_credits_expired,
            "cognitive_debt": self.cognitive_debt,
            "debt_paid": self.debt_paid,
            "credit_efficiency": (
                self.total_credits_used / self.total_credits_allocated
                if self.total_credits_allocated > 0 else 1.0
            ),
            "active_records_count": len(self.credit_records),
            "completed_records_count": len(self.completed_records),
            "priority_distribution": {
                priority.name: len(self.get_active_records_by_priority(priority))
                for priority in CreditPriority
            }
        }

    def calculate_debt_interest(self, time_period_hours: float) -> float:
        """
        Calculate accrued interest on cognitive debt over time.
        
        Args:
            time_period_hours: Time period in hours
            
        Returns:
            Accrued interest amount
        """
        if self.cognitive_debt <= 0:
            return 0.0
            
        # Calculate compound interest
        interest = self.cognitive_debt * ((1 + self.debt_interest_rate) ** time_period_hours - 1)
        return interest

    def estimate_debt_payoff_time(self, payment_rate: float) -> Optional[float]:
        """
        Estimate time needed to pay off cognitive debt.
        
        Args:
            payment_rate: Payment rate per hour
            
        Returns:
            Estimated hours to pay off debt, or None if impossible
        """
        if self.cognitive_debt <= 0:
            return 0.0
            
        if payment_rate <= 0:
            return None
            
        # If interest rate exceeds payment rate, debt will never be paid off
        effective_hourly_interest = self.cognitive_debt * self.debt_interest_rate
        if effective_hourly_interest >= payment_rate:
            return None
            
        # Calculate time to pay off debt with continuous interest
        # Formula: t = ln(P / (P - r*D)) / r
        # Where P = payment rate, r = interest rate, D = initial debt
        try:
            hours = math.log(payment_rate / (payment_rate - self.debt_interest_rate * self.cognitive_debt)) / self.debt_interest_rate
            return max(0.0, hours)
        except (ValueError, ZeroDivisionError):
            return None

    def forecast_debt(self, hours: float, payment_rate: float = 0.0) -> float:
        """
        Forecast cognitive debt level after a time period.
        
        Args:
            hours: Time period in hours
            payment_rate: Optional payment rate per hour
            
        Returns:
            Forecasted debt level
        """
        if self.cognitive_debt <= 0:
            return 0.0
            
        # Simple compound interest model with continuous payments
        if payment_rate <= 0:
            # No payments, just compound interest
            return self.cognitive_debt * ((1 + self.debt_interest_rate) ** hours)
            
        # With regular payments
        r = self.debt_interest_rate
        P = payment_rate
        D = self.cognitive_debt
        
        # If payment rate equals interest accrual rate
        if P == r * D:
            return D + (P - r * D) * hours
            
        # Formula with continuous payments and interest
        return D * (1 + r) ** hours - (P / r) * ((1 + r) ** hours - 1)
        
    def is_debt_sustainable(self) -> bool:
        """
        Check if current debt level is sustainable.
        
        Returns:
            True if debt is sustainable, False otherwise
        """
        # Debt is sustainable if it's below critical threshold
        # and credits regenerate faster than debt interest
        if self.cognitive_debt <= 0:
            return True
            
        hourly_regeneration = self.credit_regeneration_rate * 60
        hourly_interest = self.cognitive_debt * self.debt_interest_rate
        
        return (self.cognitive_debt < self.critical_debt_threshold and
                hourly_regeneration > hourly_interest)
                
    def recommend_debt_action(self) -> Tuple[str, Dict[str, Any]]:
        """
        Recommend action for managing cognitive debt.
        
        Returns:
            Tuple of (action, details)
        """
        if self.cognitive_debt <= 0:
            return "none", {"reason": "No cognitive debt"}
            
        debt_status = self.get_debt_status()
        is_sustainable = self.is_debt_sustainable()
        
        if debt_status == "critical":
            if not is_sustainable:
                return "emergency", {
                    "reason": "Critical unsustainable debt",
                    "recommended_payment": self.cognitive_debt * 0.5,
                    "urgency": "immediate"
                }
            else:
                return "significant", {
                    "reason": "Critical but sustainable debt",
                    "recommended_payment": self.cognitive_debt * 0.3,
                    "urgency": "high"
                }
        elif debt_status == "concerning":
            if not is_sustainable:
                return "major", {
                    "reason": "Concerning unsustainable debt",
                    "recommended_payment": self.cognitive_debt * 0.25,
                    "urgency": "high"
                }
            else:
                return "moderate", {
                    "reason": "Concerning but sustainable debt",
                    "recommended_payment": self.cognitive_debt * 0.15,
                    "urgency": "medium"
                }
        else:
            return "minor", {
                "reason": "Healthy debt levels",
                "recommended_payment": self.cognitive_debt * 0.1,
                "urgency": "low"
            }

    def calculate_optimal_payment_schedule(
        self, 
        target_payoff_time: float
    ) -> List[Dict[str, Any]]:
        """
        Calculate optimal payment schedule to pay off debt.
        
        Args:
            target_payoff_time: Target payoff time in hours
            
        Returns:
            List of payment intervals with amounts
        """
        if self.cognitive_debt <= 0 or target_payoff_time <= 0:
            return []
            
        # Estimate total amount needed with interest
        total_with_interest = self.forecast_debt(target_payoff_time)
        
        # Divide into payment intervals
        num_intervals = max(1, round(target_payoff_time / 24))  # Default daily payments
        
        schedule = []
        remaining_debt = self.cognitive_debt
        
        for i in range(num_intervals):
            # Calculate interest for this period
            period_hours = target_payoff_time / num_intervals
            forecasted_debt = self.forecast_debt(period_hours, 0)
            interest = forecasted_debt - remaining_debt
            
            # Calculate payment
            if i == num_intervals - 1:
                # Last payment covers whatever is left
                payment = remaining_debt + interest
            else:
                # Regular payments
                payment = (total_with_interest / num_intervals) + interest
                
            # Update remaining debt
            remaining_debt = remaining_debt + interest - payment
            
            schedule.append({
                "interval": i + 1,
                "hours_from_now": period_hours * (i + 1),
                "payment_amount": payment,
                "interest_amount": interest,
                "remaining_debt": max(0, remaining_debt)
            })
            
            if remaining_debt <= 0:
                break
                
        return schedule

def calculate_cognitive_debt(
    reflection_frequency: float,
    complexity: float,
    contradiction_rate: float,
    coherence_score: float,
    epistemic_workload: float
) -> float:
    """
    Calculate cognitive debt based on system parameters.
    
    This function provides a standardized way to calculate cognitive debt
    for AI systems based on multiple factors that contribute to cognitive load.
    
    Args:
        reflection_frequency: Frequency of reflective cycles (higher is better)
        complexity: Complexity of the knowledge base (higher means more debt)
        contradiction_rate: Rate of contradictions in memory (higher means more debt)
        coherence_score: Overall coherence of knowledge (higher is better)
        epistemic_workload: Current epistemic workload (higher means more debt)
        
    Returns:
        Calculated cognitive debt score
    """
    # Normalize inputs
    reflection_factor = 1 / max(reflection_frequency, 0.1)  # Inverse relationship
    complexity_factor = min(max(complexity, 0), 1)
    contradiction_factor = min(max(contradiction_rate, 0), 1) 
    coherence_factor = 1 - min(max(coherence_score, 0), 1)  # Inverse relationship
    workload_factor = min(max(epistemic_workload, 0), 1)
    
    # Weighted sum of factors
    weights = {
        "reflection": 0.2,
        "complexity": 0.25,
        "contradiction": 0.3,
        "coherence": 0.15,
        "workload": 0.1
    }
    
    debt_score = (
        weights["reflection"] * reflection_factor +
        weights["complexity"] * complexity_factor +
        weights["contradiction"] * contradiction_factor +
        weights["coherence"] * coherence_factor +
        weights["workload"] * workload_factor
    )
    
    # Scale to a meaningful range (0-100)
    return debt_score * 100 