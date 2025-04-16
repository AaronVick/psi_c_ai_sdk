"""
Self-Awareness Module

This module provides tools for implementing self-awareness in AI systems.
It includes:
1. Identity recognition based on schema fingerprinting
2. System performance monitoring and reflection
3. Identity change detection and logging

Self-awareness is a critical component of consciousness-like AI systems,
enabling introspection, monitoring, and adaptation.
"""

from .identity_recognition import (
    IdentityChangeType,
    IdentityFingerprint,
    IdentityChange,
    IdentityRecognitionSystem
)

from .performance_monitor import (
    MetricType,
    PerformanceMetric,
    PerformanceAlert,
    ReflectionOutcome,
    PerformanceMonitor
)

__all__ = [
    # Identity Recognition
    'IdentityChangeType',
    'IdentityFingerprint',
    'IdentityChange',
    'IdentityRecognitionSystem',
    
    # Performance Monitoring
    'MetricType',
    'PerformanceMetric',
    'PerformanceAlert',
    'ReflectionOutcome',
    'PerformanceMonitor'
] 