"""
Temporal Pattern Detection Module

This module implements algorithms for detecting temporal patterns in memory access
and content, including:
- Recurring memory themes
- Temporal anomalies
- Memory access frequency patterns
- Causal relationships between memories
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime, timedelta
import logging
from collections import Counter, defaultdict

from ..memory.memory import Memory, MemoryStore
from ..math.statistics import time_series_anomaly_detection

logger = logging.getLogger(__name__)

class TemporalPattern:
    """Represents a detected temporal pattern in memory."""
    
    def __init__(self, 
                 pattern_id: str,
                 pattern_type: str,
                 memory_ids: List[str],
                 start_time: datetime,
                 end_time: datetime,
                 confidence: float,
                 metadata: Dict[str, Any] = None):
        """
        Initialize a temporal pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_type: Type of pattern (e.g., "recurring", "causal", "anomaly")
            memory_ids: List of memory IDs involved in the pattern
            start_time: When the pattern begins
            end_time: When the pattern ends
            confidence: Confidence score for the pattern (0.0-1.0)
            metadata: Additional information about the pattern
        """
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.memory_ids = memory_ids
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the pattern to a dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "memory_ids": self.memory_ids,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalPattern':
        """Create a pattern from a dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            memory_ids=data["memory_ids"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            confidence=data["confidence"],
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        return f"TemporalPattern(id={self.pattern_id}, type={self.pattern_type}, confidence={self.confidence:.2f})"


class TemporalPatternDetector:
    """Detects temporal patterns in memory access and content."""
    
    def __init__(self, 
                 memory_store: MemoryStore,
                 min_pattern_confidence: float = 0.6,
                 time_window: timedelta = timedelta(days=7),
                 min_pattern_occurrences: int = 2):
        """
        Initialize the temporal pattern detector.
        
        Args:
            memory_store: The memory store to analyze
            min_pattern_confidence: Minimum confidence threshold for patterns
            time_window: Time window for pattern detection
            min_pattern_occurrences: Minimum number of occurrences for a pattern
        """
        self.memory_store = memory_store
        self.min_pattern_confidence = min_pattern_confidence
        self.time_window = time_window
        self.min_pattern_occurrences = min_pattern_occurrences
        self.detected_patterns: List[TemporalPattern] = []
        self.pattern_counter = 0
    
    def _generate_pattern_id(self) -> str:
        """Generate a unique pattern ID."""
        self.pattern_counter += 1
        return f"pattern_{self.pattern_counter}"
    
    def detect_recurring_themes(self) -> List[TemporalPattern]:
        """
        Detect recurring themes in memory content over time.
        
        Returns:
            List of detected recurring theme patterns
        """
        memories = self.memory_store.get_all_memories()
        if not memories:
            return []
        
        # Group memories by time periods
        time_periods = defaultdict(list)
        for memory in memories:
            # Use the day as the time period
            period = memory.timestamp.date()
            time_periods[period].append(memory)
        
        # Extract themes from each period
        period_themes = {}
        for period, period_memories in time_periods.items():
            themes = self._extract_themes(period_memories)
            period_themes[period] = themes
        
        # Find recurring themes across periods
        recurring_patterns = []
        all_themes = set()
        for themes in period_themes.values():
            all_themes.update(themes.keys())
        
        for theme in all_themes:
            occurrences = []
            for period, themes in period_themes.items():
                if theme in themes and themes[theme] >= 0.3:  # Minimum theme strength
                    occurrences.append((period, themes[theme]))
            
            if len(occurrences) >= self.min_pattern_occurrences:
                # Create a pattern for this recurring theme
                memory_ids = []
                for period, _ in occurrences:
                    period_memories = time_periods[period]
                    for memory in period_memories:
                        if self._memory_contains_theme(memory, theme):
                            memory_ids.append(memory.id)
                
                if memory_ids:
                    start_time = datetime.combine(min(period for period, _ in occurrences), 
                                                 datetime.min.time())
                    end_time = datetime.combine(max(period for period, _ in occurrences), 
                                               datetime.max.time())
                    
                    # Calculate confidence based on consistency and frequency
                    strength_values = [strength for _, strength in occurrences]
                    consistency = 1.0 - np.std(strength_values) / max(np.mean(strength_values), 0.001)
                    frequency = min(1.0, len(occurrences) / len(period_themes))
                    confidence = 0.6 * frequency + 0.4 * consistency
                    
                    pattern = TemporalPattern(
                        pattern_id=self._generate_pattern_id(),
                        pattern_type="recurring_theme",
                        memory_ids=memory_ids,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=confidence,
                        metadata={"theme": theme, "occurrences": len(occurrences)}
                    )
                    recurring_patterns.append(pattern)
        
        return recurring_patterns
    
    def _extract_themes(self, memories: List[Memory]) -> Dict[str, float]:
        """
        Extract themes from a set of memories with their strength.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            Dictionary mapping themes to their strength (0.0-1.0)
        """
        # Simple implementation using word frequency
        all_words = []
        for memory in memories:
            # Extract words from the memory content
            words = memory.content.lower().split()
            # Filter out common stop words
            words = [w for w in words if len(w) > 3 and w not in {
                "this", "that", "these", "those", "there", "their", "about", 
                "which", "would", "could", "should", "with", "from", "have"
            }]
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        total_words = sum(word_counts.values())
        
        # Calculate theme strengths
        themes = {}
        if total_words > 0:
            for word, count in word_counts.items():
                if count >= 3:  # Minimum frequency threshold
                    strength = min(1.0, count / (total_words * 0.1))  # Normalize
                    themes[word] = strength
        
        return themes
    
    def _memory_contains_theme(self, memory: Memory, theme: str) -> bool:
        """Check if a memory contains a specific theme."""
        return theme.lower() in memory.content.lower()
    
    def detect_access_patterns(self) -> List[TemporalPattern]:
        """
        Detect patterns in memory access frequency.
        
        Returns:
            List of detected access patterns
        """
        memories = self.memory_store.get_all_memories()
        if not memories or not hasattr(memories[0], 'access_history'):
            logger.warning("Memory objects don't have access history for pattern detection")
            return []
        
        # Group memories by time periods
        time_series = defaultdict(int)
        for memory in memories:
            if not hasattr(memory, 'access_history'):
                continue
                
            for access_time in memory.access_history:
                day = access_time.date()
                time_series[day] += 1
        
        if not time_series:
            return []
        
        # Convert to pandas Series for time series analysis
        days = sorted(time_series.keys())
        values = [time_series[day] for day in days]
        
        # Skip if not enough data points
        if len(days) < 7:
            return []
        
        ts = pd.Series(values, index=days)
        
        # Detect anomalies
        anomalies = time_series_anomaly_detection(ts)
        
        # Create patterns for each anomaly
        access_patterns = []
        for anomaly_date, score in anomalies:
            # Find memories accessed on this date
            memory_ids = []
            for memory in memories:
                if hasattr(memory, 'access_history'):
                    if any(access.date() == anomaly_date for access in memory.access_history):
                        memory_ids.append(memory.id)
            
            if memory_ids:
                start_time = datetime.combine(anomaly_date, datetime.min.time())
                end_time = datetime.combine(anomaly_date, datetime.max.time())
                
                pattern = TemporalPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type="access_anomaly",
                    memory_ids=memory_ids,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=score,
                    metadata={"anomaly_score": score, "access_count": time_series[anomaly_date]}
                )
                access_patterns.append(pattern)
        
        return access_patterns
    
    def detect_causal_relationships(self) -> List[TemporalPattern]:
        """
        Detect potential causal relationships between memories.
        
        Returns:
            List of detected causal patterns
        """
        memories = sorted(self.memory_store.get_all_memories(), key=lambda m: m.timestamp)
        if len(memories) < 2:
            return []
        
        causal_patterns = []
        
        # Simple causal detection: look for similar content that occurs in sequence
        for i in range(len(memories) - 1):
            for j in range(i + 1, min(i + 10, len(memories))):
                mem1, mem2 = memories[i], memories[j]
                
                # Skip if too far apart in time
                time_diff = (mem2.timestamp - mem1.timestamp).total_seconds()
                if time_diff > 86400 * 3:  # More than 3 days
                    continue
                
                # Check for potential causal link based on content similarity
                causal_score = self._calculate_causal_score(mem1, mem2)
                
                if causal_score >= self.min_pattern_confidence:
                    pattern = TemporalPattern(
                        pattern_id=self._generate_pattern_id(),
                        pattern_type="causal_relationship",
                        memory_ids=[mem1.id, mem2.id],
                        start_time=mem1.timestamp,
                        end_time=mem2.timestamp,
                        confidence=causal_score,
                        metadata={
                            "cause_id": mem1.id, 
                            "effect_id": mem2.id,
                            "time_difference_seconds": time_diff
                        }
                    )
                    causal_patterns.append(pattern)
        
        return causal_patterns
    
    def _calculate_causal_score(self, mem1: Memory, mem2: Memory) -> float:
        """
        Calculate a causal relationship score between two memories.
        
        Args:
            mem1: The potential cause memory
            mem2: The potential effect memory
            
        Returns:
            Causal relationship score (0.0-1.0)
        """
        # Simple implementation based on content similarity and time proximity
        # In a full implementation, you would use more sophisticated NLP techniques
        
        # Check for content similarity
        words1 = set(mem1.content.lower().split())
        words2 = set(mem2.content.lower().split())
        
        common_words = words1.intersection(words2)
        similarity = len(common_words) / max(len(words1), len(words2), 1)
        
        # Check for time proximity (higher score for closer events)
        time_diff = (mem2.timestamp - mem1.timestamp).total_seconds()
        time_factor = max(0, 1.0 - (time_diff / (86400 * 3)))  # Scale over 3 days
        
        # Combine factors
        causal_score = 0.7 * similarity + 0.3 * time_factor
        
        return causal_score
    
    def detect_all_patterns(self) -> List[TemporalPattern]:
        """
        Run all pattern detection algorithms and return combined results.
        
        Returns:
            List of all detected patterns
        """
        patterns = []
        
        # Detect recurring themes
        try:
            theme_patterns = self.detect_recurring_themes()
            patterns.extend(theme_patterns)
            logger.info(f"Detected {len(theme_patterns)} recurring theme patterns")
        except Exception as e:
            logger.error(f"Error detecting recurring themes: {e}")
        
        # Detect access patterns
        try:
            access_patterns = self.detect_access_patterns()
            patterns.extend(access_patterns)
            logger.info(f"Detected {len(access_patterns)} access pattern anomalies")
        except Exception as e:
            logger.error(f"Error detecting access patterns: {e}")
        
        # Detect causal relationships
        try:
            causal_patterns = self.detect_causal_relationships()
            patterns.extend(causal_patterns)
            logger.info(f"Detected {len(causal_patterns)} potential causal relationships")
        except Exception as e:
            logger.error(f"Error detecting causal relationships: {e}")
        
        # Store all detected patterns
        self.detected_patterns = patterns
        
        return patterns
    
    def get_patterns_for_memory(self, memory_id: str) -> List[TemporalPattern]:
        """
        Get all patterns involving a specific memory.
        
        Args:
            memory_id: The ID of the memory
            
        Returns:
            List of patterns involving the memory
        """
        return [p for p in self.detected_patterns if memory_id in p.memory_ids]
    
    def get_patterns_by_type(self, pattern_type: str) -> List[TemporalPattern]:
        """
        Get all patterns of a specific type.
        
        Args:
            pattern_type: The type of patterns to retrieve
            
        Returns:
            List of patterns of the specified type
        """
        return [p for p in self.detected_patterns if p.pattern_type == pattern_type]
    
    def get_high_confidence_patterns(self, min_confidence: float = 0.8) -> List[TemporalPattern]:
        """
        Get patterns with confidence above the specified threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of high-confidence patterns
        """
        return [p for p in self.detected_patterns if p.confidence >= min_confidence] 