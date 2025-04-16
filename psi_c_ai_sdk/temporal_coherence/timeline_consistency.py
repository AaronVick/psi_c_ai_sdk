"""
Timeline Consistency Module

This module implements algorithms for detecting and resolving temporal inconsistencies
in memory timelines, including:
- Timeline validation
- Contradiction detection in event sequences
- Temporal relationship verification
- Timeline repair recommendations
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict

from ..memory.memory import Memory, MemoryStore
from ..safety.contradiction_detector import ContradictionDetector

logger = logging.getLogger(__name__)


class TimelineInconsistency:
    """Represents a detected inconsistency in the memory timeline."""
    
    INCONSISTENCY_TYPES = [
        "temporal_contradiction",
        "impossible_sequence",
        "timeline_gap",
        "anachronism",
        "circular_causality"
    ]
    
    def __init__(self, 
                 inconsistency_id: str,
                 inconsistency_type: str,
                 memory_ids: List[str],
                 description: str,
                 severity: float,
                 repair_suggestions: List[str] = None):
        """
        Initialize a timeline inconsistency.
        
        Args:
            inconsistency_id: Unique identifier for the inconsistency
            inconsistency_type: Type of inconsistency (from INCONSISTENCY_TYPES)
            memory_ids: List of memory IDs involved in the inconsistency
            description: Human-readable description of the inconsistency
            severity: How severe the inconsistency is (0.0-1.0)
            repair_suggestions: Suggestions for resolving the inconsistency
        """
        if inconsistency_type not in self.INCONSISTENCY_TYPES:
            raise ValueError(f"Invalid inconsistency type: {inconsistency_type}")
            
        self.inconsistency_id = inconsistency_id
        self.inconsistency_type = inconsistency_type
        self.memory_ids = memory_ids
        self.description = description
        self.severity = severity
        self.repair_suggestions = repair_suggestions or []
        self.repaired = False
        self.repair_notes = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the inconsistency to a dictionary."""
        return {
            "inconsistency_id": self.inconsistency_id,
            "inconsistency_type": self.inconsistency_type,
            "memory_ids": self.memory_ids,
            "description": self.description,
            "severity": self.severity,
            "repair_suggestions": self.repair_suggestions,
            "repaired": self.repaired,
            "repair_notes": self.repair_notes
        }
    
    def mark_as_repaired(self, notes: str = ""):
        """Mark this inconsistency as repaired."""
        self.repaired = True
        self.repair_notes = notes
    
    def __repr__(self) -> str:
        return f"TimelineInconsistency(id={self.inconsistency_id}, type={self.inconsistency_type}, severity={self.severity:.2f})"


class TimelineConsistencyChecker:
    """Checks and maintains consistency in memory timelines."""
    
    def __init__(self, 
                 memory_store: MemoryStore,
                 contradiction_detector: Optional[ContradictionDetector] = None,
                 timeline_gap_threshold: timedelta = timedelta(days=30),
                 severity_threshold: float = 0.5):
        """
        Initialize the timeline consistency checker.
        
        Args:
            memory_store: The memory store to analyze
            contradiction_detector: Detector for content contradictions
            timeline_gap_threshold: Threshold for considering a gap significant
            severity_threshold: Minimum severity threshold for reporting inconsistencies
        """
        self.memory_store = memory_store
        self.contradiction_detector = contradiction_detector
        self.timeline_gap_threshold = timeline_gap_threshold
        self.severity_threshold = severity_threshold
        self.inconsistencies: List[TimelineInconsistency] = []
        self.inconsistency_counter = 0
        
        # Temporal keywords that indicate relationships
        self.temporal_keywords = {
            'before': 'before',
            'after': 'after',
            'during': 'during',
            'while': 'during',
            'previously': 'before',
            'following': 'after',
            'prior to': 'before',
            'subsequent to': 'after',
            'earlier': 'before',
            'later': 'after',
            'simultaneously': 'during',
            'at the same time': 'during',
            'formerly': 'before',
            'preceded': 'before',
            'succeeded': 'after',
            'now': 'present',
            'currently': 'present',
            'in the past': 'before',
            'in the future': 'after',
            'yesterday': 'before',
            'tomorrow': 'after',
            'last week': 'before',
            'next week': 'after',
            'last month': 'before',
            'next month': 'after',
            'last year': 'before',
            'next year': 'after'
        }
    
    def _generate_inconsistency_id(self) -> str:
        """Generate a unique inconsistency ID."""
        self.inconsistency_counter += 1
        return f"inconsistency_{self.inconsistency_counter}"
    
    def check_timeline_consistency(self) -> List[TimelineInconsistency]:
        """
        Run all consistency checks on the memory timeline.
        
        Returns:
            List of detected inconsistencies
        """
        inconsistencies = []
        
        # Check for temporal contradictions
        try:
            contradictions = self._check_temporal_contradictions()
            inconsistencies.extend(contradictions)
            logger.info(f"Detected {len(contradictions)} temporal contradictions")
        except Exception as e:
            logger.error(f"Error checking temporal contradictions: {e}")
        
        # Check for impossible sequences
        try:
            impossible_sequences = self._check_impossible_sequences()
            inconsistencies.extend(impossible_sequences)
            logger.info(f"Detected {len(impossible_sequences)} impossible sequences")
        except Exception as e:
            logger.error(f"Error checking impossible sequences: {e}")
        
        # Check for timeline gaps
        try:
            gaps = self._check_timeline_gaps()
            inconsistencies.extend(gaps)
            logger.info(f"Detected {len(gaps)} significant timeline gaps")
        except Exception as e:
            logger.error(f"Error checking timeline gaps: {e}")
        
        # Check for anachronisms
        try:
            anachronisms = self._check_anachronisms()
            inconsistencies.extend(anachronisms)
            logger.info(f"Detected {len(anachronisms)} potential anachronisms")
        except Exception as e:
            logger.error(f"Error checking anachronisms: {e}")
        
        # Check for circular causality
        try:
            circular_causality = self._check_circular_causality()
            inconsistencies.extend(circular_causality)
            logger.info(f"Detected {len(circular_causality)} circular causality instances")
        except Exception as e:
            logger.error(f"Error checking circular causality: {e}")
        
        # Filter by severity threshold
        self.inconsistencies = [inc for inc in inconsistencies if inc.severity >= self.severity_threshold]
        
        return self.inconsistencies
    
    def _check_temporal_contradictions(self) -> List[TimelineInconsistency]:
        """
        Check for memories that contradict each other in terms of when events occurred.
        
        Returns:
            List of temporal contradiction inconsistencies
        """
        if not self.contradiction_detector:
            logger.warning("No contradiction detector provided, skipping temporal contradiction check")
            return []
        
        memories = self.memory_store.get_all_memories()
        inconsistencies = []
        
        # First, identify potential contradictions using the detector
        contradictions = self.contradiction_detector.find_contradictions(memories)
        
        # Filter for temporal contradictions
        for contradiction in contradictions:
            mem1_id, mem2_id = contradiction.memory_ids
            mem1 = self.memory_store.get_memory(mem1_id)
            mem2 = self.memory_store.get_memory(mem2_id)
            
            if not (mem1 and mem2):
                continue
                
            # Check if the contradiction involves temporal aspects
            is_temporal = self._is_temporal_contradiction(mem1, mem2)
            
            if is_temporal:
                inconsistency = TimelineInconsistency(
                    inconsistency_id=self._generate_inconsistency_id(),
                    inconsistency_type="temporal_contradiction",
                    memory_ids=[mem1_id, mem2_id],
                    description=f"Temporal contradiction between memories: '{mem1.content[:50]}...' and '{mem2.content[:50]}...'",
                    severity=min(1.0, contradiction.confidence + 0.2),  # Increase severity for temporal contradictions
                    repair_suggestions=[
                        "Determine which memory is more reliable based on recency and source",
                        "Adjust timestamp of one memory if the content is correct but timing is wrong",
                        "Flag one memory as potentially incorrect"
                    ]
                )
                inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _is_temporal_contradiction(self, mem1: Memory, mem2: Memory) -> bool:
        """
        Check if two memories contradict each other temporally.
        
        Args:
            mem1: First memory
            mem2: Second memory
            
        Returns:
            True if there is a temporal contradiction
        """
        # First check timestamps
        time_diff = abs((mem1.timestamp - mem2.timestamp).total_seconds())
        
        # Simple heuristic: check for temporal keywords in both memories
        content1 = mem1.content.lower()
        content2 = mem2.content.lower()
        
        temporal_terms1 = [term for term in self.temporal_keywords if term in content1]
        temporal_terms2 = [term for term in self.temporal_keywords if term in content2]
        
        # If both have temporal terms, check for potential conflicts
        if temporal_terms1 and temporal_terms2:
            # Simple check: look for opposing temporal relations
            mem1_relations = [self.temporal_keywords[term] for term in temporal_terms1]
            mem2_relations = [self.temporal_keywords[term] for term in temporal_terms2]
            
            if ('before' in mem1_relations and 'after' in mem2_relations) or \
               ('after' in mem1_relations and 'before' in mem2_relations):
                return True
        
        # If timestamps differ significantly but the memories seem to refer to the same event
        # This is a simplified check - a real implementation would use NLP to identify events
        if time_diff > 86400:  # More than a day difference
            # Count word overlap as a simple similarity measure
            words1 = set(content1.split())
            words2 = set(content2.split())
            overlap = len(words1.intersection(words2))
            
            # If high similarity but timestamps differ significantly, may be a contradiction
            if overlap > 10 and overlap / max(len(words1), len(words2)) > 0.5:
                return True
        
        return False
    
    def _check_impossible_sequences(self) -> List[TimelineInconsistency]:
        """
        Check for sequences of events that are impossible given time constraints.
        
        Returns:
            List of impossible sequence inconsistencies
        """
        memories = sorted(self.memory_store.get_all_memories(), key=lambda m: m.timestamp)
        inconsistencies = []
        
        if len(memories) < 2:
            return []
        
        # Look for references to the same location with insufficient travel time
        for i in range(len(memories) - 1):
            for j in range(i + 1, min(i + 10, len(memories))):
                mem1, mem2 = memories[i], memories[j]
                
                # Skip if too far apart in time (focus on close events)
                time_diff = (mem2.timestamp - mem1.timestamp).total_seconds()
                if time_diff > 86400:  # More than a day
                    continue
                
                # Check for location references
                # This is a simplified check - a real implementation would use NLP to identify locations
                locations1 = self._extract_locations(mem1.content)
                locations2 = self._extract_locations(mem2.content)
                
                # If different locations are mentioned with insufficient time between events
                if locations1 and locations2 and not any(loc in locations2 for loc in locations1):
                    # Simplified check assuming reasonable travel time between locations
                    min_travel_time = 3600  # 1 hour as minimum travel time
                    
                    if time_diff < min_travel_time:
                        inconsistency = TimelineInconsistency(
                            inconsistency_id=self._generate_inconsistency_id(),
                            inconsistency_type="impossible_sequence",
                            memory_ids=[mem1.id, mem2.id],
                            description=f"Impossible travel sequence: {locations1} to {locations2} in {time_diff/60:.1f} minutes",
                            severity=0.7,
                            repair_suggestions=[
                                f"Adjust timestamp of second memory to allow for travel time",
                                "Check if locations are correctly identified",
                                "Verify if these are actually sequential events or parallel"
                            ]
                        )
                        inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _extract_locations(self, content: str) -> List[str]:
        """
        Extract location mentions from memory content.
        
        Args:
            content: The memory content text
            
        Returns:
            List of potential location strings
        """
        # This is a simplified implementation
        # A real implementation would use NLP named entity recognition
        
        # Simple location indicators
        location_indicators = ['in', 'at', 'near', 'to', 'from']
        locations = []
        
        words = content.split()
        for i, word in enumerate(words):
            if word.lower() in location_indicators and i < len(words) - 1:
                # Check if next words start with uppercase (potential place name)
                if i+1 < len(words) and words[i+1][0].isupper():
                    potential_location = words[i+1]
                    # Try to capture multi-word locations
                    j = i + 2
                    while j < len(words) and words[j][0].isupper():
                        potential_location += " " + words[j]
                        j += 1
                    
                    locations.append(potential_location)
        
        return locations
    
    def _check_timeline_gaps(self) -> List[TimelineInconsistency]:
        """
        Check for significant gaps in the timeline that might indicate missing memories.
        
        Returns:
            List of timeline gap inconsistencies
        """
        memories = sorted(self.memory_store.get_all_memories(), key=lambda m: m.timestamp)
        inconsistencies = []
        
        if len(memories) < 2:
            return []
        
        for i in range(len(memories) - 1):
            mem1, mem2 = memories[i], memories[i+1]
            gap = mem2.timestamp - mem1.timestamp
            
            if gap > self.timeline_gap_threshold:
                # Calculate severity based on gap size
                severity_factor = min(1.0, gap / (self.timeline_gap_threshold * 3))
                
                inconsistency = TimelineInconsistency(
                    inconsistency_id=self._generate_inconsistency_id(),
                    inconsistency_type="timeline_gap",
                    memory_ids=[mem1.id, mem2.id],
                    description=f"Significant timeline gap of {gap.days} days between memories",
                    severity=severity_factor,
                    repair_suggestions=[
                        "Check if memories from this period were incorrectly dated",
                        "Verify if memories from this period are missing",
                        "Consider if this gap represents a period of inactivity"
                    ]
                )
                inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _check_anachronisms(self) -> List[TimelineInconsistency]:
        """
        Check for anachronistic references that don't make sense in timeline context.
        
        Returns:
            List of anachronism inconsistencies
        """
        memories = self.memory_store.get_all_memories()
        inconsistencies = []
        
        # Simple anachronism detection using reference events/times
        reference_events = {
            "COVID-19 pandemic": datetime(2020, 1, 1),
            "iPhone release": datetime(2007, 6, 29),
            "9/11 attacks": datetime(2001, 9, 11),
            "Facebook launch": datetime(2004, 2, 4),
            "Twitter (X) launch": datetime(2006, 3, 21),
            "Instagram launch": datetime(2010, 10, 6),
            "TikTok launch": datetime(2016, 9, 1),
            "ChatGPT release": datetime(2022, 11, 30),
            "COVID-19 vaccine": datetime(2020, 12, 1),
            "Bitcoin creation": datetime(2009, 1, 3),
        }
        
        for memory in memories:
            content = memory.content.lower()
            memory_time = memory.timestamp
            
            for event, event_time in reference_events.items():
                event_terms = event.lower().split()
                
                # Simple check if all terms in the event appear in the memory content
                if all(term in content for term in event_terms):
                    # Check if memory is dated before the event
                    if memory_time < event_time:
                        time_diff = (event_time - memory_time).days
                        
                        # Calculate severity based on how far before the event
                        severity = min(1.0, time_diff / 365)  # Max severity if a year or more
                        
                        inconsistency = TimelineInconsistency(
                            inconsistency_id=self._generate_inconsistency_id(),
                            inconsistency_type="anachronism",
                            memory_ids=[memory.id],
                            description=f"Anachronistic reference to '{event}' {time_diff} days before it occurred",
                            severity=severity,
                            repair_suggestions=[
                                "Verify memory timestamp",
                                "Check if the reference is actually to something else",
                                f"Consider updating memory timestamp to after {event_time.strftime('%Y-%m-%d')}"
                            ]
                        )
                        inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _check_circular_causality(self) -> List[TimelineInconsistency]:
        """
        Check for circular causality references in memories.
        
        Returns:
            List of circular causality inconsistencies
        """
        memories = self.memory_store.get_all_memories()
        inconsistencies = []
        
        # Build a simple causal graph
        causal_graph = defaultdict(list)
        
        # This is a simplified implementation
        # A real version would use NLP to extract causal relationships
        causal_phrases = [
            "because of", "due to", "as a result of", "caused by",
            "led to", "resulted in", "caused", "made", "influenced"
        ]
        
        # Extract causal relationships
        for memory in memories:
            content = memory.content.lower()
            
            for phrase in causal_phrases:
                if phrase in content:
                    # Simplified: just record that this memory has causal language
                    parts = content.split(phrase)
                    if len(parts) >= 2:
                        cause_part = parts[0].strip().split()[-3:]  # Last few words before phrase
                        effect_part = parts[1].strip().split()[:3]  # First few words after phrase
                        
                        cause_key = " ".join(cause_part)
                        effect_key = " ".join(effect_part)
                        
                        causal_graph[cause_key].append((effect_key, memory.id))
        
        # Check for cycles in the causal graph
        # This is a simplified cycle detection
        visited = set()
        
        def detect_cycle(node, path):
            if node in path:
                return path[path.index(node):]
            
            if node in visited:
                return None
                
            visited.add(node)
            path.append(node)
            
            for neighbor, _ in causal_graph.get(node, []):
                cycle = detect_cycle(neighbor, path.copy())
                if cycle:
                    return cycle
            
            return None
        
        for node in causal_graph:
            cycle = detect_cycle(node, [])
            if cycle and len(cycle) > 1:
                # Find memories involved in this cycle
                cycle_memories = set()
                for i in range(len(cycle) - 1):
                    cause, effect = cycle[i], cycle[i+1]
                    for e, memory_id in causal_graph[cause]:
                        if e == effect:
                            cycle_memories.add(memory_id)
                
                if cycle_memories:
                    cycle_str = " → ".join(cycle)
                    inconsistency = TimelineInconsistency(
                        inconsistency_id=self._generate_inconsistency_id(),
                        inconsistency_type="circular_causality",
                        memory_ids=list(cycle_memories),
                        description=f"Circular causality detected: {cycle_str}",
                        severity=0.8,
                        repair_suggestions=[
                            "Review the causal relationships between these memories",
                            "Determine which relationship is incorrect",
                            "Consider temporal ordering to resolve the cycle"
                        ]
                    )
                    inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def get_inconsistencies_for_memory(self, memory_id: str) -> List[TimelineInconsistency]:
        """
        Get all inconsistencies involving a specific memory.
        
        Args:
            memory_id: The ID of the memory
            
        Returns:
            List of inconsistencies involving the memory
        """
        return [inc for inc in self.inconsistencies if memory_id in inc.memory_ids]
    
    def get_inconsistencies_by_type(self, inconsistency_type: str) -> List[TimelineInconsistency]:
        """
        Get all inconsistencies of a specific type.
        
        Args:
            inconsistency_type: The type of inconsistencies to retrieve
            
        Returns:
            List of inconsistencies of the specified type
        """
        if inconsistency_type not in TimelineInconsistency.INCONSISTENCY_TYPES:
            raise ValueError(f"Invalid inconsistency type: {inconsistency_type}")
            
        return [inc for inc in self.inconsistencies if inc.inconsistency_type == inconsistency_type]
    
    def get_high_severity_inconsistencies(self, min_severity: float = 0.7) -> List[TimelineInconsistency]:
        """
        Get inconsistencies with severity above the specified threshold.
        
        Args:
            min_severity: Minimum severity threshold
            
        Returns:
            List of high-severity inconsistencies
        """
        return [inc for inc in self.inconsistencies if inc.severity >= min_severity]
    
    def repair_inconsistency(self, 
                             inconsistency_id: str, 
                             repair_method: str,
                             repair_params: Dict[str, Any] = None) -> bool:
        """
        Repair a specific inconsistency.
        
        Args:
            inconsistency_id: The ID of the inconsistency to repair
            repair_method: Method to use for repair ('adjust_timestamp', 'remove_memory', 'flag_memory', etc.)
            repair_params: Parameters for the repair method
            
        Returns:
            True if repair was successful
        """
        # Find the inconsistency
        inconsistency = next((inc for inc in self.inconsistencies if inc.inconsistency_id == inconsistency_id), None)
        if not inconsistency:
            logger.error(f"Inconsistency not found: {inconsistency_id}")
            return False
            
        repair_params = repair_params or {}
        
        try:
            if repair_method == "adjust_timestamp":
                memory_id = repair_params.get("memory_id")
                new_timestamp = repair_params.get("new_timestamp")
                
                if not memory_id or not new_timestamp:
                    logger.error("Missing required parameters for adjust_timestamp")
                    return False
                    
                memory = self.memory_store.get_memory(memory_id)
                if not memory:
                    logger.error(f"Memory not found: {memory_id}")
                    return False
                    
                # Update timestamp
                old_timestamp = memory.timestamp
                memory.timestamp = new_timestamp
                self.memory_store.update_memory(memory)
                
                inconsistency.mark_as_repaired(
                    f"Adjusted timestamp of memory {memory_id} from {old_timestamp} to {new_timestamp}"
                )
                return True
                
            elif repair_method == "remove_memory":
                memory_id = repair_params.get("memory_id")
                
                if not memory_id:
                    logger.error("Missing required parameters for remove_memory")
                    return False
                    
                memory = self.memory_store.get_memory(memory_id)
                if not memory:
                    logger.error(f"Memory not found: {memory_id}")
                    return False
                    
                # Remove memory
                self.memory_store.remove_memory(memory_id)
                
                inconsistency.mark_as_repaired(
                    f"Removed memory {memory_id} from the store"
                )
                return True
                
            elif repair_method == "flag_memory":
                memory_id = repair_params.get("memory_id")
                flag_reason = repair_params.get("reason", "Timeline inconsistency")
                
                if not memory_id:
                    logger.error("Missing required parameters for flag_memory")
                    return False
                    
                memory = self.memory_store.get_memory(memory_id)
                if not memory:
                    logger.error(f"Memory not found: {memory_id}")
                    return False
                    
                # Flag memory (assuming Memory class has metadata for flags)
                if not hasattr(memory, "metadata"):
                    memory.metadata = {}
                    
                memory.metadata["flagged"] = True
                memory.metadata["flag_reason"] = flag_reason
                memory.metadata["flag_timestamp"] = datetime.now().isoformat()
                self.memory_store.update_memory(memory)
                
                inconsistency.mark_as_repaired(
                    f"Flagged memory {memory_id} with reason: {flag_reason}"
                )
                return True
                
            else:
                logger.error(f"Unknown repair method: {repair_method}")
                return False
                
        except Exception as e:
            logger.error(f"Error repairing inconsistency {inconsistency_id}: {e}")
            return False
    
    def get_repair_suggestions(self, inconsistency_id: str) -> List[Dict[str, Any]]:
        """
        Get detailed repair suggestions for a specific inconsistency.
        
        Args:
            inconsistency_id: The ID of the inconsistency
            
        Returns:
            List of repair suggestion dictionaries with method and params
        """
        # Find the inconsistency
        inconsistency = next((inc for inc in self.inconsistencies if inc.inconsistency_id == inconsistency_id), None)
        if not inconsistency:
            logger.error(f"Inconsistency not found: {inconsistency_id}")
            return []
            
        suggestions = []
        
        if inconsistency.inconsistency_type == "temporal_contradiction":
            # For temporal contradictions, suggest adjusting timestamps or removing one memory
            memory_ids = inconsistency.memory_ids
            if len(memory_ids) >= 2:
                mem1 = self.memory_store.get_memory(memory_ids[0])
                mem2 = self.memory_store.get_memory(memory_ids[1])
                
                if mem1 and mem2:
                    # Suggest removing the less reliable memory
                    suggestions.append({
                        "method": "remove_memory",
                        "description": f"Remove potentially incorrect memory",
                        "params": {
                            "memory_id": memory_ids[0]
                        }
                    })
                    
                    suggestions.append({
                        "method": "remove_memory",
                        "description": f"Remove potentially incorrect memory",
                        "params": {
                            "memory_id": memory_ids[1]
                        }
                    })
                    
                    # Suggest adjusting timestamps
                    avg_time = mem1.timestamp + (mem2.timestamp - mem1.timestamp) / 2
                    
                    suggestions.append({
                        "method": "adjust_timestamp",
                        "description": f"Adjust timestamp of first memory",
                        "params": {
                            "memory_id": memory_ids[0],
                            "new_timestamp": avg_time
                        }
                    })
                    
                    suggestions.append({
                        "method": "adjust_timestamp",
                        "description": f"Adjust timestamp of second memory",
                        "params": {
                            "memory_id": memory_ids[1],
                            "new_timestamp": avg_time
                        }
                    })
                    
                    # Suggest flagging memories
                    suggestions.append({
                        "method": "flag_memory",
                        "description": "Flag memory as potentially inconsistent",
                        "params": {
                            "memory_id": memory_ids[0],
                            "reason": "Temporal contradiction with another memory"
                        }
                    })
                    
                    suggestions.append({
                        "method": "flag_memory",
                        "description": "Flag memory as potentially inconsistent",
                        "params": {
                            "memory_id": memory_ids[1],
                            "reason": "Temporal contradiction with another memory"
                        }
                    })
                    
        elif inconsistency.inconsistency_type == "impossible_sequence":
            # For impossible sequences, suggest adjusting timestamps
            memory_ids = inconsistency.memory_ids
            if len(memory_ids) >= 2:
                mem1 = self.memory_store.get_memory(memory_ids[0])
                mem2 = self.memory_store.get_memory(memory_ids[1])
                
                if mem1 and mem2:
                    # Calculate a reasonable time difference
                    reasonable_time = mem1.timestamp + timedelta(hours=3)
                    
                    suggestions.append({
                        "method": "adjust_timestamp",
                        "description": "Adjust timestamp to allow reasonable travel time",
                        "params": {
                            "memory_id": memory_ids[1],
                            "new_timestamp": reasonable_time
                        }
                    })
                    
                    suggestions.append({
                        "method": "flag_memory",
                        "description": "Flag memory as part of impossible sequence",
                        "params": {
                            "memory_id": memory_ids[0],
                            "reason": "Part of impossible travel sequence"
                        }
                    })
                    
                    suggestions.append({
                        "method": "flag_memory",
                        "description": "Flag memory as part of impossible sequence",
                        "params": {
                            "memory_id": memory_ids[1],
                            "reason": "Part of impossible travel sequence"
                        }
                    })
                    
        # Add more suggestion generators for other inconsistency types
        
        return suggestions
    
    def get_timeline_health_score(self) -> float:
        """
        Calculate an overall health score for the timeline consistency.
        
        Returns:
            Health score from 0.0-1.0 (higher is better)
        """
        # Count active (unrepaired) inconsistencies
        active_inconsistencies = [inc for inc in self.inconsistencies if not inc.repaired]
        
        if not active_inconsistencies:
            return 1.0
            
        # Calculate weighted score based on severity
        total_severity = sum(inc.severity for inc in active_inconsistencies)
        
        # Normalize by number of memories to account for timeline size
        memory_count = len(self.memory_store.get_all_memories())
        if memory_count == 0:
            return 1.0
            
        # Scale factor to keep reasonable scores even with many memories
        scale_factor = min(1.0, 10 / memory_count) if memory_count > 10 else 1.0
        
        # Calculate health score (1.0 is perfect)
        health_score = max(0.0, 1.0 - (total_severity * scale_factor / memory_count))
        
        return health_score 