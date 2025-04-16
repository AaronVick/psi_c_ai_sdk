"""
Epistemic Trust Throttler

Dynamically adjusts trust in sources with high persuasion entropy, protecting
against manipulation through repeated claims or inconsistent information streams.
"""

import math
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter

@dataclass
class SourceTrustProfile:
    """Records trust metrics for an information source."""
    
    source_id: str
    initial_trust: float = 0.7
    current_trust: float = 0.7
    claim_history: Dict[str, List[float]] = field(default_factory=dict)
    persuasion_entropy: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def record_claim(self, claim_id: str, timestamp: Optional[float] = None) -> None:
        """Record a new claim from this source."""
        if timestamp is None:
            timestamp = time.time()
            
        if claim_id not in self.claim_history:
            self.claim_history[claim_id] = []
            
        self.claim_history[claim_id].append(timestamp)
        self.last_updated = timestamp


class TrustThrottler:
    """
    Dynamically adjusts trust in information sources based on persuasion patterns.
    
    Implements the persuasion entropy calculation:
    H_persuasion = -∑ p_i log p_i, where p_i = frequency(claim_i)/time_window
    
    Trust adjustment:
    T_k(t+1) = T_k(t) - η · H_persuasion
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1, 
                 time_window: float = 3600.0,
                 trust_floor: float = 0.1,
                 trust_ceiling: float = 1.0,
                 default_initial_trust: float = 0.7):
        """
        Initialize the trust throttler.
        
        Args:
            learning_rate: Rate of trust adjustment (η)
            time_window: Time window in seconds for entropy calculation
            trust_floor: Minimum trust value
            trust_ceiling: Maximum trust value
            default_initial_trust: Default trust for new sources
        """
        self.learning_rate = learning_rate
        self.time_window = time_window
        self.trust_floor = trust_floor
        self.trust_ceiling = trust_ceiling
        self.default_initial_trust = default_initial_trust
        
        self.source_profiles: Dict[str, SourceTrustProfile] = {}
        self.global_claims: Dict[str, Dict[str, List[float]]] = {}
    
    def record_claim(self, source_id: str, claim_id: str, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Record a claim from a source and update trust accordingly.
        
        Args:
            source_id: Identifier for the information source
            claim_id: Identifier for the specific claim
            timestamp: Optional timestamp of the claim
            
        Returns:
            Dictionary with updated trust information
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Get or create source profile
        if source_id not in self.source_profiles:
            self.source_profiles[source_id] = SourceTrustProfile(
                source_id=source_id,
                initial_trust=self.default_initial_trust,
                current_trust=self.default_initial_trust
            )
        
        profile = self.source_profiles[source_id]
        
        # Record the claim
        profile.record_claim(claim_id, timestamp)
        
        # Update global claims
        if claim_id not in self.global_claims:
            self.global_claims[claim_id] = {}
            
        if source_id not in self.global_claims[claim_id]:
            self.global_claims[claim_id][source_id] = []
            
        self.global_claims[claim_id][source_id].append(timestamp)
        
        # Calculate persuasion entropy and update trust
        self._update_persuasion_entropy(source_id)
        self._update_trust(source_id)
        
        return {
            "source_id": source_id,
            "current_trust": profile.current_trust,
            "persuasion_entropy": profile.persuasion_entropy,
            "timestamp": timestamp
        }
    
    def get_trust(self, source_id: str) -> float:
        """Get the current trust score for a source."""
        if source_id not in self.source_profiles:
            return self.default_initial_trust
            
        return self.source_profiles[source_id].current_trust
    
    def get_source_profile(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get the full profile for a source."""
        if source_id not in self.source_profiles:
            return None
            
        profile = self.source_profiles[source_id]
        return {
            "source_id": profile.source_id,
            "initial_trust": profile.initial_trust,
            "current_trust": profile.current_trust,
            "persuasion_entropy": profile.persuasion_entropy,
            "claim_count": sum(len(timestamps) for timestamps in profile.claim_history.values()),
            "unique_claims": len(profile.claim_history),
            "last_updated": profile.last_updated
        }
    
    def _update_persuasion_entropy(self, source_id: str) -> float:
        """
        Calculate the persuasion entropy for a source.
        
        Persuasion entropy measures how much a source repeats the same claims
        within the time window. Higher entropy means more balanced claims,
        while lower entropy suggests repetition of the same claims (potential persuasion).
        """
        profile = self.source_profiles[source_id]
        now = time.time()
        
        # Count recent claims within time window
        claim_counts = Counter()
        total_claims = 0
        
        for claim_id, timestamps in profile.claim_history.items():
            recent_claims = [ts for ts in timestamps if now - ts <= self.time_window]
            count = len(recent_claims)
            
            if count > 0:
                claim_counts[claim_id] = count
                total_claims += count
        
        # Calculate entropy
        entropy = 0.0
        if total_claims > 0:
            for claim_id, count in claim_counts.items():
                p_i = count / total_claims
                entropy -= p_i * math.log(p_i)
                
        # Normalize to [0, 1]
        max_possible_entropy = math.log(len(claim_counts)) if claim_counts else 0
        normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 0
        
        # Invert: high entropy (diverse claims) -> low persuasion entropy
        persuasion_entropy = 1.0 - normalized_entropy
        
        profile.persuasion_entropy = persuasion_entropy
        return persuasion_entropy
    
    def _update_trust(self, source_id: str) -> float:
        """
        Update trust based on persuasion entropy.
        T_k(t+1) = T_k(t) - η · H_persuasion
        """
        profile = self.source_profiles[source_id]
        
        # Apply the trust update formula
        delta = -self.learning_rate * profile.persuasion_entropy
        new_trust = profile.current_trust + delta
        
        # Enforce bounds
        profile.current_trust = max(self.trust_floor, min(self.trust_ceiling, new_trust))
        
        return profile.current_trust
    
    def get_high_persuasion_sources(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get sources with high persuasion entropy (potential manipulation)."""
        high_persuasion = []
        
        for source_id, profile in self.source_profiles.items():
            if profile.persuasion_entropy >= threshold:
                high_persuasion.append({
                    "source_id": source_id,
                    "persuasion_entropy": profile.persuasion_entropy,
                    "current_trust": profile.current_trust
                })
                
        return sorted(high_persuasion, key=lambda x: x["persuasion_entropy"], reverse=True)
    
    def calculate_cross_source_entropy(self) -> Dict[str, float]:
        """Calculate entropy across sources for each claim (identifies contested claims)."""
        result = {}
        
        for claim_id, sources in self.global_claims.items():
            # Count recent claims per source
            now = time.time()
            source_counts = Counter()
            total = 0
            
            for source_id, timestamps in sources.items():
                recent = [ts for ts in timestamps if now - ts <= self.time_window]
                count = len(recent)
                if count > 0:
                    source_counts[source_id] = count
                    total += count
            
            # Calculate entropy
            entropy = 0.0
            if total > 0:
                for source_id, count in source_counts.items():
                    p_i = count / total
                    entropy -= p_i * math.log(p_i)
            
            # Normalize
            max_possible = math.log(len(source_counts)) if source_counts else 0
            normalized = entropy / max_possible if max_possible > 0 else 0
            
            result[claim_id] = normalized
            
        return result
    
    def reset_source(self, source_id: str) -> bool:
        """Reset a source to its initial trust level."""
        if source_id not in self.source_profiles:
            return False
            
        profile = self.source_profiles[source_id]
        profile.current_trust = profile.initial_trust
        profile.persuasion_entropy = 0.0
        return True 