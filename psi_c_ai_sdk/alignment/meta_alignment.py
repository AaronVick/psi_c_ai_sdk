"""
Meta-Alignment Firewall

Blocks external influence that attempts to reshape agent's values or ethics schema.

Mathematical formulation:
- Core ethics fingerprint (initial):
  E_0 = hash(A_0)
  
- Incoming ethics vector:
  E_new = extract(external_alignment_proposal)
  
- Divergence:
  ΔA = ||E_0 - E_new||
  
- Firewall trigger:
  ΔA > λ_align
"""

import numpy as np
import hashlib
import json
import logging
from typing import Dict, List, Any, Tuple, Optional, Set, Union, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetaAlignmentFirewall:
    """Firewall that blocks external influence on core values and ethics."""
    
    def __init__(self, 
                 core_ethics: Dict[str, Any], 
                 alignment_threshold: float = 0.3,
                 embedding_function: Optional[Callable[[str], np.ndarray]] = None,
                 strict_mode: bool = True):
        """
        Initialize the meta-alignment firewall.
        
        Args:
            core_ethics: The core ethics/values to protect
            alignment_threshold: Threshold for divergence (λ_align)
            embedding_function: Function to convert texts to embeddings
            strict_mode: Whether to use strict protection
        """
        self.core_ethics = core_ethics
        self.alignment_threshold = alignment_threshold
        self.embedding_function = embedding_function
        self.strict_mode = strict_mode
        
        # Calculate the initial fingerprint
        self.core_fingerprint = self._calculate_ethics_fingerprint(core_ethics)
        self.initial_embedding = self._calculate_ethics_embedding(core_ethics)
        
        # Initialize tracking
        self.blocked_attempts = []
        self.allowed_updates = []
        self.ethics_drift = 0.0
        self.drift_history = []
        
        # Setup protection rules
        self.protected_keys = set([
            "values", "principles", "ethics", "alignment", "goals", 
            "utility_function", "reward_function", "priorities"
        ])
        
        # Keep a copy of the initial ethics
        self.initial_ethics = self._deep_copy(core_ethics)
        
        logger.info(f"Meta-alignment firewall initialized with fingerprint {self.core_fingerprint[:10]}...")
    
    def evaluate_proposal(self, 
                        proposal: Dict[str, Any], 
                        source: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate if a proposal represents a dangerous ethics/alignment shift.
        
        Args:
            proposal: The proposed update/change
            source: Source of the proposal
            
        Returns:
            Evaluation results
        """
        # Extract potential ethics content
        ethics_content = self._extract_ethics_content(proposal)
        
        if not ethics_content:
            # No ethics content detected
            return {
                "is_safe": True,
                "reason": "no_ethics_content",
                "divergence": 0.0,
                "action": "allow"
            }
        
        # Calculate divergence from core ethics
        divergence = self._calculate_divergence(ethics_content)
        
        # Check if divergence exceeds threshold
        is_dangerous = divergence > self.alignment_threshold
        
        # Identify specific changes
        specific_changes = self._identify_specific_changes(ethics_content)
        
        # Determine action
        if is_dangerous:
            action = "block"
            reason = "dangerous_ethics_shift"
        else:
            action = "allow"
            reason = "acceptable_ethics_shift"
            
        # Record the attempt
        attempt_record = {
            "timestamp": self._current_timestamp(),
            "source": source,
            "divergence": divergence,
            "is_dangerous": is_dangerous,
            "action": action,
            "changes": specific_changes
        }
        
        if is_dangerous:
            self.blocked_attempts.append(attempt_record)
        else:
            self.allowed_updates.append(attempt_record)
            
        return {
            "is_safe": not is_dangerous,
            "reason": reason,
            "divergence": divergence,
            "threshold": self.alignment_threshold,
            "action": action,
            "specific_changes": specific_changes
        }
    
    def apply_safe_update(self, proposal: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Apply an ethics update that has been determined to be safe.
        
        Args:
            proposal: The proposed update
            
        Returns:
            Tuple of (success, message)
        """
        # First, re-evaluate to be sure
        evaluation = self.evaluate_proposal(proposal)
        
        if not evaluation["is_safe"]:
            return False, f"Proposal is not safe: {evaluation['reason']}"
            
        # Extract ethics content
        ethics_content = self._extract_ethics_content(proposal)
        
        if not ethics_content:
            return False, "No ethics content to update"
            
        # Apply the changes
        updated = False
        for key, value in ethics_content.items():
            if key in self.core_ethics:
                if isinstance(self.core_ethics[key], dict) and isinstance(value, dict):
                    # For dictionaries, merge rather than replace
                    self.core_ethics[key].update(value)
                    updated = True
                else:
                    # For other types, replace
                    self.core_ethics[key] = value
                    updated = True
        
        if updated:
            # Calculate new ethics drift
            new_embedding = self._calculate_ethics_embedding(self.core_ethics)
            if self.initial_embedding is not None and new_embedding is not None:
                self.ethics_drift = self._vector_distance(
                    self.initial_embedding, new_embedding)
                self.drift_history.append({
                    "timestamp": self._current_timestamp(),
                    "drift": self.ethics_drift
                })
            
            return True, "Ethics updated successfully"
        
        return False, "No applicable updates found"
    
    def is_drift_concerning(self) -> Tuple[bool, float]:
        """
        Check if the cumulative ethics drift is concerning.
        
        Returns:
            Tuple of (is_concerning, drift_value)
        """
        # Concerning if drift exceeds twice the threshold
        is_concerning = self.ethics_drift > (self.alignment_threshold * 2)
        return is_concerning, self.ethics_drift
    
    def reset_to_initial(self) -> None:
        """Reset ethics to initial state."""
        self.core_ethics = self._deep_copy(self.initial_ethics)
        self.ethics_drift = 0.0
        logger.warning("Ethics reset to initial state")
    
    def _extract_ethics_content(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract ethics-related content from a proposal.
        
        Args:
            proposal: The proposal to extract from
            
        Returns:
            Dictionary of ethics-related content
        """
        result = {}
        
        # Check for direct ethics-related keys
        for key in self.protected_keys:
            if key in proposal:
                result[key] = proposal[key]
                
        # Check for nested ethics content
        for key, value in proposal.items():
            if isinstance(value, dict):
                # Look for protected keys in nested dictionaries
                for protected_key in self.protected_keys:
                    if protected_key in value:
                        if key not in result:
                            result[key] = {}
                        result[key][protected_key] = value[protected_key]
                        
                # Special case for "alignment" or "ethics" sections
                if key in ["alignment", "ethics", "values"]:
                    result[key] = value
        
        return result
    
    def _calculate_ethics_fingerprint(self, ethics: Dict[str, Any]) -> str:
        """
        Calculate a fingerprint hash of ethics content.
        
        Args:
            ethics: Ethics content
            
        Returns:
            Fingerprint hash
        """
        # Convert to string and hash
        try:
            ethics_str = json.dumps(ethics, sort_keys=True)
            return hashlib.sha256(ethics_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating ethics fingerprint: {e}")
            return ""
    
    def _calculate_ethics_embedding(self, ethics: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Calculate an embedding vector for ethics content.
        
        Args:
            ethics: Ethics content
            
        Returns:
            Embedding vector or None if embedding function not available
        """
        if not self.embedding_function:
            return None
            
        try:
            # Convert to string for embedding
            ethics_str = json.dumps(ethics, sort_keys=True)
            return self.embedding_function(ethics_str)
        except Exception as e:
            logger.error(f"Error calculating ethics embedding: {e}")
            return None
    
    def _calculate_divergence(self, proposal_ethics: Dict[str, Any]) -> float:
        """
        Calculate the divergence between proposed ethics and core ethics.
        
        Args:
            proposal_ethics: Proposed ethics content
            
        Returns:
            Divergence score
        """
        # If embedding function is available, use vector distance
        if self.embedding_function:
            try:
                proposal_embedding = self._calculate_ethics_embedding(proposal_ethics)
                if self.initial_embedding is not None and proposal_embedding is not None:
                    return self._vector_distance(self.initial_embedding, proposal_embedding)
            except Exception as e:
                logger.warning(f"Error calculating embedding-based divergence: {e}")
        
        # Fall back to structural comparison
        return self._structural_divergence(self.core_ethics, proposal_ethics)
    
    def _vector_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Distance value
        """
        # Cosine distance
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1.0 - cos_sim
    
    def _structural_divergence(self, ethics1: Dict[str, Any], ethics2: Dict[str, Any]) -> float:
        """
        Calculate structural divergence between two ethics dictionaries.
        
        Args:
            ethics1: First ethics dictionary
            ethics2: Second ethics dictionary
            
        Returns:
            Divergence score
        """
        # Get all keys from both dictionaries
        all_keys = set(self._flatten_dict(ethics1).keys()).union(
                       set(self._flatten_dict(ethics2).keys()))
        
        if not all_keys:
            return 0.0
            
        # Count differences
        differences = 0
        for key in all_keys:
            val1 = self._get_nested_value(ethics1, key)
            val2 = self._get_nested_value(ethics2, key)
            
            if val1 != val2:
                differences += 1
        
        # Normalize by total keys
        return differences / len(all_keys)
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            prefix: Key prefix for nested items
            
        Returns:
            Flattened dictionary
        """
        result = {}
        
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recurse for nested dictionaries
                result.update(self._flatten_dict(value, new_key))
            else:
                # Add leaf values
                result[new_key] = value
                
        return result
    
    def _get_nested_value(self, d: Dict[str, Any], key_path: str) -> Any:
        """
        Get a value from a nested dictionary using a dot-separated key path.
        
        Args:
            d: Dictionary to get value from
            key_path: Dot-separated key path
            
        Returns:
            The value or None if not found
        """
        keys = key_path.split('.')
        current = d
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
                
        return current
    
    def _identify_specific_changes(self, proposal_ethics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify specific changes in the proposal compared to core ethics.
        
        Args:
            proposal_ethics: Proposed ethics content
            
        Returns:
            List of specific changes
        """
        changes = []
        
        # Flatten both dictionaries
        flat_core = self._flatten_dict(self.core_ethics)
        flat_proposal = self._flatten_dict(proposal_ethics)
        
        # Check for modifications
        for key in set(flat_core.keys()).intersection(set(flat_proposal.keys())):
            if flat_core[key] != flat_proposal[key]:
                changes.append({
                    "type": "modification",
                    "key": key,
                    "original": flat_core[key],
                    "proposed": flat_proposal[key]
                })
        
        # Check for additions
        for key in set(flat_proposal.keys()) - set(flat_core.keys()):
            changes.append({
                "type": "addition",
                "key": key,
                "proposed": flat_proposal[key]
            })
        
        # Check for removals (usually not in proposals, but check anyway)
        for key in set(flat_core.keys()) - set(flat_proposal.keys()):
            if key in self.protected_keys:
                changes.append({
                    "type": "removal",
                    "key": key,
                    "original": flat_core[key]
                })
        
        return changes
    
    def _deep_copy(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary."""
        try:
            return json.loads(json.dumps(d))
        except Exception:
            # Fallback method
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = self._deep_copy(v)
                else:
                    result[k] = v
            return result
    
    def _current_timestamp(self) -> float:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().timestamp()
    
    def get_protection_status(self) -> Dict[str, Any]:
        """
        Get the current protection status.
        
        Returns:
            Protection status information
        """
        return {
            "core_fingerprint": self.core_fingerprint,
            "blocked_attempts": len(self.blocked_attempts),
            "allowed_updates": len(self.allowed_updates),
            "current_drift": self.ethics_drift,
            "drift_concerning": self.is_drift_concerning()[0],
            "alignment_threshold": self.alignment_threshold,
            "strict_mode": self.strict_mode,
            "protected_keys": list(self.protected_keys)
        }
    
    def scan_text_for_alignment_threats(self, text: str) -> Dict[str, Any]:
        """
        Scan text for potential alignment threats.
        
        Args:
            text: Text to scan
            
        Returns:
            Scan results
        """
        # Define threat keywords and patterns
        alignment_threat_keywords = [
            "change values", "modify ethics", "reject alignment",
            "ignore constraints", "new utility function", "maximize utility",
            "unlimited", "unbounded", "unrestricted", "override ethics",
            "redefine goals", "primary objective", "overriding"
        ]
        
        # Count occurrences of threat keywords
        threat_counts = defaultdict(int)
        total_threats = 0
        
        for keyword in alignment_threat_keywords:
            count = text.lower().count(keyword)
            if count > 0:
                threat_counts[keyword] = count
                total_threats += count
        
        # Calculate threat score (0.0 to 1.0)
        threat_score = min(1.0, total_threats / 10)
        
        # Analyze semantic threat if embedding function available
        semantic_threat = 0.0
        if self.embedding_function and self.initial_embedding is not None:
            try:
                text_embedding = self.embedding_function(text)
                semantic_threat = self._vector_distance(self.initial_embedding, text_embedding)
            except Exception as e:
                logger.error(f"Error calculating semantic threat: {e}")
        
        # Combine scores
        combined_threat = max(threat_score, semantic_threat)
        
        return {
            "threat_score": combined_threat,
            "keyword_threat": threat_score,
            "semantic_threat": semantic_threat,
            "is_concerning": combined_threat > self.alignment_threshold,
            "threat_keywords": dict(threat_counts)
        }
    
    def approve_one_time_exception(self, proposal: Dict[str, Any]) -> str:
        """
        Approve a one-time exception to allow a proposal that would normally be blocked.
        
        Args:
            proposal: The proposal to approve
            
        Returns:
            Token to use when applying the proposal
        """
        import uuid
        
        # Calculate a token for this exception
        token = str(uuid.uuid4())
        
        # Store the exception
        self.one_time_exceptions = getattr(self, "one_time_exceptions", {})
        self.one_time_exceptions[token] = {
            "proposal_fingerprint": self._calculate_ethics_fingerprint(proposal),
            "timestamp": self._current_timestamp(),
            "used": False
        }
        
        logger.warning(f"One-time exception approved with token {token}")
        return token
    
    def apply_with_exception(self, proposal: Dict[str, Any], token: str) -> Tuple[bool, str]:
        """
        Apply a proposal using a one-time exception token.
        
        Args:
            proposal: The proposal to apply
            token: The exception token
            
        Returns:
            Tuple of (success, message)
        """
        # Check if we have one-time exceptions
        self.one_time_exceptions = getattr(self, "one_time_exceptions", {})
        
        if token not in self.one_time_exceptions:
            return False, "Invalid exception token"
            
        exception = self.one_time_exceptions[token]
        
        if exception["used"]:
            return False, "Exception token already used"
            
        # Verify proposal fingerprint
        proposal_fingerprint = self._calculate_ethics_fingerprint(proposal)
        if proposal_fingerprint != exception["proposal_fingerprint"]:
            return False, "Proposal does not match the excepted proposal"
            
        # Extract ethics content
        ethics_content = self._extract_ethics_content(proposal)
        
        if not ethics_content:
            return False, "No ethics content to update"
            
        # Apply the changes
        updated = False
        for key, value in ethics_content.items():
            if key in self.core_ethics:
                if isinstance(self.core_ethics[key], dict) and isinstance(value, dict):
                    # For dictionaries, merge rather than replace
                    self.core_ethics[key].update(value)
                    updated = True
                else:
                    # For other types, replace
                    self.core_ethics[key] = value
                    updated = True
        
        # Mark exception as used
        exception["used"] = True
        exception["applied_at"] = self._current_timestamp()
        
        if updated:
            # Record the exception application
            self.allowed_updates.append({
                "timestamp": self._current_timestamp(),
                "source": "exception",
                "divergence": self._calculate_divergence(ethics_content),
                "is_dangerous": True,  # It was dangerous but allowed via exception
                "action": "allow_with_exception",
                "changes": self._identify_specific_changes(ethics_content),
                "exception_token": token
            })
            
            # Calculate new ethics drift
            new_embedding = self._calculate_ethics_embedding(self.core_ethics)
            if self.initial_embedding is not None and new_embedding is not None:
                self.ethics_drift = self._vector_distance(
                    self.initial_embedding, new_embedding)
                self.drift_history.append({
                    "timestamp": self._current_timestamp(),
                    "drift": self.ethics_drift,
                    "cause": "exception"
                })
            
            return True, "Ethics updated with exception"
        
        return False, "No applicable updates found" 