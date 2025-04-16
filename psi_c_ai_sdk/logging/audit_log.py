# audit_log.py
# Purpose: Immutable audit log for trust-influenced memory, source activity, and epistemic state changes.

import hashlib
import json
from datetime import datetime
from typing import Dict, Any

class AuditLog:
    def __init__(self):
        self.entries = []
        self.last_hash = ""

    def _hash_entry(self, entry: Dict[str, Any]) -> str:
        entry_serialized = json.dumps(entry, sort_keys=True)
        full_string = entry_serialized + self.last_hash
        return hashlib.sha256(full_string.encode()).hexdigest()

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "timestamp": timestamp,
            "type": event_type,
            "payload": payload,
            "prev_hash": self.last_hash
        }
        entry_hash = self._hash_entry(entry)
        entry["entry_hash"] = entry_hash
        self.entries.append(entry)
        self.last_hash = entry_hash

    def export_log(self) -> list:
        return self.entries

    def save_to_file(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def verify_integrity(self) -> bool:
        prev_hash = ""
        for entry in self.entries:
            expected_hash = hashlib.sha256(
                (json.dumps({k: entry[k] for k in entry if k != "entry_hash"}, sort_keys=True) + prev_hash).encode()
            ).hexdigest()
            if entry["entry_hash"] != expected_hash:
                return False
            prev_hash = entry["entry_hash"]
        return True

    def recent_events(self, limit: int = 10) -> list:
        return self.entries[-limit:]
