from datetime import datetime
from psi_c_ai_sdk.epistemic_status import EpistemicStatus
from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.schema.schema_graph import SchemaGraph
from psi_c_ai_sdk.logging.audit_log import log_event


class TrustGate:
    def __init__(self, memory_store: MemoryStore, schema_graph: SchemaGraph, epistemic_status: EpistemicStatus):
        self.memory_store = memory_store
        self.schema_graph = schema_graph
        self.epistemic_status = epistemic_status

    def ingest_memory(self, memory_id: str, content: str, source_id: str, reflective: bool = False) -> bool:
        # Evaluate trust before ingesting
        if self.epistemic_status.should_throttle(source_id):
            log_event("trust_blocked", {
                "memory_id": memory_id,
                "source_id": source_id,
                "reason": "low trust + high entropy"
            })
            return False

        # Tag metadata
        self.epistemic_status.label_memory(memory_id, source_id, reflective)

        # Store memory
        self.memory_store.add_memory(memory_id, content)

        # Link in schema
        self.schema_graph.add_memory_node(memory_id, content)

        log_event("memory_ingested", {
            "memory_id": memory_id,
            "source_id": source_id,
            "reflective": reflective
        })
        return True

    def permit_schema_mutation(self, memory_id: str) -> bool:
        allowed = self.epistemic_status.is_major_belief_shift_allowed(memory_id)
        if not allowed:
            log_event("reflection_paused", {
                "memory_id": memory_id,
                "reason": "pause window active"
            })
        return allowed

    def quarantine_memory(self, memory_id: str):
        self.memory_store.archive_memory(memory_id)
        self.schema_graph.remove_node(memory_id)
        log_event("memory_quarantined", {
            "memory_id": memory_id,
            "timestamp": datetime.utcnow().isoformat()
        })

    def elevate_trust(self, source_id: str, delta: float = 0.05):
        self.epistemic_status.update_trust(source_id, delta)
        log_event("trust_adjusted", {
            "source_id": source_id,
            "delta": delta,
            "new_score": self.epistemic_status.get_trust(source_id)
        })

    def demote_trust(self, source_id: str, delta: float = -0.05):
        self.epistemic_status.update_trust(source_id, delta)
        log_event("trust_adjusted", {
            "source_id": source_id,
            "delta": delta,
            "new_score": self.epistemic_status.get_trust(source_id)
        })
