# psi_c_ai_sdk/multi_agent/group_schema_snapshot.py

from typing import Dict
from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.schema.fingerprint import diff_fingerprints, fingerprint_schema

class GroupSchemaSnapshot:
    def __init__(self):
        self.snapshots = {}

    def save_snapshot(self, agent_id: str, schema: SchemaGraph):
        self.snapshots[agent_id] = fingerprint_schema(schema)

    def compare_agents(self, agent_a: str, agent_b: str) -> Dict:
        fp_a = self.snapshots.get(agent_a)
        fp_b = self.snapshots.get(agent_b)
        if not fp_a or not fp_b:
            return {"error": "Missing snapshots"}

        diff = diff_fingerprints(fp_a, fp_b)
        return {
            "agents": (agent_a, agent_b),
            "differences": diff,
            "count": len(diff)
        }
