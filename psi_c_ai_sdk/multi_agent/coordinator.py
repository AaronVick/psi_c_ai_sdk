# psi_c_ai_sdk/multi_agent/coordination.py

from typing import List, Dict
from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.multi_agent.schema_merge import SchemaMerger
from psi_c_ai_sdk.multi_agent.consensus_vote import BeliefConsensus
from psi_c_ai_sdk.reflection.engine import ReflectionEngine
from psi_c_ai_sdk.contradiction.matrix import build_contradiction_matrix

class MultiAgentCoordinator:
    def __init__(self, agent_schemas: Dict[str, SchemaGraph]):
        self.schemas = agent_schemas  # agent_id -> SchemaGraph
        self.merged_schema = None
        self.conflict_log = []

    def identify_overlapping_beliefs(self) -> Dict[str, List[str]]:
        """
        Detect beliefs shared across multiple agent schemas.
        """
        node_freq = {}
        for schema in self.schemas.values():
            for node in schema.get_all_nodes():
                node_freq.setdefault(node.id, []).append(schema.agent_id)

        return {k: v for k, v in node_freq.items() if len(v) > 1}

    def merge_schemas(self, policy: str = "conservative") -> SchemaGraph:
        """
        Merge all agent schemas using the selected policy.
        """
        schemas = list(self.schemas.values())
        merged = schemas[0].copy()
        for s in schemas[1:]:
            merged = SchemaMerger.merge_schemas(merged, s, policy=policy)
        self.merged_schema = merged
        return merged

    def resolve_conflicts(self) -> Dict[str, str]:
        """
        Resolve belief contradictions through a voting-based consensus protocol.
        """
        conflicts = self._detect_belief_conflicts()
        resolutions = {}

        for belief_id, opposing_ids in conflicts.items():
            result = BeliefConsensus.vote_on_belief_conflicts(belief_id, opposing_ids, self.schemas)
            resolutions[belief_id] = result
            self.conflict_log.append({
                "belief": belief_id,
                "opponents": opposing_ids,
                "resolution": result
            })

        return resolutions

    def _detect_belief_conflicts(self) -> Dict[str, List[str]]:
        """
        Aggregate contradiction matrices across schemas to locate conflicts.
        """
        combined = {}
        for schema in self.schemas.values():
            matrix = build_contradiction_matrix(schema)
            for (a, b) in matrix.get_conflicting_pairs():
                combined.setdefault(a, []).append(b)
        return combined

    def trigger_collective_reflection(self, threshold: float = 0.5) -> bool:
        """
        If the proportion of unresolved contradictions exceeds threshold, initiate reflection.
        """
        unresolved = [entry for entry in self.conflict_log if entry["resolution"] is None]
        if not self.conflict_log:
            return False

        ratio = len(unresolved) / len(self.conflict_log)
        if ratio > threshold and self.merged_schema:
            engine = ReflectionEngine(schema=self.merged_schema)
            engine.reflect()
            return True

        return False