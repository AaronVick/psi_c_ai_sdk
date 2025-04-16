# psi_c_ai_sdk/multi_agent/collective_reflection.py

from typing import List, Dict, Any, Tuple
from psi_c_ai_sdk.multi_agent.belief_negotiation import BeliefNegotiator
from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.reflection.reflection_engine import ReflectionEngine

class CollectiveReflection:
    def __init__(self):
        self.negotiator = BeliefNegotiator()
        self.session_log: List[Dict[str, Any]] = []

    def initiate_collective_reflection(
        agents: List[Tuple[str, SchemaGraph, ReflectionEngine]]
    ) -> Dict[str, Any]:
        """
        Triggers a multi-agent reflection session. Assumes each tuple is (agent_id, schema, reflection_engine).
        """
        contradictions = []
        updates = {}

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent_a, schema_a, _ = agents[i]
                agent_b, schema_b, _ = agents[j]
                conflicts = BeliefNegotiator().negotiate_conflicts(
                    schema_a, schema_b, agent_a, agent_b
                )
                contradictions.extend(conflicts)

        # If no contradictions, no collective reflection needed
        if not contradictions:
            return {"status": "no_conflict", "message": "Schemas consistent."}

        for agent_id, schema, reflector in agents:
            reflection_result = reflector.reflect(trigger="collective", reason="inter-agent contradiction")
            updates[agent_id] = reflection_result

        session = {
            "agents": [a[0] for a in agents],
            "contradictions_found": len(contradictions),
            "reflection_updates": updates,
        }
        return session
