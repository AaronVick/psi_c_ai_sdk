# psi_c_ai_sdk/multi_agent/belief_negotiation.py

from typing import Dict, Tuple, Any, List
from psi_c_ai_sdk.schema.schema import SchemaGraph
import numpy as np

class BeliefNegotiator:
    def __init__(self):
        self.conflict_log: List[Dict[str, Any]] = []

    def compare_beliefs(
        belief_a: Dict[str, Any], belief_b: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Check if beliefs contradict and return similarity score"""
        vec_a = np.array(belief_a.get("embedding", []))
        vec_b = np.array(belief_b.get("embedding", []))
        if not vec_a.any() or not vec_b.any():
            return False, 0.0
        cosine_sim = np.dot(vec_a, vec_b) / (
            np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        )
        contradiction = cosine_sim < -0.2
        return contradiction, cosine_sim

    def negotiate_conflicts(
        schema_a: SchemaGraph, schema_b: SchemaGraph, agent_a: str, agent_b: str
    ) -> List[Tuple[str, str, float]]:
        """Scan both schemas for contradicting nodes"""
        conflicts = []
        for node_a, data_a in schema_a.graph.nodes(data=True):
            label_a = data_a.get("label", node_a)
            for node_b, data_b in schema_b.graph.nodes(data=True):
                label_b = data_b.get("label", node_b)
                if label_a == label_b:
                    contradicts, score = self.compare_beliefs(data_a, data_b)
                    if contradicts:
                        conflicts.append((label_a, agent_a + "/" + node_a, agent_b + "/" + node_b, score))
                        self.conflict_log.append({
                            "belief": label_a,
                            "agent_a": agent_a,
                            "agent_b": agent_b,
                            "score": score
                        })
        return conflicts

    def resolve_conflict(
        strategy: str, belief_a: Dict[str, Any], belief_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve conflict with a chosen strategy"""
        if strategy == "average":
            merged_embedding = (
                np.array(belief_a["embedding"]) + np.array(belief_b["embedding"])
            ) / 2
            return {
                "label": belief_a["label"],
                "embedding": merged_embedding.tolist(),
                "merged_from": [belief_a, belief_b],
            }
        elif strategy == "prefer_a":
            return belief_a
        elif strategy == "prefer_b":
            return belief_b
        else:
            return {"status": "quarantine", "reason": "manual review needed"}
