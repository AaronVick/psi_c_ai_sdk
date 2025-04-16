# psi_c_ai_sdk/examples/agents.py

from psi_c_ai_sdk.agent.interface import PsiCAgentInterface
from psi_c_ai_sdk.schema.schema_graph import SchemaGraph

class DummyPsiCAgent(PsiCAgentInterface):
    def __init__(self, name):
        self.name = name
        self.schema = SchemaGraph.load_from_file(f"{name}_schema.json")

    def get_schema(self):
        return self.schema

    def get_identity_fingerprint(self):
        return f"agent:{self.name}"

    def trigger_reflection(self, reason):
        print(f"[{self.name}] Reflecting due to: {reason}")

    def get_belief_vector(self):
        return self.schema.compute_belief_vector()

def get_active_agents():
    return [DummyPsiCAgent("alpha"), DummyPsiCAgent("beta")]
