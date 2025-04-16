# psi_c_ai_sdk/reflection/collective.py

from typing import List
from psi_c_ai_sdk.agent.interface import PsiCAgentInterface

def trigger_collective_reflection(agents: List[PsiCAgentInterface], conflict_key: str):
    """
    Ask all agents to reflect if a shared contradiction cannot be resolved.
    """
    for agent in agents:
        agent.trigger_reflection(f"Collective contradiction on: {conflict_key}")
