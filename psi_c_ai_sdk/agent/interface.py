# psi_c_ai_sdk/agent/interface.py

from abc import ABC, abstractmethod
from typing import Any
from psi_c_ai_sdk.schema.schema_graph import SchemaGraph

class PsiCAgentInterface(ABC):
    @abstractmethod
    def get_schema(self) -> SchemaGraph:
        pass

    @abstractmethod
    def get_identity_fingerprint(self) -> str:
        pass

    @abstractmethod
    def trigger_reflection(self, reason: str) -> None:
        pass

    @abstractmethod
    def get_belief_vector(self) -> Any:
        """Optional: Used for goal alignment or value negotiation."""
        pass
