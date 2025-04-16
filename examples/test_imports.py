"""
Test imports to verify the package structure.
"""

import os
import sys

# Add parent directory to path to import from psi_c_ai_sdk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test style 1: Import from package
print("Testing import style 1: Direct imports")
from psi_c_ai_sdk import MemoryStore, EmbeddingEngine, CoherenceScorer, ContradictionDetector

print(f"MemoryStore: {MemoryStore.__name__}")
print(f"EmbeddingEngine: {EmbeddingEngine.__name__}")
print(f"CoherenceScorer: {CoherenceScorer.__name__}")
print(f"ContradictionDetector: {ContradictionDetector.__name__}")

# Test style 2: Import from submodules
print("\nTesting import style 2: Submodule imports")
from psi_c_ai_sdk.memory import Memory, MemoryStore as MS
from psi_c_ai_sdk.embedding import EmbeddingEngine as EE
from psi_c_ai_sdk.coherence import CoherenceScorer as CS
from psi_c_ai_sdk.contradiction import ContradictionDetector as CD

print(f"Memory: {Memory.__name__}")
print(f"MemoryStore: {MS.__name__}")
print(f"EmbeddingEngine: {EE.__name__}")
print(f"CoherenceScorer: {CS.__name__}")
print(f"ContradictionDetector: {CD.__name__}")

print("\nAll imports successful!") 