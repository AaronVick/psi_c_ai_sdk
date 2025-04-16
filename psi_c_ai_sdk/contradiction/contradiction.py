"""
Contradiction Detection: Module for identifying semantic contradictions between memories.
"""

from typing import List, Dict, Set, Tuple, Optional, Union, Any
import re
import numpy as np

from ..memory.memory import Memory
from ..embedding.embedding import EmbeddingEngine


class ContradictionDetector:
    """
    Detects contradictions between memories using keyword and semantic analysis.
    
    Implements the contradiction detection heuristic as described in the formula:
    Contradict(A, B) = {
        1 if ∃k ∈ K: k ∈ A ∧ semantic_match(B)
        0 otherwise
    }
    
    Where:
    - K: Set of negation/contradiction keywords (not, never, false, etc.)
    """
    
    # Common negation keywords to detect contradictions
    DEFAULT_NEGATION_KEYWORDS = {
        "not", "never", "no", "false", "isn't", "aren't", "wasn't", "weren't",
        "doesn't", "don't", "didn't", "cannot", "can't", "won't", "wouldn't",
        "shouldn't", "couldn't", "impossible", "incorrect", "untrue", "opposite"
    }
    
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        negation_keywords: Optional[Set[str]] = None,
        similarity_threshold: float = 0.7,
        use_nli: bool = False,
        nli_threshold: float = 0.8
    ):
        """
        Initialize the contradiction detector.
        
        Args:
            embedding_engine: Embedding engine for semantic similarity
            negation_keywords: Set of words that indicate negation/contradiction
            similarity_threshold: Threshold for semantic similarity for contradiction
            use_nli: Whether to use more advanced NLI model for contradiction detection
            nli_threshold: Threshold for NLI-based contradiction confidence
        """
        self.embedding_engine = embedding_engine
        self.negation_keywords = negation_keywords or self.DEFAULT_NEGATION_KEYWORDS
        self.similarity_threshold = similarity_threshold
        self.use_nli = use_nli
        self.nli_threshold = nli_threshold
        
        # Lazy load the NLI model only if needed
        self.nli_model = None
    
    def _load_nli_model(self):
        """Load the NLI model if it's not already loaded."""
        if self.nli_model is None and self.use_nli:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                model_name = "facebook/bart-large-mnli"
                self.nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            except ImportError:
                # Fallback to keyword method if transformers is not available
                self.use_nli = False
                print(
                    "Warning: transformers package not found. "
                    "Falling back to keyword-based contradiction detection."
                )
    
    def contains_negation(self, text: str) -> bool:
        """
        Check if the text contains negation keywords.
        
        Args:
            text: Text to check for negation
            
        Returns:
            True if text contains negation keywords, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for each negation keyword
        for keyword in self.negation_keywords:
            # Use word boundary to ensure we match whole words
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def detect_keyword_contradiction(
        self, memory1: Memory, memory2: Memory
    ) -> Tuple[bool, float, str]:
        """
        Detect contradictions based on keywords and semantic similarity.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Tuple of (is_contradiction, confidence, explanation)
        """
        # Ensure embeddings exist
        if memory1.embedding is None:
            memory1.embedding = self.embedding_engine.get_embedding(memory1.content)
            
        if memory2.embedding is None:
            memory2.embedding = self.embedding_engine.get_embedding(memory2.content)
        
        # Calculate semantic similarity
        similarity = self.embedding_engine.cosine_similarity(
            memory1.embedding, memory2.embedding
        )
        
        # Check for similarities in content but with negation in one
        if similarity >= self.similarity_threshold:
            if self.contains_negation(memory1.content) or self.contains_negation(memory2.content):
                # Check if only one contains negation (potential contradiction)
                if self.contains_negation(memory1.content) != self.contains_negation(memory2.content):
                    confidence = similarity
                    explanation = (
                        f"Semantic similarity ({similarity:.2f}) above threshold "
                        f"with negation in only one memory"
                    )
                    return True, confidence, explanation
        
        return False, 0.0, "No contradiction detected"
    
    def detect_nli_contradiction(
        self, memory1: Memory, memory2: Memory
    ) -> Tuple[bool, float, str]:
        """
        Detect contradictions using Natural Language Inference model.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Tuple of (is_contradiction, confidence, explanation)
        """
        if not self.use_nli:
            return self.detect_keyword_contradiction(memory1, memory2)
        
        self._load_nli_model()
        
        try:
            import torch
            
            # Prepare inputs for model
            inputs = self.nli_tokenizer(
                memory1.content, memory2.content, 
                return_tensors="pt", padding=True, truncation=True
            )
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get contradiction probability (index 0: entailment, 1: neutral, 2: contradiction)
            contradiction_prob = predictions[0, 2].item()
            
            is_contradiction = contradiction_prob >= self.nli_threshold
            explanation = f"NLI model contradiction probability: {contradiction_prob:.3f}"
            
            return is_contradiction, contradiction_prob, explanation
            
        except Exception as e:
            # Fallback to keyword method if NLI fails
            print(f"NLI model error: {e}. Falling back to keyword detection.")
            return self.detect_keyword_contradiction(memory1, memory2)
    
    def detect_contradiction(
        self, memory1: Memory, memory2: Memory
    ) -> Tuple[bool, float, str]:
        """
        Detect contradictions between two memories.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Tuple of (is_contradiction, confidence, explanation)
        """
        if self.use_nli:
            return self.detect_nli_contradiction(memory1, memory2)
        else:
            return self.detect_keyword_contradiction(memory1, memory2)
    
    def create_contradiction_matrix(self, memories: List[Memory]) -> np.ndarray:
        """
        Create a contradiction matrix for a set of memories.
        
        Args:
            memories: List of memories to check for contradictions
            
        Returns:
            Contradiction matrix where 1 indicates contradiction
        """
        n = len(memories)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                is_contradiction, _, _ = self.detect_contradiction(
                    memories[i], memories[j]
                )
                
                # Contradiction is symmetric
                if is_contradiction:
                    matrix[i, j] = 1
                    matrix[j, i] = 1
        
        return matrix
    
    def find_contradictions(
        self, memories: List[Memory]
    ) -> List[Tuple[Memory, Memory, float, str]]:
        """
        Find all contradicting pairs in a set of memories.
        
        Args:
            memories: List of memories to check
            
        Returns:
            List of tuples (memory1, memory2, confidence, explanation)
        """
        contradictions = []
        
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                is_contradiction, confidence, explanation = self.detect_contradiction(
                    memories[i], memories[j]
                )
                
                if is_contradiction:
                    contradictions.append((
                        memories[i], memories[j], confidence, explanation
                    ))
        
        # Sort by confidence
        return sorted(contradictions, key=lambda x: x[2], reverse=True)
    
    def get_memory_contradiction_score(
        self, memory: Memory, memories: List[Memory]
    ) -> float:
        """
        Calculate how many contradictions a memory has with others.
        
        Args:
            memory: The memory to check
            memories: List of memories to check against
            
        Returns:
            Ratio of contradictions to total possible contradictions
        """
        contradictions = 0
        total = 0
        
        for other in memories:
            if memory.uuid != other.uuid:
                is_contradiction, _, _ = self.detect_contradiction(memory, other)
                if is_contradiction:
                    contradictions += 1
                total += 1
        
        return contradictions / max(1, total) 