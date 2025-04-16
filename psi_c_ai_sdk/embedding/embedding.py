"""
Embedding Engine: Module for generating and managing semantic embeddings of memory content.
"""

from typing import List, Dict, Optional, Union, Any
import os
import json
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """
    Generates and manages embeddings for memory content using sentence transformers.
    
    The EmbeddingEngine uses pre-trained language models to convert text into
    vector representations that can be used for semantic similarity calculations,
    contradiction detection, and schema building.
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    CACHE_DIR = ".embedding_cache"
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the sentence transformer model to use
            use_cache: Whether to cache embeddings to avoid redundant generation
            cache_dir: Directory to store embedding cache (default: .embedding_cache)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.use_cache = use_cache
        
        # Set up caching
        self.cache_dir = cache_dir or self.CACHE_DIR
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, f"{model_name}_cache.json")
            self.load_cache()
    
    def load_cache(self) -> None:
        """Load the embedding cache from disk if it exists."""
        self.cache: Dict[str, List[float]] = {}
        
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If there's an error loading the cache, start with an empty cache
                self.cache = {}
    
    def save_cache(self) -> None:
        """Save the embedding cache to disk."""
        if self.use_cache:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
    
    def get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Args:
            text: The text to generate a cache key for
            
        Returns:
            A hash string to use as a cache key
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a piece of text.
        
        Args:
            text: The text to embed
            
        Returns:
            A vector embedding of the text
        """
        if not text.strip():
            # Return a zero vector for empty text
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        if self.use_cache:
            cache_key = self.get_cache_key(text)
            
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        # Cache the result
        if self.use_cache:
            cache_key = self.get_cache_key(text)
            self.cache[cache_key] = embedding_list
            
            # Periodically save the cache (could be optimized to save less frequently)
            if len(self.cache) % 10 == 0:
                self.save_cache()
        
        return embedding_list
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings
        """
        # Filter out texts that need to be embedded
        texts_to_embed = []
        cached_embeddings = {}
        indices = []
        
        for i, text in enumerate(texts):
            if not text.strip():
                # Handle empty text
                cached_embeddings[i] = [0.0] * self.model.get_sentence_embedding_dimension()
                continue
                
            if self.use_cache:
                cache_key = self.get_cache_key(text)
                if cache_key in self.cache:
                    cached_embeddings[i] = self.cache[cache_key]
                    continue
            
            texts_to_embed.append(text)
            indices.append(i)
        
        # If all embeddings were cached, return them
        if not texts_to_embed:
            return [cached_embeddings[i] for i in range(len(texts))]
        
        # Generate embeddings for remaining texts
        new_embeddings = self.model.encode(texts_to_embed)
        
        # Convert to list and cache results
        result = [None] * len(texts)
        for idx, embedding in zip(indices, new_embeddings):
            embedding_list = embedding.tolist()
            result[idx] = embedding_list
            
            if self.use_cache:
                cache_key = self.get_cache_key(texts[idx])
                self.cache[cache_key] = embedding_list
        
        # Fill in cached embeddings
        for i, emb in cached_embeddings.items():
            result[i] = emb
        
        # Save cache if we added new embeddings
        if texts_to_embed and self.use_cache:
            self.save_cache()
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.use_cache:
            self.cache = {}
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings produced by this engine.
        
        Returns:
            The dimensionality of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Convert to numpy arrays for efficient calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2)) 