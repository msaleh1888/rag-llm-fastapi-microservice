# rag/embedder.py

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    """
    Simple wrapper around a SentenceTransformer model.
    Responsible for turning text into numerical vectors (embeddings).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the embedding model.
        
        Args:
            model_name: name of the SentenceTransformer model to load.
            device: "cpu" or "cuda". If None, SentenceTransformer decides automatically.
        """
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of documents (chunks).
        
        Args:
            texts: list of strings to embed.
        
        Returns:
            A NumPy array of shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            text: the query to embed.
        
        Returns:
            A 1D NumPy array representing the query embedding.
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding