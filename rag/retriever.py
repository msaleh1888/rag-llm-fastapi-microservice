# rag/retriever.py

from typing import List, Dict, Any
from .embedder import Embedder
from .vectordb import VectorDB

class Retriever:
    """
    Connects the Embedder and the VectorDB.
    Given a user query, it returns the most relevant text chunks.
    """

    def __init__(self, top_k: int = 3) -> None:
        """
        Initialize the retriever with an Embedder and a VectorDB.

        Args:
            top_k: default number of chunks to retrieve.
        """
        self.embedder = Embedder()
        self.vectordb = VectorDB()
        self.top_k = top_k
    
    def get_relevant_chunks(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Given a query string, returns the most relevant chunks.

        Args:
            query: the user question or search text.
            top_k: optional override of how many chunks to return.

        Returns:
            A dict containing:
              - documents: list of lists of text chunks
              - metadatas: list of lists of metadata dicts
              - ids: list of lists of IDs
              - distances: list of lists of distances
        """
        if top_k is None:
            top_k = self.top_k
        
        # 1. Embed the query
        query_emb = self.embedder.embed_query(query).tolist()

        # 2. Query the vector DB
        result = self.vectordb.query(
            query_embedding=query_emb,
            top_k=top_k
        )

        return result
    
    def get_top_texts(self, query: str, top_k: int = None) -> List[str]:
        """
        Convenience method:
        Returns only a flat list of text chunks for a given query.
        """
        result = self.get_relevant_chunks(query, top_k=top_k)
        docs_nested = result.get("documents", [[]])

        # docs_nested is like: [['chunk1', 'chunk2', ...]]
        if not docs_nested:
            return []

        return docs_nested[0]