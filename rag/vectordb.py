# rag/vectordb.py

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorDB:
    """
    Simple wrapper around ChromaDB for storing and querying document embeddings.
    """

    def __init__(
            self,
            collection_name: str = "documents",
            persist_directory: str = "chroma_db",
    ) -> None:
        """
        Initializes a persistent Chroma client and gets/creates a collection.

        Args:
            collection_name: Name of the Chroma collection.
            persist_directory: Directory on disk to store the Chroma DB.
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    def add_texts(
            self,
            texts: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents and their embeddings to the vector database.

        Args:
            texts: List of text chunks.
            embeddings: List of embedding vectors (same length as texts).
            metadatas: Optional list of metadata dicts for each text.
            ids: Optional list of IDs. If None, they will be auto-generated.
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Query the vector database for the most similar documents.

        Args:
            query_embedding: Embedding vector for the query.
            top_k: Number of top results to return.

        Returns:
            A dict with 'documents', 'metadatas', 'ids', and 'distances'.
        """
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return result
