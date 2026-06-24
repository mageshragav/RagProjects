"""ChromaDB vector store wrapper.

Handles storing, searching, and managing document chunks with embeddings.
This is the 'brain's memory' — everything the system knows lives here.
"""
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.embeddings import get_embeddings


class VectorStore:
    """Production-grade ChromaDB wrapper with metadata filtering."""
    
    def __init__(self):
        # Ensure directory exists
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding function
        self.embeddings = get_embeddings()
        self.collection_name = settings.CHROMA_COLLECTION
        
        # Create/connect to Chroma collection
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
    
    # ═══════════════════════════════════════════════════
    # WRITE OPERATIONS
    # ═══════════════════════════════════════════════════
    
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add document chunks to the vector store.
        
        Args:
            documents: List of LangChain Document objects with metadata
        
        Returns:
            List of generated chunk IDs
        """
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add to Chroma (auto-embeds using our embedding function)
        self._store.add_documents(documents=documents, ids=ids)
        
        return ids
    
    # ═══════════════════════════════════════════════════
    # READ OPERATIONS
    # ═══════════════════════════════════════════════════
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filters: dict | None = None,
        score_threshold: float = None,
    ) -> list[tuple[Document, float]]:
        """Search for chunks similar to the query.
        
        Args:
            query: User's question
            k: Number of results to return
            filters: Metadata filters (e.g., {"department": "billing"})
            score_threshold: Minimum similarity score (0.0 to 1.0)
        
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        k = k or settings.TOP_K
        threshold = score_threshold or settings.SIMILARITY_THRESHOLD
        
        # Build Chroma WHERE filter
        chroma_filter = self._build_chroma_filter(filters)
        import pdb;pdb.set_trace()
        # Search (returns more than k for filtering)
        results = self._store.similarity_search_with_relevance_scores(
            query=query,
            k=k * 2,  # Over-fetch in case some are below threshold
            filter=chroma_filter,
        )
        
        # Filter by minimum score and limit to k
        filtered = [
            (doc, score) for doc, score in results
            if score >= threshold
        ][:k]
        
        return filtered
    
    def _build_chroma_filter(self, filters: dict | None) -> dict | None:
        """Convert Python dict to Chroma WHERE clause.
        
        Example:
            {"department": "billing"} 
            → {"department": {"$eq": "billing"}}
            
            {"department": "billing", "doc_type": "policy"}
            → {"$and": [{"department": {"$eq": "billing"}}, {"doc_type": {"$eq": "policy"}}]}
        """
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None
    
    # ═══════════════════════════════════════════════════
    # DELETE OPERATIONS
    # ═══════════════════════════════════════════════════
    
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source file.
        
        Useful when a document is updated — delete old chunks, re-ingest new ones.
        """
        self._store.delete(where={"source": {"$eq": source}})
    
    # ═══════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_dir),
            "document_count": self._store._collection.count(),
        }


# Singleton instance — imported by other modules
vector_store = VectorStore()