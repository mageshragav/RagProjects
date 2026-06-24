"""Document ingestion pipeline.

Takes raw files (PDF, MD, HTML, TXT) → extracts text → splits into chunks → stores in ChromaDB.
This is how the system 'learns' from documents.
"""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.vector_store import vector_store


class DocumentIngestor:
    """Handles messy real-world documents and turns them into searchable chunks."""
    
    # Map file extensions to their loaders
    LOADERS = {
        ".pdf": PDFPlumberLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
        ".html": UnstructuredHTMLLoader,
    }
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def ingest_file(self, file_path: str | Path, metadata: dict = None) -> list[str]:
        """Ingest a single file into the vector store.
        
        Args:
            file_path: Path to PDF, MD, HTML, or TXT file
            metadata: Extra metadata to attach (e.g., {"department": "billing"})
        
        Returns:
            List of chunk IDs stored in ChromaDB
        """
        path = Path(file_path)

        if path.suffix.lower() not in settings.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported: {settings.SUPPORTED_EXTENSIONS}"
            )
        # Step 1: Load document
        loader_cls = self.LOADERS.get(path.suffix.lower(), TextLoader)
        loader = loader_cls(path)
        documents = loader.load()

        # Step 2: Enrich metadata
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        base_metadata = {
            "source": path.name,
            "file_path": str(path),
            "file_type": path.suffix,
            "file_hash": file_hash,
            **(metadata or {}),
        }
        for doc in documents:
            doc.metadata.update(base_metadata)

        # Step 3: Split into chunks
        chunks = self.splitter.split_documents(documents)

        # Step 4: Add chunk index for citations
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        # Step 5: Store in ChromaDB
        ids = vector_store.add_documents(chunks)
        
        return ids

    def ingest_folder(self, dir_path: str, glob: str = "**/*") -> dict:
        """Batch ingest all supported files in a directory.
        
        Args:
            dir_path: Root directory to scan
            glob: Pattern to match files (default: all files recursively)
        
        Returns:
            Dict with 'processed' and 'failed' lists
        """
        path = Path(dir_path)
        results = {"processed": [], "failed": []}
        for file_path in path.glob(glob):
            if file_path.suffix.lower() in settings.SUPPORTED_EXTENSIONS:
                try:
                    ids = self.ingest_file(file_path)
                    results["processed"].append({
                        "file": str(file_path),
                        "chunks": len(ids),
                    })
                except Exception as e:
                    results["failed"].append({
                        "file": str(file_path),
                        "error": str(e),
                    })
        
        return results

ingestor = DocumentIngestor()
