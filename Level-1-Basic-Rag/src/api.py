"""FastAPI server — the external interface for the RAG system.

All business logic lives in the pipeline. This file only handles:
- HTTP routing
- File uploads
- Error handling
- Response formatting
"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.config import settings
from src.ingestion import ingestor
from src.models import AskRequest, AskResponse, IngestRequest, IngestResponse
from src.pipeline import rag_pipeline
from src.vector_store import vector_store

# ═══════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════

app = FastAPI(
    title="Internal Knowledge Base RAG",
    description="Customer support knowledge base with semantic search",
    version="1.0.0",
)

# Enable CORS (for frontend integration later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════
# HEALTH & STATS
# ═══════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Check system health and vector store status."""
    stats = vector_store.get_collection_stats()
    return {
        "status": "healthy",
        "vector_store": stats,
    }

@app.get("/stats")
async def get_stats():
    """Get system configuration and statistics."""
    return {
        "vector_store": vector_store.get_collection_stats(),
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "llm_model": settings.NVIDIA_LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    }

# ═══════════════════════════════════════════════════
# CORE RAG ENDPOINT
# ═══════════════════════════════════════════════════

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Main RAG endpoint — answer a support question.
    
    - Retrieves relevant chunks from ChromaDB
    - Generates answer with citations
    - Returns confidence score and sources
    """
    try:
        return rag_pipeline.run(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# INGESTION ENDPOINTS
# ═══════════════════════════════════════════════════

@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    department: str | None = None,
    doc_type: str | None = None,
):
    """Upload and ingest a single document.
    
    Supports: .pdf, .md, .txt, .html
    """
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {settings.SUPPORTED_EXTENSIONS}",
        )
    
    # Save uploaded file to temp location
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Build metadata
        metadata = {}
        if department:
            metadata["department"] = department
        if doc_type:
            metadata["doc_type"] = doc_type
        
        # Ingest
        ids = ingestor.ingest_file(tmp_path, metadata=metadata)
        
        return {
            "status": "success",
            "file": file.filename,
            "chunks_ingested": len(ids),
        }
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

@app.post("/ingest/batch")
async def ingest_batch(request: IngestRequest):
    """Batch ingest from a directory path (for cron/scheduled jobs)."""
    try:
        result = ingestor.ingest_folder(request.file_path)
        return {
            "status": "completed",
            "details": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════

@app.delete("/documents/{source_name}")
async def delete_document(source_name: str):
    """Remove all chunks from a specific source file."""
    try:
        vector_store.delete_by_source(source_name)
        return {"status": "deleted", "source": source_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL,
    )