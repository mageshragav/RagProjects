"""Pydantic models for API contracts and data shapes.

These define the 'language' our system speaks.
Every request, response, and internal data structure is declared here.
"""
from pydantic import BaseModel, Field, field_validator

# ═══════════════════════════════════════════════════════════════
# REQUEST MODELS (what the user sends us)
# ═══════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    """POST /ask request body.
    
    The support agent sends this when they want an answer.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The support question to answer",
        examples=["How do I reset my password?"],
    )
    
    department: str | None = Field(
        default=None,
        description="Filter results by department (e.g., 'billing', 'tech')",
        examples=["tech"],
    )
    
    doc_type: str | None = Field(
        default=None,
        description="Filter by document type (e.g., 'policy', 'guide')",
        examples=["guide"],
    )
    
    top_k: int = Field(
        default=5,
        ge=1,  # greater than or equal to 1
        le=20,  # less than or equal to 20
        description="Number of document chunks to retrieve",
    )
    
    # Custom validator: strip whitespace from question
    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()

class IngestRequest(BaseModel):
    """POST /ingest/batch request body.
    
    Used by cron jobs or admins to bulk-ingest a directory.
    """
    file_path: str = Field(
        ...,
        description="Absolute or relative path to file or directory",
        examples=["./data/raw/policies"],
    )
    
    metadata: dict = Field(
        default_factory=dict,
        description="Extra metadata to attach to all chunks",
        examples=[{"department": "billing", "doc_type": "policy"}],
    )

# ═══════════════════════════════════════════════════════════════
# RESPONSE MODELS (what we send back to the user)
# ═══════════════════════════════════════════════════════════════

class DocumentChunk(BaseModel):
    """A single retrieved chunk with its source and score."""
    id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="The chunk content (truncated in responses)")
    metadata: dict = Field(description="Source file, page, department, etc.")
    score: float | None = Field(
        default=None,
        description="Similarity score (0.0 to 1.0, higher is better)",
    )

class AskResponse(BaseModel):
    """POST /ask response body.
    
    Everything the agent needs: answer, sources, confidence, timing.
    """
    answer: str = Field(description="The generated answer with citations")
    
    sources: list[DocumentChunk] = Field(
        description="The chunks used to generate the answer",
    )
    
    confidence: str = Field(
        description="How confident we are: 'high', 'medium', or 'low'",
        pattern="^(high|medium|low)$",  # Regex validation
    )
    
    latency_ms: float = Field(
        description="Total request time in milliseconds",
    )

class IngestResponse(BaseModel):
    """Response after ingesting documents."""
    status: str
    file: str | None = None
    chunks_ingested: int | None = None
    details: dict | None = None

# ═══════════════════════════════════════════════════════════════
# INTERNAL MODELS (used inside the pipeline, not exposed via API)
# ═══════════════════════════════════════════════════════════════

class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk in ChromaDB.
    
    This is what we filter on during retrieval.
    """
    source: str          # filename, e.g., "user_guide.pdf"
    file_path: str       # full path
    file_type: str       # ".pdf", ".md", etc.
    file_hash: str       # md5 prefix for deduplication
    department: str | None = None
    doc_type: str | None = None
    chunk_index: int     # which chunk this is (0, 1, 2...)
    total_chunks: int    # total chunks from this file