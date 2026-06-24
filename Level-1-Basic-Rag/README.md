# Internal Knowledge Base Chatbot for Customer Support

A Basic RAG (Retrieval-Augmented Generation) system that centralizes scattered documentation (PDFs, Confluence, Zendesk, Slack) to instantly answer customer support tickets.

## What is RAG?

Retrieval-Augmented Generation (RAG) combines:
1. **Retrieval** - Finding relevant text chunks from a knowledge base
2. **Generation** - Using an LLM to synthesize an answer with the retrieved context

```
User Query → Embedding → Vector Search → LLM → Answer
```

## Project Structure

```
src/
├── api.py             # FastAPI endpoints
├── config.py          # Configuration and environment variables
├── embeddings.py      # OpenRouter/OpenAI embeddings client
├── ingestion.py       # Document loading and chunking pipeline
├── llm.py             # LLM answer generation (using OpenRouter)
├── models.py          # Pydantic request/response schemas
├── pipeline.py        # Main RAG pipeline orchestration
└── vector_store.py    # ChromaDB wrapper for document storage

test/
├── test_config.py     # Configuration validation
├── test_embeddings.py # Embedding functionality
├── test_ingestion.py  # Document ingestion tests
├── test_llm.py        # LLM generation tests
├ -- test_pipeline.py  # Pipeline integration tests
└── test_vector_store.py # Vector store operations
```

## Quick Start

### Prerequisites

- Python 3.13+
- uv (recommended) or pip

### Installation

```bash
uv sync  # Install all dependencies
```

### Configuration

Copy the example environment file (if exists) or create a `.env` file with:

```bash
# If .env.example exists:
cp .env.example .env

# Otherwise create .env with:
cat > .env << EOF
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
EOF
```

**Required variables:**
- `OPENAI_API_KEY` - Your OpenRouter API key (starts with `sk-or-v1-`)
- `OPENAI_BASE_URL` - Set to `https://openrouter.ai/api/v1` for OpenRouter

**Optional embedding-specific overrides:**
- `EMBEDDING_API_KEY` - Separate key for embeddings (if different from LLM)
- `OPENAI_EMBEDDING_BASE_URL` - Separate endpoint for embeddings
- `EMBEDDING_MODEL` - Default: `nvidia/llama-nemotron-embed-vl-1b-v2:free`

### Running the Application

Start the FastAPI server:

```bash
uv run python src/api.py
```

The API will be available at `http://localhost:8000`.

## Usage

### Document Ingestion

```python
from src.ingestion import ingestor

ids = ingestor.ingest_file(
    file_path="data/raw/document.pdf",
    metadata={"department": "tech", "doc_type": "guide"}
)
```

### Vector Search

```python
from src.vector_store import vector_store

results = vector_store.similarity_search(
    query="How do I reset my password?",
    k=5,
    score_threshold=0.5
)
```

### Generate Answer

```python
from src.pipeline import rag_pipeline

answer = rag_llm.generate_answer(
    query="How do I reset my password?",
    context_results=results
)
```

Or use the pipeline directly:

```python
from src.pipeline import rag_pipeline

result = rag_pipeline.run(
    query="How do I reset my password?",
    top_k=5,
    score_threshold=0.5
)
print(result.answer)
```

## API Endpoints

The FastAPI server provides:

- `POST /ingest` - Upload and process documents
- `POST /query` - Ask a question and get an answer
- `GET /health` - Health check
- `GET /stats` - Vector store statistics

## Technology Stack

| Component      | Choice                                     | Why                                      |
|----------------|--------------------------------------------|------------------------------------------|
| Orchestration  | Custom pipeline                            | Learning-focused, explicit control       |
| Embeddings     | OpenRouter (`nvidia/llama-nemotron-embed-vl-1b-v2`) | Free, high-quality embeddings          |
| Vector DB      | ChromaDB                                   | Easy setup, persistent, good for learning|
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` | Proven text splitting                    |
| Configuration  | Pydantic Settings, environs                | Type-safe, environment-based config      |
| LLM            | OpenRouter (various models)                | Access to multiple LLMs via one API      |
| API Layer      | FastAPI                                    | High performance, async, automatic docs  |

## Roadmap

Based on the learning roadmap:

1. **Foundations & Setup** ✓
   - Environment setup, basic project structure
2. **Document Ingestion Pipeline** ✓
   - PDF, Markdown, TXT, HTML support
   - Chunking with metadata
3. **Vector Search & Embeddings** ✓
   - ChromaDB integration, similarity search
4. **RAG Chain & LLM Integration** ✓
   - End-to-end question answering
5. **FastAPI Layer** ✓
   - RESTful API for ingestion and querying
6. **Database Layer** (Planned)
   - PostgreSQL + Tortoise ORM for metadata and user management
7. **Production Features** (Planned)
   - Logging, monitoring, Docker deployment, authentication

## Running Tests

Set the `PYTHONPATH` to include the project root:

```bash
export PYTHONPATH=/path/to/Level-1-Basic-Rag
```

Then run individual test files:

```bash
uv run python test/test_config.py
uv run python test/test_embeddings.py
uv run python test/test_ingestion.py
uv run python test/test_llm.py
uv run python test/test_pipeline.py
uv run python test/test_vector_store.py
```

Or run all tests with pytest (if installed):

```bash
uv run pytest test/
```

## License

MIT