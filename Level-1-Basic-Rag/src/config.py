# app/config.py
from pydantic_settings import BaseSettings
from environs import Env
from pathlib import Path

env = Env()

env.read_env()

class Settings(BaseSettings):
    # API
    HOST: str = env.str("HOST", "0.0.0.0")
    PORT: int = env.int("PORT", 8000)
    LOG_LEVEL: str = env.str("LOG_LEVEL", "info")

    # OpenAI / Custom LLM
    OPENAI_API_KEY: str = env.str("OPENAI_API_KEY", "")
    NVIDIA_API_KEY: str = env.str("NVIDIA_API_KEY", "")
    OPENAI_BASE_URL: str | None = env.str("OPENAI_BASE_URL", None)
    NVIDIA_API_URL: str | None = env.str("NVIDIA_API_URL", None)

    # Separate embedding credentials; optional, separate from LLM
    EMBEDDING_API_KEY: str | None = env.str("EMBEDDING_API_KEY", None)
    OPENAI_EMBEDDING_BASE_URL: str | None = env.str("OPENAI_EMBEDDING_BASE_URL", None)

    LLM_MODEL: str = env.str("LLM_MODEL", "gpt-4o-mini")
    NVIDIA_LLM_MODEL: str = env.str("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    EMBEDDING_MODEL: str = env.str("EMBEDDING_MODEL", "text-embedding-3-small")
    NVIDIA_EMBEDDING_MODEL: str = env.str("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embed-v1")

    # ChromaDB
    CHROMA_PERSIST_DIR: Path = Path(env.str("CHROMA_PERSIST_DIR", "./data/chroma"))
    CHROMA_COLLECTION: str = env.str("CHROMA_COLLECTION", "knowledge_base")

    # Retrieval
    CHUNK_SIZE: int = env.int("CHUNK_SIZE", 512)
    CHUNK_OVERLAP: int = env.int("CHUNK_OVERLAP", 50)
    TOP_K: int = env.int("TOP_K", 5)
    SIMILARITY_THRESHOLD: float = env.float("SIMILARITY_THRESHOLD", 0.7)

    # Ingestion
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".md", ".txt", ".html")

    @property
    def embedding_api_key(self) -> str:
        """Return the key used for embeddings; use the separate key if provided."""
        return self.EMBEDDING_API_KEY or self.OPENAI_API_KEY

    @property
    def is_custom_llm(self) -> bool:
        """Check if using a custom LLM endpoint."""
        return self.OPENAI_BASE_URL is not None

    @property
    def chroma_path(self) -> str:
        """Return ChromaDB path as string (Chroma expects string)."""
        return str(self.CHROMA_PERSIST_DIR)


# Singleton instance
settings = Settings()