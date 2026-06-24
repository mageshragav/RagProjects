"""Embedding model factory.

Converts text -> vectors (arrays of numbers) that capture semantic meaning.
Uses OpenRouter's embeddings-compatible endpoint directly so the provider's
request shape (nested content blocks) is preserved and the test utilities
keep working.
"""

from __future__ import annotations

from typing import Sequence
from typing import List

import httpx
from src.config import settings

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.embeddings import Embeddings


class OpenRouterEmbeddings(Embeddings):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": [
                {
                    "content": [
                        {"type": "text", "text": text}
                    ]
                }
                for text in texts
            ],
            "encoding_format": "float",
        }

        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        data = response.json()

        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
    
def get_embeddings() -> OpenRouterEmbeddings:
    """Create and return an embedding model instance."""
    return OpenRouterEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.embedding_api_key,
        base_url=settings.OPENAI_BASE_URL or "https://openrouter.ai/api/v1",
    )

def get_nvidia_embeddings() -> NVIDIAEmbeddings:
    
    """Create and return an nvidia embedding model instance."""
    return NVIDIAEmbeddings(
        model=settings.NVIDIA_EMBEDDING_MODEL, 
        api_key=settings.NVIDIA_API_KEY, 
        truncate="NONE", 
    )

_embedding_instance = None


def get_embedding_instance() -> OpenRouterEmbeddings:
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = get_embeddings()
    return _embedding_instance
