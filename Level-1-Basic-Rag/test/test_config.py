from src.config import settings

print(f"Host: {settings.HOST}")
print(f"Port: {settings.PORT}")
print(f"LLM Model: {settings.LLM_MODEL}")
print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
print(f"Chroma Dir: {settings.CHROMA_PERSIST_DIR}")
print(f"Chunk Size: {settings.CHUNK_SIZE}")
print(f"Custom LLM? {settings.is_custom_llm}")
print(f"Chroma Path: {settings.chroma_path}")