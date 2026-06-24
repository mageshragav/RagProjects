from src.ingestion import ingestor
from src.vector_store import vector_store

print("=" * 50)
print("TEST 1: Ingest Single Markdown File")
print("=" * 50)

ids = ingestor.ingest_file(
    file_path="data/raw/Reset Password_Oracle.pdf",
    metadata={"department": "tech", "doc_type": "guide"}
)

print(f"Ingested {len(ids)} chunks")
print(f"   IDs: {ids[:3]}...")

print("\n" + "=" * 50)
print("TEST 2: Search What We Just Ingested")
print("=" * 50)

results = vector_store.similarity_search(
    query="How do I reset my password?",
    k=3,
    score_threshold=0.0,
)

print(f"Found {len(results)} results:")
for i, (doc, score) in enumerate(results, 1):
    print(f"\n  [{i}] Score: {score:.4f}")
    print(f"      Source: {doc.metadata['source']}")
    print(f"      Department: {doc.metadata.get('department')}")
    print(f"      Chunk: {doc.metadata['chunk_index'] + 1}/{doc.metadata['total_chunks']}")
    print(f"      Text: {doc.page_content[:100]}...")

print("\n" + "=" * 50)
print("TEST 3: Filtered Search (Oracle Only)")
print("=" * 50)

billing_results = vector_store.similarity_search(
    query="How do I reset password?",
    k=3,
    filters={"department": "Oracle"},  # Should still find it (same file)
    score_threshold=0.0,
)

print(f"Found {len(billing_results)} results:")
for i, (doc, score) in enumerate(billing_results, 1):
    print(f"  [{i}] {doc.page_content[:80]}...")

print("\n" + "=" * 50)
print("TEST 4: Stats")
print("=" * 50)
print(vector_store.get_collection_stats())