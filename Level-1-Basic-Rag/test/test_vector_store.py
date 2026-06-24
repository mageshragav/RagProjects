from langchain_core.documents import Document
from src.vector_store import vector_store

print("=" * 50)
print("TEST 1: Collection Stats")
print("=" * 50)
stats = vector_store.get_collection_stats()
print(f"Collection: {stats['collection_name']}")
print(f"Documents: {stats['document_count']}")
print(f"Location: {stats['persist_directory']}")

print("\n" + "=" * 50)
print("TEST 2: Add Documents")
print("=" * 50)

# Create sample chunks
docs = [
    Document(
        page_content="To reset your password, go to Settings > Security and click 'Reset Password'.",
        metadata={"source": "user_guide.pdf", "department": "tech", "chunk_index": 0}
    ),
    Document(
        page_content="For billing issues, contact billing@company.com or call 1-800-123-4567.",
        metadata={"source": "billing_policy.pdf", "department": "billing", "chunk_index": 0}
    ),
    Document(
        page_content="Password reset requires admin approval if account is locked.",
        metadata={"source": "admin_guide.pdf", "department": "tech", "chunk_index": 0}
    ),
]

ids = vector_store.add_documents(docs)
print(f"Added {len(ids)} chunks with IDs: {ids[:2]}...")

print("\n" + "=" * 50)
print("TEST 3: Search Without Filter")
print("=" * 50)

results = vector_store.similarity_search(
    query="How do I reset my password?",
    k=3,
    score_threshold=0.0,  # Show everything for demo
)

print(f"Found {len(results)} results:")
for i, (doc, score) in enumerate(results, 1):
    print(f"  [{i}] Score: {score:.4f} | Source: {doc.metadata['source']}")
    print(f"      Text: {doc.page_content[:60]}...")

print("\n" + "=" * 50)
print("TEST 4: Search With Department Filter")
print("=" * 50)

filtered_results = vector_store.similarity_search(
    query="How do I reset my password?",
    k=3,
    filters={"department": "tech"},
    score_threshold=0.0,
)

print(f"Found {len(filtered_results)} results (filtered to 'tech'):")
for i, (doc, score) in enumerate(filtered_results, 1):
    print(f"  [{i}] Score: {score:.4f} | Source: {doc.metadata['source']}")

print("\n" + "=" * 50)
print("TEST 5: Search With No Match")
print("=" * 50)

no_match = vector_store.similarity_search(
    query="What is the weather in Tokyo?",
    k=3,
    score_threshold=0.7,  # High threshold — should find nothing
)

print(f"Found {len(no_match)} results (should be 0 or very few)")

print("\n" + "=" * 50)
print("TEST 6: Stats After Insert")
print("=" * 50)
stats = vector_store.get_collection_stats()
print(f"Total documents now: {stats['document_count']}")