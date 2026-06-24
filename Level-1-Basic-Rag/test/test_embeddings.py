from src.embeddings import get_embeddings
# Create embedding model
embedder = get_embeddings()


# Test 1: Embed a single text
text = "How do I reset my password?"
vector = embedder.embed_query(text=text)

print(f"Input text: '{text}'")
print(f"Vector length: {len(vector)} dimensions")
print(f"First 5 numbers: {vector[:5]}")
print(f"Last 5 numbers: {vector[-5:]}")


# Test 2: Embed multiple texts (batch — faster for ingestion)
texts = [
    "How do I reset my password?",
    "What is the refund policy?",
    "How to contact billing department?",
]
vectors = embedder.embed_documents(texts)

print(f"\n✅ Batch embedded {len(vectors)} texts")
for i, v in enumerate(vectors):
    print(f"   Text {i+1}: {len(v)} dims — '{texts[i][:30]}...'")
