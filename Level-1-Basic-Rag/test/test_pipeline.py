from src.models import AskRequest
from src.pipeline import rag_pipeline

print("=" * 60)
print("TEST 1: High-Confidence Question (password reset)")
print("=" * 60)

req1 = AskRequest(
    question="How do I reset my password?",
    department="Oracle",
    top_k=3,
)

resp1 = rag_pipeline.run(req1)
print(f"Answer: {resp1.answer}")
print(f"Confidence: {resp1.confidence}")
print(f"Sources: {len(resp1.sources)}")
print(f"Latency: {resp1.latency_ms}ms")
for s in resp1.sources:
    print(f"  - {s.metadata['source']} (score: {s.score})")

print("\n" + "=" * 60)
print("TEST 2: Low-Confidence Question (off-topic)")
print("=" * 60)

req2 = AskRequest(
    question="What is the weather in Tokyo today?",
    top_k=3,
)

resp2 = rag_pipeline.run(req2)
print(f"Answer: {resp2.answer}")
print(f"Confidence: {resp2.confidence}")
print(f"Sources: {len(resp2.sources)}")

print("\n" + "=" * 60)
print("TEST 3: Medium-Confidence (billing question)")
print("=" * 60)

req3 = AskRequest(
    question="How do I get a refund?",
    top_k=3,
)

resp3 = rag_pipeline.run(req3)
print(f"Answer: {resp3.answer}")
print(f"Confidence: {resp3.confidence}")
print(f"Sources: {len(resp3.sources)}")
for s in resp3.sources:
    print(f"  - {s.metadata['source']} (score: {s.score})")