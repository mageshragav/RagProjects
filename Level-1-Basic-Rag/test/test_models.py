from src.models import AskRequest, AskResponse, DocumentChunk

# Test 1: Valid request
req = AskRequest(question="How do I reset my password?", department="tech")
print(f"Valid request: {req.question}")
print(f"   Department: {req.department}")
print(f"   Top-K: {req.top_k}")

# Test 2: Invalid request (too short) — should raise error
try:
    bad_req = AskRequest(question="Hi")
except Exception as e:
    print(f"Caught invalid request: {type(e).__name__}")

# Test 3: Invalid top_k — should raise error
try:
    bad_req = AskRequest(question="How do I reset my password?", top_k=25)
except Exception as e:
    print(f"Caught invalid top_k: {type(e).__name__}")

# Test 4: Response model
chunk = DocumentChunk(
    id="abc123",
    text="To reset password, go to Settings...",
    metadata={"source": "guide.pdf", "page": 5},
    score=0.89,
)
resp = AskResponse(
    answer="Go to Settings > Security [1].",
    sources=[chunk],
    confidence="high",
    latency_ms=1245.67,
)
print(f"Response created: {resp.confidence}, {len(resp.sources)} sources")

# Test 5: Auto-strip whitespace
req2 = AskRequest(question="  How do I reset?  ")
print(f"Stripped question: '{req2.question}'")