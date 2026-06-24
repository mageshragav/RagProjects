from src.llm import get_llm, build_messages

print("=" * 50)
print("TEST 1: Create LLM Instance")
print("=" * 50)

llm = get_llm(temperature=0.1)
print(f" LLM created: {llm.model}")
print(f"   Temperature: {llm.temperature}")

print("\n" + "=" * 50)
print("TEST 2: Build Messages")
print("=" * 50)

context = """[1] Source: user_guide.pdf
To reset your password, go to Settings > Security and click 'Reset Password'.

[2] Source: admin_guide.pdf
If your account is locked, contact your administrator."""

messages = build_messages(
    question="How do I reset my password?",
    context=context,
)

print(f" Messages built: {len(messages)} messages")
print(f"\n--- System Message ---\n{messages[0].content[:200]}...")
print(f"\n--- Human Message ---\n{messages[1].content}")

print("\n" + "=" * 50)
print("TEST 3: Generate Answer (costs ~$0.001)")
print("=" * 50)

response = llm.invoke(messages)
print(f"\n Answer:\n{response.content}")

print("\n" + "=" * 50)
print("TEST 4: Low-Confidence Scenario")
print("=" * 50)

bad_context = """[1] Source: billing_policy.pdf
Refunds are processed within 5-7 business days."""

bad_messages = build_messages(
    question="How do I reset my password?",
    context=bad_context,
)

response2 = llm.invoke(bad_messages)
print(f"\n Answer (should refuse or hedge):\n{response2.content}")