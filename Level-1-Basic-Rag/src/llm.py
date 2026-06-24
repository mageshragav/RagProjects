"""LLM client for generating answers.

Supports OpenAI and any OpenAI-compatible endpoint (vLLM, Ollama, etc.).
Enforces strict prompting rules to prevent hallucination.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA


from src.config import settings


def get_llm(temperature: float = 0.1) -> ChatNVIDIA:
    """Create and return an LLM instance.
    
    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
            We use 0.1 for factual Q&A (consistent, grounded).
    
    Returns:
        Configured ChatNVIDIA instance
    """
    kwargs = {
        "model": settings.NVIDIA_LLM_MODEL,
        "temperature": temperature,
        "api_key": settings.NVIDIA_API_KEY
    }
    
    # Support custom LLM endpoints
    # if settings.NVIDIA_API_URL:
    #     kwargs["base_url"] = settings.NVIDIA_API_URL
    
    return ChatNVIDIA(**kwargs)


# ═══════════════════════════════════════════════════
# PROMPT ENGINEERING
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise customer support assistant.

RULES — FOLLOW STRICTLY:
1. Answer ONLY using the provided context. Do not use outside knowledge.
2. Cite sources using [1], [2], etc. after each fact.
3. If the context doesn't contain the answer, say exactly:
   "I don't have enough information to answer this confidently. Here are related articles: [list sources]"
4. Be concise. Max 3-4 sentences for simple questions.
5. For technical steps, use numbered lists.

CONTEXT:
{context}
"""


def build_messages(question: str, context: str) -> list[SystemMessage | HumanMessage]:
    """Build the message list for the LLM.
    
    Args:
        question: The user's question
        context: Retrieved chunks formatted with citations
    
    Returns:
        List of messages for LangChain
    """
    return [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=f"Question: {question}"),
    ]