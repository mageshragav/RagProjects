"""Complete RAG pipeline: retrieval + generation + confidence scoring.

This is the 'brain' — everything comes together here.
"""

import time

from langchain_core.documents import Document

from src.config import settings
from src.llm import build_messages, get_llm
from src.models import AskRequest, AskResponse, DocumentChunk
from src.vector_store import vector_store



class RAGPipeline:
    """Production RAG with confidence scoring and fallback responses."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.min_confidence_threshold = 0.7

    def _calculate_confidence(
        self,
        results: list[tuple[Document, float]],
        query: str,
    ) -> str:
        """Score confidence based on retrieval quality.
        
        Returns: 'high' | 'medium' | 'low'
        """
        if not results:
            return "low"
        
        scores = [score for _, score in results]
        avg_score = sum(scores) / len(scores)
        top_score = scores[0]
        
        # High: top score > 0.85 AND average > 0.75
        if top_score > 0.85 and avg_score > 0.75:
            return "high"
        
        # Medium: top score > 0.70
        elif top_score > 0.70:
            return "medium"
        
        # Low: everything else
        else:
            return "low"
    
    def _fallback_response(self, results: list[tuple[Document, float]]) -> str:
        """Safe response when confidence is low.
        
        Never hallucinate — always redirect to available docs.
        """
        if not results:
            return (
                "I don't have enough information to answer this confidently. "
                "Please contact a senior agent or check the knowledge base directly."
            )
        
        # List top 3 related sources
        sources_list = "\n".join([
            f"- {doc.metadata.get('source', 'Unknown')}"
            for doc, _ in results[:3]
        ])
        
        return (
            "I'm not confident about the exact answer. "
            f"Here are {len(results)} related articles that might help:\n{sources_list}"
        )
    
    def run(self, request: AskRequest) -> AskResponse:
        """Execute full RAG pipeline.
        
        1. Retrieve relevant chunks
        2. Score confidence
        3. Generate answer (or fallback)
        4. Return structured response
        """
        start_time = time.time()
        # Step 1: Build metadata filters from request
        filters = {}
        if request.department:
            filters["department"] = request.department
        if request.doc_type:
            filters["doc_type"] = request.doc_type
        
        # Step 2: Retrieve documents
        results = vector_store.similarity_search(
            query=request.question,
            k=request.top_k,
            filters=filters if filters else None,
            score_threshold=settings.SIMILARITY_THRESHOLD,
        )
        # Step 3: Calculate confidence
        confidence = self._calculate_confidence(results, request.question)

        # Step 4: Build context with citations
        if results:
            context = "\n\n".join([
                f"[{i+1}] Source: {doc.metadata.get('source', 'Unknown')}, "
                f"Chunk {doc.metadata.get('chunk_index', 0)}/{doc.metadata.get('total_chunks', 1)}\n"
                f"{doc.page_content}"
                for i, (doc, _) in enumerate(results)
            ])
        else:
            context = "No relevant documents found."
        print(context)
        # Step 5: Generate answer or fallback
        if confidence == "low" or not results:
            answer = self._fallback_response(results)
        else:
            messages = build_messages(request.question, context)
            response = self.llm.invoke(messages)
            answer = response.content

        # Step 6: Format sources for response
        sources = [
            DocumentChunk(
                id=doc.metadata.get("file_hash", "unknown"),
                text=doc.page_content[:300] + "...",  # Truncate for response
                metadata={
                    "source": doc.metadata.get("source"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "score": round(score, 3),
                },
                score=round(score, 3),
            )
            for doc, score in results
        ]

        # Step 7: Calculate latency
        latency = (time.time() - start_time) * 1000
        
        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            latency_ms=round(latency, 2),
        )
    
rag_pipeline = RAGPipeline()