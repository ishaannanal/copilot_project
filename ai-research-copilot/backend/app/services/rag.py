from typing import List, Optional

from app.config import settings
from app.prompts import (
    RAG_ANSWER_SYSTEM,
    RAG_ANSWER_USER,
    STRUCTURED_ANALYSIS_SYSTEM,
    STRUCTURED_ANALYSIS_USER,
)
from app.schemas import (
    AnalyzeResponse,
    QueryResponse,
    SourceChunk,
    parse_structured_analysis,
)
from app.services.embeddings import embed_texts
from app.services.faiss_store import FaissStore
from app.services.llm import chat_completion_json, chat_completion_text


def _format_context(chunks: List[dict]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("title", "")
        parts.append(
            f"[{i}] (doc {c['document_id'][:8]}…, {title})\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


class RAGService:
    def __init__(self, store: FaissStore) -> None:
        self.store = store

    async def answer(self, question: str, top_k: Optional[int]) -> QueryResponse:
        k = top_k or settings.retrieval_top_k
        qvec = (await embed_texts([question]))[0]
        hits = self.store.search(qvec, k)
        indices = [h[0] for h in hits]
        scores = [h[1] for h in hits]
        chunks = self.store.get_chunks(indices)
        context = _format_context(chunks)
        user = RAG_ANSWER_USER.format(context=context, question=question)
        answer = await chat_completion_text(RAG_ANSWER_SYSTEM, user)
        sources: List[SourceChunk] = []
        for ch, sc in zip(chunks, scores):
            preview = ch["text"][:280] + ("…" if len(ch["text"]) > 280 else "")
            sources.append(
                SourceChunk(
                    document_id=ch["document_id"],
                    chunk_index=ch["chunk_index"],
                    score=sc,
                    text_preview=preview,
                )
            )
        return QueryResponse(answer=answer, sources=sources)

    async def analyze(
        self, question: Optional[str], top_k: Optional[int]
    ) -> AnalyzeResponse:
        k = top_k or settings.retrieval_top_k
        if question and question.strip():
            qvec = (await embed_texts([question.strip()]))[0]
            hits = self.store.search(qvec, k)
            task = (
                f"The user asked: {question.strip()}\n"
                "Extract key concepts, relationships, and a summary "
                "grounded in the passages."
            )
        else:
            generic = (
                "Key technical concepts, definitions, architecture, "
                "and relationships in this documentation."
            )
            qvec = (await embed_texts([generic]))[0]
            hits = self.store.search(qvec, k)
            task = (
                "Produce an overview: key concepts, how they relate, "
                "and a short summary grounded in the passages."
            )

        indices = [h[0] for h in hits]
        scores = [h[1] for h in hits]
        chunks = self.store.get_chunks(indices)
        context = _format_context(chunks)
        user = STRUCTURED_ANALYSIS_USER.format(
            context=context, task_instruction=task
        )
        raw = await chat_completion_json(STRUCTURED_ANALYSIS_SYSTEM, user)
        analysis = parse_structured_analysis(raw)
        sources = [
            SourceChunk(
                document_id=ch["document_id"],
                chunk_index=ch["chunk_index"],
                score=sc,
                text_preview=(ch["text"][:280] + "…")
                if len(ch["text"]) > 280
                else ch["text"],
            )
            for ch, sc in zip(chunks, scores)
        ]
        return AnalyzeResponse(analysis=analysis, sources=sources)
