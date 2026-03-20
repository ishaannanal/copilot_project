from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestTextBody(BaseModel):
    title: str = Field(default="untitled", max_length=256)
    text: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class AnalyzeRequest(BaseModel):
    question: Optional[str] = Field(
        default=None,
        max_length=4000,
        description="Optional focus question; defaults to broad document analysis.",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Relationship(BaseModel):
    from_concept: str
    to_concept: str
    relation: str


class StructuredAnalysis(BaseModel):
    key_concepts: List[str]
    relationships: List[Relationship]
    summary: str


class AnalyzeResponse(BaseModel):
    analysis: StructuredAnalysis
    sources: List[SourceChunk]
    raw_context_used: bool = True


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    has_api_key: bool


class ErrorDetail(BaseModel):
    detail: str


def parse_structured_analysis(data: Dict[str, Any]) -> StructuredAnalysis:
    concepts = data.get("key_concepts") or []
    if not isinstance(concepts, list):
        concepts = []
    concepts = [str(c).strip() for c in concepts if str(c).strip()]

    rels_raw = data.get("relationships") or []
    relationships: List[Relationship] = []
    if isinstance(rels_raw, list):
        for r in rels_raw:
            if not isinstance(r, dict):
                continue
            fc = str(r.get("from_concept", "")).strip()
            tc = str(r.get("to_concept", "")).strip()
            rel = str(r.get("relation", "")).strip()
            if fc and tc and rel:
                relationships.append(
                    Relationship(from_concept=fc, to_concept=tc, relation=rel)
                )

    summary = str(data.get("summary", "")).strip()
    if not summary:
        summary = "No summary produced."

    return StructuredAnalysis(
        key_concepts=concepts[:50],
        relationships=relationships[:100],
        summary=summary,
    )
