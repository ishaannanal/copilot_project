from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    IngestResponse,
    IngestTextBody,
    QueryRequest,
    QueryResponse,
)
from app.services.faiss_store import FaissStore, read_uploaded_text
from app.services.rag import RAGService

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(
    title="AI Research Copilot",
    description="RAG over technical documents with FAISS + LLM APIs",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = FaissStore(settings.data_dir)
rag = RAGService(store)

if FRONTEND_DIR.is_dir():
    app.mount(
        "/ui",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="ui",
    )


@app.get("/", include_in_schema=False)
async def root():
    if FRONTEND_DIR.is_dir():
        return RedirectResponse(url="/ui/")
    return {"message": "AI Research Copilot API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        indexed_chunks=store.chunk_count,
        has_api_key=bool(settings.openai_api_key),
    )


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(body: IngestTextBody):
    try:
        doc_id, n = await store.add_document(body.title, body.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return IngestResponse(
        document_id=doc_id,
        chunks_indexed=n,
        message=f"Indexed {n} chunk(s).",
    )


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        text = read_uploaded_text(file.filename or "", raw)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not read file: {e}"
        ) from e
    if not text.strip():
        raise HTTPException(
            status_code=400, detail="No extractable text in file."
        )
    doc_title = title or (file.filename or "uploaded")
    try:
        doc_id, n = await store.add_document(doc_title, text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return IngestResponse(
        document_id=doc_id,
        chunks_indexed=n,
        message=f"Indexed {n} chunk(s) from file.",
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if store.chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Ingest text or a file first.",
        )
    try:
        return await rag.answer(req.question, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if store.chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Ingest text or a file first.",
        )
    try:
        return await rag.analyze(req.question, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model output could not be parsed: {e}",
        ) from e


@app.get("/corpus", include_in_schema=True)
async def corpus_info():
    return store.export_state()
