import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import faiss
import numpy as np

from app.config import settings
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


class FaissStore:
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir or settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.data_dir / "vectors.faiss"
        self._meta_path = self.data_dir / "metadata.pkl"
        self.dimension: int = 0
        self._index: Optional[faiss.Index] = None
        self._meta: List[dict] = []
        self._load()

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self._meta = pickle.load(f)
            self.dimension = self._index.d

    def _save(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._meta, f)

    @property
    def chunk_count(self) -> int:
        return len(self._meta)

    async def add_document(self, title: str, text: str) -> Tuple[str, int]:
        doc_id = str(uuid4())
        chunks = chunk_text(
            text, settings.chunk_size, settings.chunk_overlap
        )
        if not chunks:
            return doc_id, 0
        vectors = await embed_texts(chunks)
        dim = len(vectors[0])
        arr = np.array(vectors, dtype=np.float32)
        arr = _l2_normalize(arr)

        if self._index is None:
            self.dimension = dim
            self._index = faiss.IndexFlatIP(dim)
            self._meta = []

        if dim != self.dimension:
            raise ValueError(
                f"Embedding dim {dim} != index dim {self.dimension}"
            )

        self._index.add(arr)
        for i, ch in enumerate(chunks):
            self._meta.append(
                {
                    "document_id": doc_id,
                    "title": title,
                    "chunk_index": i,
                    "text": ch,
                }
            )
        self._save()
        return doc_id, len(chunks)

    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[int, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = np.array([query_vector], dtype=np.float32)
        q = _l2_normalize(q)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        out: List[Tuple[int, float]] = []
        for idx, sc in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            out.append((int(idx), float(sc)))
        return out

    def get_chunks(self, indices: List[int]) -> List[dict]:
        return [self._meta[i] for i in indices if 0 <= i < len(self._meta)]

    def export_state(self) -> Dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "dimension": self.dimension,
            "documents": self._documents_summary(),
        }

    def _documents_summary(self) -> List[dict]:
        seen: Dict[str, dict] = {}
        for m in self._meta:
            did = m["document_id"]
            if did not in seen:
                seen[did] = {
                    "document_id": did,
                    "title": m.get("title", ""),
                    "chunks": 0,
                }
            seen[did]["chunks"] += 1
        return list(seen.values())


def read_uploaded_text(filename: str, raw: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(raw))
        parts: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts)
    return raw.decode("utf-8", errors="replace")
