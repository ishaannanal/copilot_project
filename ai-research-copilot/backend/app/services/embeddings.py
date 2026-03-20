from typing import List

import httpx

from app.config import settings


async def embed_texts(texts: List[str]) -> List[List[float]]:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env to enable embeddings."
        )
    url = f"{settings.openai_base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.embedding_model, "input": texts}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    items = data.get("data") or []
    by_idx = {int(x["index"]): x["embedding"] for x in items if "embedding" in x}
    return [by_idx[i] for i in range(len(texts))]
