# AI Research Copilot

Retrieval-augmented generation (RAG) stack for querying and analyzing technical documents: **FAISS** vector search, **OpenAI-compatible** embeddings and chat APIs, **REST** endpoints, and a **lightweight web UI**.

## What you get

| Piece | Role |
|--------|------|
| Ingest | Paste text or upload `.txt` / `.md` / `.pdf` → chunk → embed → add to FAISS |
| `/query` | Retrieve top-k chunks, then LLM answer grounded in context |
| `/analyze` | Same retrieval + JSON structured output: concepts, relationships, summary |
| Frontend | Static page at `/ui/` calling the API on the same origin |

## Quick start

1. **Python 3.9+** (3.10+ recommended).

2. Create a virtualenv and install dependencies:

```bash
cd ai-research-copilot/backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure the API key (OpenAI or any OpenAI-compatible base URL):

```bash
cp ../.env.example ../.env
# Edit ../.env — set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, models)
```

4. Run the server **from the `backend` directory** (so `data/` and `.env` resolve predictably):

```bash
cd ai-research-copilot/backend
cp ../.env .env   # optional: keep one .env next to backend for pydantic-settings
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If you prefer a single `.env` at the repo root, either copy it into `backend/` or export variables before starting uvicorn.

5. Open **http://127.0.0.1:8000/ui/** — ingest `samples/example-spec.md`, then try **RAG answer** and **Structured analysis**.

API docs: **http://127.0.0.1:8000/docs**

## Deployment notes

- Point `OPENAI_BASE_URL` at your provider (OpenAI, Azure OpenAI proxy, etc.).
- Persist the `data/` directory (FAISS index + metadata) for a stable corpus across restarts.
- Put the app behind HTTPS and add auth if exposed beyond a trusted network.

## Project layout

```
ai-research-copilot/
  backend/
    app/
      main.py           # FastAPI routes + static UI mount
      config.py
      prompts.py
      schemas.py
      services/         # chunking, embeddings, FAISS, RAG, LLM
    requirements.txt
  frontend/             # Served at /ui/
  samples/              # Example markdown for testing
```

Development History

This project was originally built and iterated on locally before being published to GitHub. Earlier versions were not tracked in this repository, so the commit history here represents the point at which the project was prepared for public release.

Ongoing development and improvements are now tracked through GitHub commits.

## License

Use and modify freely for learning and portfolio work.
