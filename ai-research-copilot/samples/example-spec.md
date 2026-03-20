# Example: Retrieval-Augmented Generation

## Overview
RAG combines dense retrieval over a document corpus with a large language model. The retriever returns passages likely relevant to the user query; the model conditions its answer on those passages.

## Key components
- **Chunking**: Long documents are split into overlapping segments for embedding.
- **Vector index**: Embeddings are stored in FAISS for fast similarity search.
- **Prompting**: System and user prompts constrain the model to cite and respect context.

## Relationships
- Chunking **enables** accurate embedding of long PDFs and markdown.
- FAISS **supports** low-latency nearest-neighbor search at query time.
- The LLM **synthesizes** an answer **from** retrieved chunks, reducing hallucination when context is sufficient.
