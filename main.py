"""
MedFin-RAG: Retrieval-Augmented Q&A over Healthcare/Finance Documents
----------------------------------------------------------------------
A FastAPI service that lets users upload domain documents (clinical
guidelines, financial reports, policy PDFs) and ask natural-language
questions answered strictly from the retrieved context (RAG pattern).

Pipeline:
  1. Ingest docs -> chunk -> embed (sentence-transformers)
  2. Store vectors in FAISS index
  3. On query -> embed query -> retrieve top-k chunks
  4. Build grounded prompt -> call LLM -> return answer + sources

Run:
  uvicorn app.main:app --reload
"""

import os
import uuid
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import anthropic

app = FastAPI(title="MedFin-RAG", version="1.0.0")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# In-memory store (swap for a persistent vector DB like Chroma/Pinecone in prod)
index = faiss.IndexFlatL2(EMBED_DIM)
chunk_store: List[dict] = []  # {id, text, source}

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a .pdf or .txt document into the knowledge base."""
    os.makedirs("data/uploads", exist_ok=True)
    save_path = f"data/uploads/{uuid.uuid4()}_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    text = extract_text(save_path)
    if not text.strip():
        raise HTTPException(400, "No extractable text found in document.")

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index.add(embeddings.astype("float32"))

    for c in chunks:
        chunk_store.append({"text": c, "source": file.filename})

    return {"status": "ingested", "chunks_added": len(chunks), "total_chunks": len(chunk_store)}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question; answer is grounded only in retrieved document chunks."""
    if index.ntotal == 0:
        raise HTTPException(400, "No documents ingested yet. Call /ingest first.")

    q_emb = embedder.encode([req.question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, min(req.top_k, index.ntotal))

    retrieved = [chunk_store[i] for i in indices[0] if i != -1]
    context = "\n\n---\n\n".join(r["text"] for r in retrieved)
    sources = list({r["source"] for r in retrieved})

    prompt = f"""You are a domain assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information in the provided documents."

Context:
{context}

Question: {req.question}

Answer:"""

    if client.api_key:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text
    else:
        # Fallback: no API key configured, return retrieved context directly
        answer = f"[No LLM key set — showing raw retrieved context]\n\n{context[:800]}"

    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": len(chunk_store)}
