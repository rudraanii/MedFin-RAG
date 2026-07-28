# MedFin-RAG — Retrieval-Augmented Q&A for Healthcare & Finance Documents

A production-style RAG (Retrieval-Augmented Generation) service that answers
natural-language questions strictly from uploaded domain documents — clinical
guidelines, insurance policies, financial reports, compliance filings, etc.
Built to eliminate hallucination risk by grounding every answer in retrieved
source text, and to return citations alongside the answer.

## Why this project

Most "AI chatbot" resume projects are just an API wrapper. This one implements
the actual retrieval pipeline that real GenAI products (support bots, clinical
decision support, financial research assistants) are built on:

- Document ingestion & chunking
- Semantic embedding (Sentence-Transformers)
- Vector similarity search (FAISS)
- Context-grounded prompt construction
- LLM-based answer generation with source attribution

## Architecture

```
 Upload (.pdf/.txt)
        │
        ▼
  Text Extraction → Chunking (500 tokens, 50 overlap)
        │
        ▼
  Sentence-Transformer Embeddings (all-MiniLM-L6-v2)
        │
        ▼
      FAISS Index  ◄──────────────┐
        │                         │
        ▼                         │
  Query → Embed → Top-K Retrieve ─┘
        │
        ▼
  Grounded Prompt → Claude API → Answer + Sources
```

## Tech Stack
- **FastAPI** — REST API layer
- **Sentence-Transformers** — embedding model
- **FAISS** — vector similarity search
- **PyPDF** — PDF text extraction
- **Anthropic API** — generation (swap for OpenAI/local LLM easily)

## Setup

```bash
git clone https://github.com/<your-username>/medfin-rag.git
cd medfin-rag
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"   # optional — falls back to raw context retrieval without it
uvicorn app.main:app --reload
```

## Usage

**1. Ingest a document**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_clinical_guideline.pdf"
```

**2. Ask a question**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended dosage for adult patients?", "top_k": 4}'
```

**Response**
```json
{
  "answer": "Based on the provided guideline, the recommended adult dosage is ...",
  "sources": ["sample_clinical_guideline.pdf"]
}
```

## Example Use Cases
- Clinical guideline Q&A for hospital staff
- Insurance policy explainer chatbot
- Financial 10-K / earnings report research assistant
- Internal compliance document search

## Possible Extensions
- Swap FAISS for a persistent vector DB (Pinecone, Weaviate, Chroma)
- Add re-ranking (cross-encoder) for better top-k precision
- Add authentication + per-user document namespaces
- Stream responses via Server-Sent Events

## Evaluation Notes
Retrieval quality was validated by manually checking top-k chunks against
ground-truth answers on a small held-out Q&A set (see `data/eval_qa.json`
if included) — precision@4 was the primary metric tracked.

## License
MIT
