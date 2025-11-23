# RAG Microservice with FastAPI, ChromaDB, SentenceTransformers, and Grok (xAI)

A production-style Retrieval-Augmented Generation (RAG) microservice that answers questions using your own documents.  
Built with FastAPI, ChromaDB, SentenceTransformers, Docker, and Grok from xAI.  
Supports TXT/PDF ingestion, persistent vector search, and LLM-powered reasoning.

---

## Features

- Chunk and index documents (TXT + PDF)
- Generate embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)
- Store and query vectors with ChromaDB (persistent)
- Retrieve the most relevant chunks for any query
- Build grounded RAG prompts to avoid hallucination
- Generate accurate answers using Grok (xAI)
- Serve everything through a FastAPI `/ask` endpoint
- Containerized with Docker and Docker Compose

---

## Tech Stack

- **FastAPI**
- **SentenceTransformers**
- **ChromaDB**
- **Grok (xAI)**
- **pypdf**
- **Docker / Docker Compose**
- **Uvicorn**
- **Python**

---

## 📁 Project Structure

```
rag-llm-fastapi-microservice/
│
├── data/
│   └── raw/
│
├── rag/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectordb.py
│   ├── retriever.py
│   └── generator.py
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── scripts/
│   └── ingest.py
│
├── chroma_db/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Architecture Overview

This project is split into two main flows:

1. **Ingestion pipeline** – prepares your documents for retrieval  
2. **Query pipeline** – answers user questions using Retrieval-Augmented Generation (RAG)

```mermaid
flowchart LR
    classDef comp fill=#0f172a,stroke=#0f172a,color=#e5e7eb,stroke-width=1,rx=6,ry=6
    classDef data fill=#0369a1,stroke=#0f172a,color=#e5e7eb,stroke-width=1,rx=6,ry=6
    classDef ext fill=#15803d,stroke=#0f172a,color=#e5e7eb,stroke-width=1,rx=6,ry=6

    subgraph Ingestion["Ingestion Pipeline (offline)"]
        D[data/raw (.txt / .pdf)]:::data
        S[scripts/ingest.py]:::comp
        C[rag/chunker.py\nsentence-based chunking]:::comp
        E[rag/embedder.py\nSentenceTransformers]:::comp
        V[rag/vectordb.py\nChromaDB (persistent)]:::comp

        D --> S --> C --> E --> V
    end

    subgraph Query["Query Pipeline (online API)"]
        U[(User / Client)]:::ext
        A[app/main.py\nFastAPI /ask]:::comp
        R[rag/retriever.py\nEmbeds query + vector search]:::comp
        G[rag/generator.py\nGrok (xAI) LLM]:::comp
        V2[(ChromaDB\nchroma_db/)]:::data

        U --> A --> R --> V2
        R --> G --> A
        A --> U
    end

    V -. writes embeddings .-> V2
```

---

## Ingest Documents

Add your `.txt` or `.pdf` files to:

```
data/raw/
```

Then run:

```
python scripts/ingest.py
```

This extracts text, chunks it, embeds it, and stores it in ChromaDB.

---

## Run the API Locally

Start FastAPI:

```
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

Example query:

```json
{
  "query": "How long are loyalty points valid?"
}
```

---

## Running with Docker

### Build:

```
docker build -t rag-llm-fastapi-microservice .
```

### Run:

```
docker run -p 8000:8000 -e GROK_API_KEY="your-key" rag-llm-fastapi-microservice
```

---

## Running with Docker Compose

### 1. Create `.env`

```
GROK_API_KEY=your-key
```

### 2. Start:

```
docker compose up --build
```

Access:

```
http://localhost:8000/docs
```

---

## Requirements

```
pip install -r requirements.txt
export GROK_API_KEY="your-key"
```

---

## License

MIT License
