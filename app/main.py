# app/main.py

from fastapi import FastAPI
from .schemas import AskRequest, AskResponse
from rag.retriever import Retriever
from rag.generator import Generator

app = FastAPI(
    title="Document RAG Chatbot API",
    description="Ask questions over your documents using embeddings + vector search + LLM.",
    version="0.1.0",
)

# Initialize RAG components once at startup
retriever = Retriever(top_k=3)
generator = Generator()

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Useful for monitoring / readiness probes.
    """
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """
    Main RAG endpoint.
    1) Takes a user question.
    2) Retrieves the most relevant chunks from the vector DB.
    3) Calls the LLM with the chunks as context.
    4) Returns the answer and the contexts used.
    """
    query = request.query

    # 1. Retrieve relevant chunks
    chunks = retriever.get_top_texts(query)

    if not chunks:
        # If nothing is found in the DB, be honest
        return AskResponse(
            answer="I don't know based on the provided information.",
            contexts=[],
        )

    # 2. Generate answer using LLM + context
    result = generator.generate_answer(query, chunks)

    return AskResponse(
        answer=result["answer"],
        contexts=result["context_used"],
    )