# scripts/ingest.py

import os
from typing import List, Tuple
from pypdf import PdfReader

from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.vectordb import VectorDB

def load_text_files(directory: str = "data/raw") -> List[Tuple[str, str]]:
    """
    Load .txt files from a directory.

    Returns:
        List of (source_name, text_content).
    """
    docs  = []

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".txt"):
            continue

        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        docs.append((filename, text))
    
    return docs

def load_pdf_files(directory: str = "data/raw") -> List[Tuple[str, str]]:
    """
    Load .pdf files from a directory and extract their text.

    Returns:
        List of (source_name, text_content).
    """
    docs = []

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".pdf"):
            continue
    
        path = os.path.join(directory, filename)
        reader = PdfReader(path)
    
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
    
        full_text = "\n".join(pages_text)
        docs.append((filename, full_text))

    return docs

def ingest():
    # 1. Load raw documents (txt + pdf)
    txt_docs = load_text_files()
    pdf_docs = load_pdf_files()

    all_docs = txt_docs + pdf_docs

    print(f"Loaded {len(all_docs)} documents from data/raw.")

    if not all_docs:
        print("No documents found in data/raw. Add .txt or .pdf files and try again.")
        return

    # 2. Split each document into chunks (and keep basic metadata)
    all_chunks: List[str] = []
    all_metadatas = []

    for source_name, text in all_docs:
        chunks = chunk_text(text, chunk_size=4, overlap=1)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": source_name})

    print(f"Created {len(all_chunks)} chunks from all documents.")

    # 3. Embed chunks
    embedder = Embedder()
    embeddings = embedder.embed_documents(all_chunks).tolist()

    # 4. Insert into Chroma vector DB
    vectordb = VectorDB()
    vectordb.add_texts(
        texts=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )

    print("Ingestion completed. Vector DB is ready.")


if __name__ == "__main__":
    ingest()