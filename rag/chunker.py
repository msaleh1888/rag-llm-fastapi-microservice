# rag/chunker.py

import re
from typing import List

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using basic punctuation.
    This is simple but effective for most documents.
    """
    # Split on '.', '?', '!' and keep delimiters
    sentences = re.split(r'(?<=[\.!?]) +', text.strip())

    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def chunk_text(
        text: str,
        chunk_size: int = 4,
        overlap: int = 1
) -> List[str]:
    """
    Splits text into chunks of N sentences with M overlap.
    Returns a list of chunk strings.
    """
    sentences = split_into_sentences(text)
    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(sentences):
        end = start + chunk_size
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += step

    return chunks