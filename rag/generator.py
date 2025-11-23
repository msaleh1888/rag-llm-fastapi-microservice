# rag/main.py

import os
from groq import Groq


class Generator:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ):
        # Load key from parameter or env
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is missing. Set GROQ_API_KEY.")

        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name

    def generate_answer(self, query: str, chunks: list[str]) -> str:
        context = "\n\n".join(chunks)

        prompt = f"""
You are a helpful assistant. Answer using ONLY the context.

Context:
{context}

Question: {query}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )

        answer_text = response.choices[0].message.content.strip()

        return {
        "answer": answer_text,
        "context_used": chunks,
    }