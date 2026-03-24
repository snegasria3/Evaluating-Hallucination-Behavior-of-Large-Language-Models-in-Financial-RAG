# src/llm_clients_groq.py

import os
from groq import Groq

# ---------------------------------------------------
# MODEL MAP: small / medium / large
# ---------------------------------------------------
MODEL_MAP = {
    "small":  "allam-2-7b",
    "medium": "llama-3.1-8b-instant",
    "large":  "qwen/qwen3-32b",

    # legacy keys, in case anything still uses old names
    "llama8b":  "llama-3.1-8b-instant",
    "qwen32b":  "qwen/qwen3-32b",
}


class GroqLLMClient:
    """
    Groq LLM wrapper used by the RAG pipeline.
    """

    def __init__(self, model_key: str):
        # Validate model key
        if model_key not in MODEL_MAP:
            raise ValueError(
                f"❌ Unknown model '{model_key}'. "
                f"Available: {list(MODEL_MAP.keys())}"
            )

        # Load API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY is missing.")

        self.client = Groq(api_key=api_key)
        self.model_id = MODEL_MAP[model_key]

    # ---------------------------------------------------
    # Generate text from model
    # ---------------------------------------------------
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious financial RAG assistant. "
                        "Use ONLY the information from the provided documents. "
                        "If the answer is not in the context, reply exactly:\n"
                        '"The answer is not found in the provided documents."'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        # groq python client: message is an object, use `.content`
        return response.choices[0].message.content
