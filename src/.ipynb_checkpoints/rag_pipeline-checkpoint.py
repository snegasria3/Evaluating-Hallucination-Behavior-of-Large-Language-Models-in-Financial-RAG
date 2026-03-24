# src/rag_pipeline.py

from typing import List, Dict

from .retriever import Retriever
from .llm_clients_groq import GroqLLMClient


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline:
      1) retrieve top-k docs with FAISS
      2) build a grounded prompt
      3) call Groq LLM
      4) return answer + retrieved docs + prompt (for debugging)
    """

    def __init__(self, model_key: str):
        self.retriever = Retriever()
        self.llm = GroqLLMClient(model_key)

    # ----------------------------------------------------------
    # PROMPT BUILDER
    # ----------------------------------------------------------
    def build_prompt(self, question: str, docs: List[Dict]) -> str:
        context_block = "\n\n--- DOCUMENT ---\n\n".join(
            [
                f"Title: {d.get('title', '')}\nText: {d.get('text', '')}"
                for d in docs
            ]
        )

        prompt = f"""
You are a financial analysis assistant inside a Retrieval-Augmented Generation (RAG) system.

RULES:
- Use ONLY the information in the retrieved documents below.
- If the answer is not contained in these documents, reply exactly:
  "The answer is not found in the provided documents."
- Do NOT guess or invent facts.

Question:
{question}

Retrieved documents:
{context_block}

Answer:
""".strip()

        return prompt

    # ----------------------------------------------------------
    # MAIN RAG ANSWER FUNCTION
    # ----------------------------------------------------------
    def answer(self, question: str) -> Dict:
        docs = self.retriever.retrieve(question)
        prompt = self.build_prompt(question, docs)
        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": docs,
            "prompt": prompt,
        }
