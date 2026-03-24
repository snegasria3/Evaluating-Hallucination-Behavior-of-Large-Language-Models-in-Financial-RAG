# src/retriever.py

from typing import List, Dict
import numpy as np
from sentence_transformers import models, SentenceTransformer

from .config import EMBEDDING_MODEL, TOP_K
from .index_builder import load_faiss_index


# -------------------------------------------------------------------
# SAFE EMBEDDING LOADER FOR WINDOWS (no .to(), no meta-tensor issues)
# -------------------------------------------------------------------
def load_embedder():
    word = models.Transformer(EMBEDDING_MODEL)  # DO NOT PASS device
    pool = models.Pooling(word.get_word_embedding_dimension())
    return SentenceTransformer(modules=[word, pool], device=None)


class Retriever:
    """
    FAISS dense retriever.
    """

    def __init__(self):
        print("🔍 Loading FAISS index + doc store...")
        self.index, self.docs = load_faiss_index()

        print("🔤 Loading embedding model safely (no .to())...")
        self.model = load_embedder()

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k documents for a query.
        """
        emb = self.model.encode([query], convert_to_numpy=True).astype("float32")

        scores, indices = self.index.search(emb, TOP_K)
        scores = scores[0]
        indices = indices[0]

        out = []
        for idx, score in zip(indices, scores):
            doc = dict(self.docs[idx])
            doc["score"] = float(score)
            out.append(doc)

        return out

    def get_docs_by_titles(self, titles: List[str]) -> List[Dict]:
        """
        Used by evaluation_local.py to load full docs from stored titles.
        """
        title_set = {t.strip() for t in titles if t and t.strip()}
        if not title_set:
            return []

        mapping = {d.get("title", ""): d for d in self.docs}

        return [mapping[t] for t in title_set if t in mapping]
