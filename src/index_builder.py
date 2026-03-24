# src/index_builder.py

import json
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, DOC_STORE_PATH, FAISS_INDEX_PATH


def load_doc_store(path: str = DOC_STORE_PATH) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"📖 Loaded {len(docs)} docs from doc_store")
    return docs


def build_faiss_index(batch_size: int = 512):
    """
    Build a FAISS index over the full corpus using SentenceTransformer embeddings.
    """
    docs = load_doc_store(DOC_STORE_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [d["text"] for d in docs]
    print(f"🧮 Encoding {len(texts)} documents...")

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        print(f"   → Batch {start} to {start + len(batch)}")
        emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")

    dim = embeddings.shape[1]
    print(f"📐 Embedding matrix: {embeddings.shape}")

    # FAISS Index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")


def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    docs = load_doc_store(DOC_STORE_PATH)
    print(f"📥 Loaded FAISS index with {index.ntotal} vectors")
    return index, docs
