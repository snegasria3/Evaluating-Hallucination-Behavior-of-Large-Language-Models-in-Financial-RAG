# src/data_loader.py

from datasets import load_dataset
from typing import List, Dict
import json

from .config import DATASETS, DOC_STORE_PATH


def load_financial_corpus() -> List[Dict]:
    """
    Load and unify documents from both HuggingFace datasets into a single list.
    Returns: List[Dict] with entries {id, source, title, text}
    """
    docs: List[Dict] = []

    # ----------------------------
    # 1) Load Financial News Dataset
    # ----------------------------
    print("📌 Loading dataset:", DATASETS["news"])
    news_ds = load_dataset(DATASETS["news"], split="train")
    print("   → Rows:", len(news_ds))

    for i, row in enumerate(news_ds):
        text = row.get("text") or row.get("content") or ""
        title = row.get("title") or row.get("headline") or ""
        if not text:
            continue

        docs.append({
            "id": f"news_{i}",
            "source": "news",
            "title": title,
            "text": text,
        })

    # ----------------------------
    # 2) Load Central Bank Speeches Dataset (STREAMING)
    # ----------------------------
    print("📌 Loading dataset (streaming):", DATASETS["speeches"])
    speeches_stream = load_dataset(
        DATASETS["speeches"],
        split="train",
        streaming=True
    )

    speeches_count = 0

    for i, row in enumerate(speeches_stream):
        text = (
            row.get("text")
            or row.get("document")
            or row.get("chunk_text")
            or row.get("summary")
            or ""
        )
        title = (
            row.get("title")
            or row.get("source")
            or row.get("doc_id")
            or ""
        )

        if not text:
            continue

        docs.append({
            "id": f"speech_{i}",
            "source": "speech",
            "title": title,
            "text": text,
        })

        speeches_count += 1
        if speeches_count >= 20000:
            break

    print(f"⚡ Total documents collected: {len(docs)}")
    return docs


def save_doc_store(docs: List[Dict], path: str = DOC_STORE_PATH) -> None:
    """Save unified corpus into JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"💾 Saved doc store to: {path} (rows: {len(docs)})")
