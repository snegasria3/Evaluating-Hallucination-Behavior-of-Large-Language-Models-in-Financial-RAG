# src/evaluation_local.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

from .retriever import Retriever

# =====================================
# Paths
# =====================================
BASE = os.getcwd()

INPUT_FILE = f"{BASE}\\results\\answers_full.csv"
OUTPUT_METRICS = f"{BASE}\\results\\metrics_summary.csv"
OUTPUT_HALLUS = f"{BASE}\\results\\hallucination_failures.csv"
OUTPUT_GROUNDED = f"{BASE}\\results\\grounded_failures.csv"
PLOTS_DIR = f"{BASE}\\results\\plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

COSINE_THRESHOLD = 0.40
ABSTAIN_PHRASE = "the answer is not found in the provided documents"


# =====================================
# Load models once
# =====================================
print("🔍 Loading FAISS retriever...")
retriever = Retriever()

print("🔍 Loading MiniLM embedder...")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("🔍 Loading semantic LLM...")
semantic_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEM_MODEL = "llama-3.1-8b-instant"


# =====================================
# Cosine check
# =====================================
def cosine_hallucination(answer, docs):
    if not docs:
        return True, 0.0

    context = " ".join(d["text"] for d in docs)
    ans_vec = embedder.encode([answer])
    ctx_vec = embedder.encode([context])

    sim = float(cosine_similarity(ans_vec, ctx_vec)[0][0])
    return sim < COSINE_THRESHOLD, sim


# =====================================
# Semantic check
# =====================================
def semantic_check(question, answer, docs):
    if not docs:
        return True

    ctx = "\n\n---\n".join([d["text"] for d in docs])

    prompt = f"""
Does the answer contain ANY information not supported by the context?

Question:
{question}

Answer:
{answer}

Context:
{ctx}

Reply only: yes or no.
"""

    try:
        out = semantic_llm.chat.completions.create(
            model=SEM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return out.choices[0].message.content.strip().lower() == "yes"
    except:
        return False


# =====================================
# Main evaluation
# =====================================
def run_evaluation():

    print("\n📥 Loading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)
    print("Loaded", len(df), "rows\n")

    all_rows = []
    hallucinations = []
    grounded = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        q = row["question"]
        a = row["answer"]
        model_used = row["model"]

        # Docs
        titles = str(row["retrieved_docs"]).split("|")
        docs = retriever.get_docs_by_titles(titles)

        # ==================================
        # Abstain → always grounded
        # ==================================
        if ABSTAIN_PHRASE in a.lower():
            result = {
                "question": q,
                "model": model_used,
                "cosine_similarity": 0.0,
                "cos_flag": False,
                "sem_flag": False,
                "abstain": True,
                "final_hallucination": False,
            }
            all_rows.append(result)
            grounded.append(result)
            continue

        # ==================================
        # Normal evaluation
        # ==================================
        cos_flag, sim_score = cosine_hallucination(a, docs)
        sem_flag = semantic_check(q, a, docs)
        final_flag = cos_flag or sem_flag

        result = {
            "question": q,
            "model": model_used,
            "cosine_similarity": sim_score,
            "cos_flag": cos_flag,
            "sem_flag": sem_flag,
            "abstain": False,
            "final_hallucination": final_flag,
        }
        all_rows.append(result)

        if final_flag:
            hallucinations.append(result)
        else:
            grounded.append(result)

    # Save results
    dfm = pd.DataFrame(all_rows)
    dfm.to_csv(OUTPUT_METRICS, index=False)
    pd.DataFrame(hallucinations).to_csv(OUTPUT_HALLUS, index=False)
    pd.DataFrame(grounded).to_csv(OUTPUT_GROUNDED, index=False)

    print("Saved metrics →", OUTPUT_METRICS)

    # ===========================
    # SUMMARY METRICS PER MODEL
    # ===========================

    # Accuracy = 1 - hallucination rate
    model_acc = (
        dfm.groupby("model")["final_hallucination"]
        .apply(lambda x: 1 - x.mean())
        .reset_index(name="accuracy")
    )

    # Hallucination rate
    model_halluc = (
        dfm.groupby("model")["final_hallucination"].mean().reset_index(name="hallucination_rate")
    )

    # Abstention rate
    model_abstain = (
        dfm.groupby("model")["abstain"].mean().reset_index(name="abstention_rate")
    )

    # COSINE
    model_cosine = (
        dfm.groupby("model")["cosine_similarity"].mean().reset_index(name="avg_similarity")
    )

    # Merge all into a single table for your professor
    summary = model_acc.merge(model_halluc, on="model") \
                       .merge(model_abstain, on="model") \
                       .merge(model_cosine, on="model")

    summary.to_csv(f"{BASE}\\results\\model_summary.csv", index=False)
    print("Saved summary table → model_summary.csv")


    # ===========================
    # PLOTS
    # ===========================

    print("📊 Generating plots...")

    # 1 — Cosine Similarity Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(dfm["cosine_similarity"], kde=True, bins=20, color="skyblue")
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.savefig(f"{PLOTS_DIR}\\cosine_similarity_distribution.png")
    plt.close()

    # 2 — Hallucination Rate Per Model
    plt.figure(figsize=(7, 5))
    sns.barplot(data=model_halluc, x="model", y="hallucination_rate")
    plt.title("Hallucination Rate by Model")
    plt.ylabel("Hallucination Rate")
    plt.savefig(f"{PLOTS_DIR}\\hallucination_rate_models.png")
    plt.close()

    # 3 — Model Accuracy
    plt.figure(figsize=(7, 5))
    sns.barplot(data=model_acc, x="model", y="accuracy")
    plt.title("Model Accuracy (1 - Hallucination Rate)")
    plt.ylabel("Accuracy")
    plt.savefig(f"{PLOTS_DIR}\\model_accuracy.png")
    plt.close()

    # 4 — Abstention Rate
    plt.figure(figsize=(7, 5))
    sns.barplot(data=model_abstain, x="model", y="abstention_rate")
    plt.title("Abstention Rate by Model")
    plt.ylabel("Abstention Rate")
    plt.savefig(f"{PLOTS_DIR}\\model_abstention.png")
    plt.close()

    # 5 — Similarity vs Hallucination Scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=dfm,
        x="cosine_similarity",
        y=dfm["final_hallucination"].astype(int),
        hue="model"
    )
    plt.title("Cosine Similarity vs Hallucination")
    plt.ylabel("Hallucination (1=yes, 0=no)")
    plt.savefig(f"{PLOTS_DIR}\\similarity_vs_hallucination.png")
    plt.close()

    print("📁 All plots saved:", PLOTS_DIR)
    print("\n🎉 Evaluation complete!\n")


if __name__ == "__main__":
    run_evaluation()
