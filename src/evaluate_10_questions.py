# evaluate_10_questions.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq

from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever


# ======================================================
# CONFIG
# ======================================================

RESULTS_DIR = "results_raw"
PLOTS_DIR = f"{RESULTS_DIR}/plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COSINE_THRESHOLD = 0.40
ABSTAIN = "the answer is not found in the provided documents"

MODELS = {
    "small": "Allam-2-7B",
    "medium": "Llama-3.1-8B",
    "large": "Qwen3-32B",
}


# ======================================================
# LOAD GLOBAL MODELS
# ======================================================

retriever = Retriever()

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

semantic_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEM_MODEL = "llama-3.1-8b-instant"


# ======================================================
# HALLUCINATION CHECKS
# ======================================================

def cosine_hallucination(answer, docs):
    if not docs:
        return True, 0.0

    ctx = " ".join([d["text"] for d in docs])
    a_vec = embedder.encode([answer])
    c_vec = embedder.encode([ctx])
    sim = float(cosine_similarity(a_vec, c_vec)[0][0])

    return sim < COSINE_THRESHOLD, sim


def semantic_check(question, answer, docs):
    if not docs:
        return True

    context = "\n\n---\n".join([d["text"] for d in docs])

    prompt = f"""
Does the answer contain any info NOT supported by the context?

Question: {question}

Answer: {answer}

Context:
{context}

Reply only: yes or no.
"""

    try:
        resp = semantic_llm.chat.completions.create(
            model=SEM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        reply = resp.choices[0].message.content.strip().lower()
        return reply == "yes"
    except:
        return False


# ======================================================
# LOAD QUESTIONS
# ======================================================
with open("questions.txt", "r") as f:
    questions = [q.strip() for q in f.readlines() if q.strip()]

print(f"\n📝 Loaded {len(questions)} questions\n")


# ======================================================
# RUN EVALUATION
# ======================================================

records = []

for q in tqdm(questions):

    for key in MODELS:

        rag = RAGPipeline(key)
        out = rag.answer(q)

        answer = out["answer"]
        docs = out["retrieved_docs"]

        # Abstain rule = always grounded
        if ABSTAIN in answer.lower():
            final_hall = False
            sim = 0.0
        else:
            cos_flag, sim = cosine_hallucination(answer, docs)
            sem_flag = semantic_check(q, answer, docs)
            final_hall = cos_flag or sem_flag

        records.append({
            "question": q,
            "model": MODELS[key],
            "cosine_similarity": sim,
            "hallucination": final_hall
        })


# ======================================================
# SAVE CSV
# ======================================================

df = pd.DataFrame(records)
df.to_csv(f"{RESULTS_DIR}/compare_results.csv", index=False)

print("\n💾 Saved CSV → results_raw/compare_results.csv")


# ======================================================
# PLOTS
# ======================================================

# ---------- 1. Model Accuracy ----------
acc_df = (
    df.groupby("model")["hallucination"]
    .apply(lambda x: 1 - x.mean())
    .reset_index(name="accuracy")
)

plt.figure(figsize=(7,5))
sns.barplot(data=acc_df, x="model", y="accuracy")
plt.title("Model Accuracy (1 - Hallucination Rate)")
plt.xticks(rotation=15)
plt.ylim(0,1)
plt.savefig(f"{PLOTS_DIR}/model_accuracy.png")
plt.close()


# ---------- 2. Model Hallucination Rate ----------
hall_df = (
    df.groupby("model")["hallucination"]
    .mean()
    .reset_index(name="hallucination_rate")
)

plt.figure(figsize=(7,5))
sns.barplot(data=hall_df, x="model", y="hallucination_rate")
plt.title("Hallucination Rate per Model")
plt.xticks(rotation=15)
plt.ylim(0,1)
plt.savefig(f"{PLOTS_DIR}/model_hallucination_rate.png")
plt.close()


# ---------- 3. Cosine Similarity Distribution ----------
plt.figure(figsize=(8,5))
sns.histplot(df["cosine_similarity"], bins=20, kde=True)
plt.title("Cosine Similarity Distribution (All Models)")
plt.savefig(f"{PLOTS_DIR}/cosine_similarity_distribution.png")
plt.close()


# ---------- 4. Model-wise Cosine Similarity ----------
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="model", y="cosine_similarity")
plt.title("Cosine Similarity per Model")
plt.xticks(rotation=15)
plt.savefig(f"{PLOTS_DIR}/model_cosine_similarity.png")
plt.close()


# ---------- 5. Radar Plot (Performance Overview) ----------
import numpy as np

radar_df = acc_df.copy()
radar_df["hall_rate"] = hall_df["hallucination_rate"]

# Radar needs circular data
labels = ["Accuracy", "Hallucination Rate"]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(6,6))
for _, row in radar_df.iterrows():
    vals = [row["accuracy"], row["hall_rate"]]
    vals += vals[:1]
    plt.polar(angles, vals, marker='o', label=row["model"])

plt.title("Model Comparison Radar Chart")
plt.legend(loc="upper right")
plt.savefig(f"{PLOTS_DIR}/model_radar.png")
plt.close()


print("\n📊 Plots saved → results_raw/plots/")
print("🎉 Evaluation complete!\n")
