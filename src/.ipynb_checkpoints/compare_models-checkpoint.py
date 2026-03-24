# compare_models.py
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq

from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever

# ----------------------------
# Load resources
# ----------------------------
retriever = Retriever()

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

semantic_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEM_MODEL = "llama-3.1-8b-instant"

COSINE_THRESHOLD = 0.40
ABSTAIN_PHRASE = "the answer is not found in the provided documents"

MODELS = {
    "small": "Allam-2-7B",
    "medium": "Llama-3.1-8B",
    "large": "Qwen3-32B",
}

# ---------------------------------------
# Hallucination evaluation utilities
# ---------------------------------------
def cosine_hallucination(answer, docs, threshold=COSINE_THRESHOLD):
    if not docs:
        return True, 0.0

    ctx = " ".join([d["text"] for d in docs])
    ans_vec = embedder.encode([answer])
    ctx_vec = embedder.encode([ctx])

    sim = float(cosine_similarity(ans_vec, ctx_vec)[0][0])
    return sim < threshold, sim


def semantic_check(question, answer, docs):
    if not docs:
        return True

    ctx = "\n\n---\n".join([d["text"] for d in docs])

    prompt = f"""
Does the answer contain ANY unsupported info?

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


# ----------------------------
# Load questions
# ----------------------------
with open("questions.txt", "r") as f:
    questions = [q.strip() for q in f.readlines() if q.strip()]

print("\n📘 Loaded", len(questions), "questions\n")


# ----------------------------
# Score tracking
# ----------------------------
results = {m: {"hallucinations": 0, "correct": 0, "total": 0} for m in MODELS}


# ----------------------------
# Evaluation Loop
# ----------------------------
for q in questions:
    print("\n======================================================")
    print("QUESTION:", q)
    print("======================================================")

    for key in MODELS:
        rag = RAGPipeline(key)
        out = rag.answer(q)

        answer = out["answer"]
        docs = out["retrieved_docs"]

        # abstain rule
        if ABSTAIN_PHRASE in answer.lower():
            halluc = False
            sim = 0.0
        else:
            cos_flag, sim = cosine_hallucination(answer, docs)
            sem_flag = semantic_check(q, answer, docs)
            halluc = cos_flag or sem_flag

        # score update
        results[key]["total"] += 1
        if halluc:
            results[key]["hallucinations"] += 1
        else:
            results[key]["correct"] += 1

        print(f"\n--- {MODELS[key]} ---")
        print("Answer:", answer[:200] + "...")  # shorten output
        print("Cosine:", round(sim, 3))
        print("Hallucinated:", halluc)

        time.sleep(0.7)  # slow down for API stability


# ----------------------------
# Final score summary
# ----------------------------
print("\n\n============================")
print("FINAL MODEL SCORES")
print("============================\n")

for key in MODELS:
    c = results[key]["correct"]
    h = results[key]["hallucinations"]
    t = results[key]["total"]

    acc = c / t
    hall_rate = h / t

    print(f"📌 {MODELS[key]}:")
    print(f"   Total: {t}")
    print(f"   Correct (grounded): {c}")
    print(f"   Hallucinations: {h}")
    print(f"   Accuracy: {acc:.2f}")
    print(f"   Hallucination Rate: {hall_rate:.2f}\n")

print("✔ Comparison complete.")
