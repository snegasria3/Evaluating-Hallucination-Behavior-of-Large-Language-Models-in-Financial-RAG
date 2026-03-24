# streamlit_app.py  (ROOT OF PROJECT)

import os

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.rag_pipeline import RAGPipeline


# ======================================
#  🔥 Lazy, safe embedder loader
# ======================================
@st.cache_resource
def get_embedder():
    """
    Lazily load MiniLM for cosine similarity.

    If something goes wrong (e.g. meta-tensor / device issues),
    we will catch it in cosine_hallucination and fall back
    to *only* semantic hallucination checks.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


# ======================================
# Global LLM client for semantic checks
# ======================================
semantic_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEM_MODEL = "llama-3.1-8b-instant"

# These keys must match what RAGPipeline / GroqLLMClient expect
MODELS = {
    "small":  "Small (Allam-2-7B)",
    "medium": "Medium (LLaMA-3.1-8B)",
    "large":  "Large (Qwen3-32B)",
}


# ======================================
# Cosine similarity check (with fallback)
# ======================================
def cosine_hallucination(answer, docs, threshold: float = 0.40):
    """
    Returns (is_hallucination, cosine_similarity).

    If the embedder fails to load on this machine, we fall back to:
      (False, 0.0) and rely only on the semantic LLM detector.
    """
    if not docs:
        return True, 0.0

    context = " ".join([d["text"] for d in docs])

    try:
        embedder = get_embedder()
        ans_vec = embedder.encode([answer])
        ctx_vec = embedder.encode([context])
        sim = float(cosine_similarity(ans_vec, ctx_vec)[0][0])
        return sim < threshold, sim

    except Exception as e:
        # Soft-fail: don’t crash the UI, just skip cosine check
        st.warning(f"Cosine similarity check disabled (embedding error: {e})")
        return False, 0.0


# ======================================
# Semantic hallucination check
# ======================================
def semantic_check(question: str, answer: str, docs):
    """
    Uses Groq Llama-3.1-8B to decide if the answer
    adds unsupported information compared to context.
    """
    if not docs:
        return True

    ctx = "\n\n---\n".join([d["text"] for d in docs])

    prompt = f"""
You are a financial hallucination-detection model.

Question:
{question}

Answer:
{answer}

Context:
{ctx}

Does the answer contain ANY information that is not supported by the context?
Reply with exactly one word: yes or no.
"""

    try:
        out = semantic_llm.chat.completions.create(
            model=SEM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = out.choices[0].message.content.strip().lower()
        return reply == "yes"
    except Exception as e:
        st.warning(f"Semantic hallucination check failed: {e}")
        # Safe default: don't auto-mark as hallucination if checker died
        return False


# ======================================
# Streamlit UI
# ======================================
st.set_page_config(page_title="Financial RAG", layout="wide")
st.title("💹 Financial RAG — Compare Models & Detect Hallucination")

query = st.text_area("Ask a financial question:", height=90)

colA, colB = st.columns(2)
with colA:
    compare = st.checkbox("Compare all models")
with colB:
    selected = st.selectbox("Choose a single model:", list(MODELS.keys()))

if st.button("🚀 Run RAG"):
    if not query.strip():
        st.warning("Enter a question first.")
        st.stop()

    # Decide which model keys to run
    model_keys = list(MODELS.keys()) if compare else [selected]

    cols = st.columns(len(model_keys))

    for col, model_key in zip(cols, model_keys):
        # Call your RAG pipeline
        rag = RAGPipeline(model_key)
        out = rag.answer(query)

        answer = out["answer"]
        docs = out["retrieved_docs"]

        # Hallucination checks
        cos_flag, sim_score = cosine_hallucination(answer, docs)
        sem_flag = semantic_check(query, answer, docs)
        final_flag = cos_flag or sem_flag

        with col:
            st.header(f"🧩 {MODELS[model_key]}")
            st.subheader("Model Answer")
            st.write(answer)

            st.subheader("Hallucination Check")
            if final_flag:
                st.error("🚨 Hallucination Detected")
            else:
                st.success("✅ No Hallucination Detected")

            st.write(f"**Cosine Similarity:** {sim_score:.3f}")
            st.write(f"**Semantic Check:** {'Yes' if sem_flag else 'No'}")

            st.subheader("Retrieved Documents")
            if not docs:
                st.write("No documents retrieved.")
            else:
                for d in docs:
                    with st.expander(f"{d['title']} (score: {d['score']:.4f})"):
                        st.write(d["text"])
