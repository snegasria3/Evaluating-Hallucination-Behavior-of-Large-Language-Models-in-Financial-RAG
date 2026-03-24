# streamlit_app.py  

import os

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

from src.rag_pipeline import RAGPipeline

# -------------------------------------------------------
# Global components
# -------------------------------------------------------

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
)

semantic_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEM_MODEL = "llama-3.1-8b-instant"   # semantic checker always uses 8B

MODELS = {
    "small":  "Small (Allam-2-7B)",
    "medium": "Medium (LLaMA-3.1-8B)",
    "large":  "Large (Qwen3-32B)",
}


def cosine_hallucination(answer, docs, threshold=0.40):
    if not docs:
        return True, 0.0

    context = " ".join([d.get("text", "") for d in docs])

    ans_vec = embedder.encode([answer])
    ctx_vec = embedder.encode([context])

    sim = float(cosine_similarity(ans_vec, ctx_vec)[0][0])
    return sim < threshold, sim


def semantic_check(question, answer, docs):
    if not docs:
        return True

    context = "\n\n---\n".join([d.get("text", "") for d in docs])

    prompt = f"""
You are a financial hallucination-detection model.

Question:
{question}

Answer:
{answer}

Context (retrieved documents):
{context}

Does the answer contain ANY information not supported by the context?
Reply exactly with: yes or no.
"""

    try:
        resp = semantic_llm.chat.completions.create(
            model=SEM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = resp.choices[0].message.content.strip().lower()
        return result == "yes"
    except Exception:
        return False


# -------------------------------------------------------
# UI
# -------------------------------------------------------

st.set_page_config(page_title="Financial RAG", layout="wide")
st.title("🧠 Financial RAG — Compare Models & Detect Hallucination")

query = st.text_area("Ask a financial question:", height=90)

colA, colB = st.columns(2)
with colA:
    compare = st.checkbox("Compare all models")
with colB:
    selected = st.selectbox("Or choose a single model:", list(MODELS.keys()))

if st.button("🚀 Run RAG"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    model_list = list(MODELS.keys()) if compare else [selected]

    st.markdown("---")
    st.subheader("🔎 Model Outputs")

    cols = st.columns(len(model_list))

    for col, model_key in zip(cols, model_list):

        rag = RAGPipeline(model_key)
        out = rag.answer(query)

        answer = out["answer"]
        docs = out["retrieved_docs"]

        cos_flag, sim_score = cosine_hallucination(answer, docs)
        sem_flag = semantic_check(query, answer, docs)
        final = cos_flag or sem_flag

        with col:
            st.header(f"🧩 {MODELS[model_key]}")
            st.subheader("📘 Model Answer")
            st.write(answer)

            st.subheader("🧪 Hallucination Check")
            if final:
                st.error("🚨 Hallucination Detected")
            else:
                st.success("✅ No Hallucination Detected")

            st.write(f"**Cosine Similarity:** {sim_score:.3f}")
            st.write(f"**Semantic Check:** {'Yes' if sem_flag else 'No'}")

            st.subheader("📚 Retrieved Documents")
            if not docs:
                st.write("No documents retrieved.")
            else:
                for d in docs:
                    with st.expander(f"{d['title']} (score: {d['score']:.4f})"):
                        st.write(d["text"])
