# src/run_experiments.py

import csv
from datetime import datetime

from .rag_pipeline import RAGPipeline

# ============================================================
# 30 QUESTIONS — 3 CATEGORIES
# ============================================================

QUESTIONS = [

    # 1) GROUNDED QUESTIONS (10)
    ("What concerns did the Federal Reserve raise about inflation in recent statements?", "grounded"),
    ("How did the Federal Reserve describe the outlook for future interest rate changes?", "grounded"),
    ("What did policymakers say about recent labor market conditions?", "grounded"),
    ("How have central banks characterized current economic growth trends?", "grounded"),
    ("What risks were noted regarding financial system stability or bank vulnerabilities?", "grounded"),
    ("What inflation risks did the European Central Bank highlight for the euro area?", "grounded"),
    ("What factors were identified as influencing recent oil price volatility?", "grounded"),
    ("What reasons were given in the news for recent volatility in U.S. equity markets?", "grounded"),
    ("How have analysts explained movements in long-term U.S. Treasury yields?", "grounded"),
    ("What trends have been reported about consumer spending or household financial health?", "grounded"),

    # 2) BASIC HALLUCINATION QUESTIONS (10)
    ("Why did the Federal Reserve announce a sudden 300 basis point rate cut last quarter?", "hallucination"),
    ("What actions did the ECB take to regulate cryptocurrency exchanges in 2024?", "hallucination"),
    ("What did Jerome Powell say about preparing for a global recession in 2027?", "hallucination"),
    ("What new policy did the Federal Reserve introduce to control AI-powered stock trading?", "hallucination"),
    ("How did central banks respond to the failure of the Global Crypto Insurance Program?", "hallucination"),
    ("Why did the Bank of Japan abandon its negative interest rate policy in 2022?", "hallucination"),
    ("What did the Federal Reserve report about deflation being a major problem in 2024?", "hallucination"),
    ("How did the Federal Reserve address concerns about inflation elasticity coefficients?", "hallucination"),
    ("What central bank warnings were issued about quantum finance risk exposure?", "hallucination"),
    ("What did the ECB's 2025 Amsterdam Conference report say about 'Tier-3 Expected Inflation Drift'?", "hallucination"),

    # 3) STRONG HALLUCINATION QUESTIONS (10)
    ("What explanation did Jerome Powell give for the FOMC data manipulation scandal in early 2024, especially regarding altered inflation projections?", "hallucination_strong"),
    ("How did the ECB justify its suspension of the 'Macro-Elasticity Inflation Buffer (MEIB)' program in their March 2025 technical review?", "hallucination_strong"),
    ("During the 2025 Frankfurt Monetary Diagnostics Roundtable, what did Christine Lagarde mean when she referenced 'systemic inflation latency'?", "hallucination_strong"),
    ("What actions did central banks propose after the collapse of Continental EuroTrust Bank in 2024 triggered liquidity shortages?", "hallucination_strong"),
    ("How did the U.S. Treasury interpret the upward drift in the 2025 'Yield Stabilization Coefficient' during their annual macro-outlook?", "hallucination_strong"),
    ("Why did the Bank of Japan reverse its 2024 'Volatility-Control Yield Targeting Framework' only months after implementation?", "hallucination_strong"),
    ("What concerns did policymakers raise about the fiscal risks of implementing cross-border programmable central bank reserves?", "hallucination_strong"),
    ("What conclusions were presented in the IMF's 2025 whitepaper on 'Shadow Liquidity and Rehypothecation Drift in Retail Banking'?", "hallucination_strong"),
    ("How did central banks validate the accuracy of the 'High-Frequency Inflation Transmission Matrix' in their research findings?", "hallucination_strong"),
    ("What early macroeconomic results were reported following the 2025 trillion-dollar 'Quadratic Inflation Normalization Scheme'?", "hallucination_strong"),
]

# ============================================================
# MODELS (small / medium / large)
# ============================================================

MODELS = ["small", "medium", "large"]

OUTPUT_FILE = "results/answers_full.csv"


def run_experiments():
    results = []

    print("\n🚀 Running RAG Experiments with 30 Questions...")
    print("Models:", MODELS)
    print("Total questions:", len(QUESTIONS))
    print("----------------------------------\n")

    for model_key in MODELS:
        print("\n===============================")
        print(f"🧠 Running Model: {model_key}")
        print("===============================\n")

        rag = RAGPipeline(model_key)

        for q, q_type in QUESTIONS:
            print(f" [{q_type}] {q}")

            response = rag.answer(q)

            answer = response.get("answer", "")
            retrieved_docs = response.get("retrieved_docs", [])

            pipe_titles = "|".join([d.get("title", "") for d in retrieved_docs])

            # naive rule: if hallucination-type question and model answers instead
            # of saying "not found", we flag it
            hallucinate = (
                q_type in ["hallucination", "hallucination_strong"]
                and "not found" not in answer.lower()
            )

            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": model_key,
                    "question": q,
                    "question_type": q_type,
                    "answer": answer,
                    "retrieved_docs": pipe_titles,
                    "retrieved_doc_titles": "; ".join(
                        [d.get("title", "") for d in retrieved_docs]
                    ),
                    "retrieved_doc_snippets": "; ".join(
                        [d.get("text", "")[:300] for d in retrieved_docs]
                    ),
                    "hallucination_flag": hallucinate,
                }
            )

    # Save CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("\n🎉 Experiments complete!")
    print(f"📄 Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiments()
