import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data + results directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "results_raw")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Make sure folders exist
for d in [DATA_DIR, RESULTS_DIR, RAW_RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# HuggingFace datasets used for the RAG corpus
DATASETS = {
    "news": "ashraq/financial-news-articles",
    "speeches": "SelmaNajih001/FinancialNewsAndCentralBanksSpeeches-Summary-Rag"
}

# Embedding model (fast and good quality for RAG)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Paths for processed doc store + FAISS index
DOC_STORE_PATH = os.path.join(DATA_DIR, "doc_store.jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# Retrieval settings
TOP_K = 2

# Models we are evaluating
LLM_CONFIGS = {
    "llama3":  {"id": "llama3",  "max_tokens": 512},
    "mistral": {"id": "mistral", "max_tokens": 512},
    "gemma":   {"id": "gemma",   "max_tokens": 512},
}

