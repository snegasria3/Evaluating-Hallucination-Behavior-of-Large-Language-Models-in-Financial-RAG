# 📊 Evaluating Hallucination Behavior of Large Language Models in Financial RAG

---

## 📌 Overview  

This project focuses on analyzing **hallucination behavior in Large Language Models (LLMs)** using a **Financial Retrieval-Augmented Generation (RAG) system**.  

LLMs often generate incorrect or unsupported information, especially in critical domains like finance. This project builds an end-to-end system that ensures **factual grounding, reliability, and transparency** in AI-generated financial answers.  

---

## 🎯 Objectives  

- Build a financial knowledge base using real-world datasets  
- Retrieve relevant documents using dense vector search  
- Generate answers using LLMs constrained by retrieved evidence  
- Detect hallucinations using similarity and semantic validation  
- Compare performance of multiple LLMs  
- Analyze reliability of LLMs in financial QA  

---

## 🚀 Key Features  

- 📚 Financial RAG pipeline  
- 🔍 FAISS-based dense retrieval  
- 🤖 Multi-model comparison:
  - Allam-2-7B (Small)  
  - Llama-3.1-8B (Medium)  
  - Qwen3-32B (Large)  
- ⚠️ Hallucination detection using:
  - Cosine similarity  
  - Semantic LLM-based judge  
- 📊 Evaluation on 30 + 10 financial questions  
- 📈 Visualization of results  

---

## 🧠 System Architecture  

The pipeline works as follows:

1. User query input  
2. Query embedding using MiniLM  
3. FAISS retrieves top-k relevant documents  
4. RAG prompt construction  
5. LLM generates answer  
6. Hallucination detection using:
   - Cosine similarity  
   - Semantic verification  

---

## 🗂️ Dataset  

The system uses two financial datasets:

- 📄 Financial News Articles (~300,000+ rows)  
- 🏦 Central Bank Communications (~140,000+ rows)  

All documents are:
- Cleaned and normalized  
- Stored in JSONL format  
- Converted into dense embeddings  

---

## ⚙️ Methodology  

### 🔹 1. Dense Vector Representation  
- Model: `all-MiniLM-L6-v2`  
- Generates 384-dimensional embeddings  

### 🔹 2. FAISS Retrieval  
- Fast nearest neighbor search  
- Retrieves top-k relevant documents  

### 🔹 3. RAG Pipeline  
- Combines retrieval + generation  
- Ensures answers are based on evidence  

### 🔹 4. Multi-Model Evaluation  
- Small: Allam-2-7B  
- Medium: Llama-3.1-8B  
- Large: Qwen3-32B  

---

## 🔍 Hallucination Detection  

### ✅ Cosine Similarity  
- Measures similarity between answer and context  
- Threshold = **0.40**

### ✅ Semantic LLM Judge  
- Checks if answer contains unsupported info  

### ✅ Final Rule  
- If abstention → NOT hallucination  
- Else → hallucination if any method fails  

---

## 📊 Evaluation Metrics  

- Hallucination Rate  
- Grounded Accuracy  
- Abstention Rate  
- Average Similarity  

---
