# 🚅 ChatrAIn - Local RAG for Railway Maintenance
![Logo app](img/contact-pic.png)
    
## APP PREVIEW
    
![Chat screenshot](img/screenshot_webapp.png)
     
**ChatrAIn** is an intelligent assistant designed to support railway maintenance workers by querying technical manuals via natural language. 

> **Evolution Note:** This project started as a prototype on Jupyter Notebooks (still available in `notebooks/`) and has evolved into a **modular, production-oriented Python architecture**.

## 🎯 Project Goal
Unlike generic chatbots, ChatrAIn answers technical questions based **exclusively** on the provided Maintenance Manuals (Vector DB). 

The architecture is fully **Air-Gapped** (Offline) and optimized for consumer hardware (Apple Silicon M1), ensuring:
* **Total Privacy:** Zero data sent to the cloud.
* **Zero Hallucinations:** The model admits ignorance if the manual doesn't contain the answer.

## ⚙️ Key Features
* **Modular RAG Pipeline:** Decoupled Ingestion and Retrieval logic for scalability.
* **Smart Chunking:** Powered by **LangChain** to preserve semantic context.
* **Memory-Safe Ingestion:** Implemented batch processing to prevent OOM crashes on large PDF datasets.
* **Precision Embedding:** Uses **BGE-M3** (FlagEmbedding) for high-accuracy retrieval.
* **Automated Benchmarking:** Auto-logging of inference time and responses to CSV for model comparison.

## 🛠 Tech Stack
* **Core:** Python 3.11+
* **LLM Runtime:** Ollama (Llama 3, Mistral, Granite)
* **Vector DB:** ChromaDB (Persistent)
* **Embedding:** BGE-M3 (BAAI)
* **Parsing:** PyMuPDF (fitz) & LangChain
* **Hardware Accel:** Native Apple Metal (MPS) support.

## 📂 Project Structure

```text
ChatrAIn_project/
│
├── 📂 src/                 # The Engine (Core Logic: Ingestion, Retrieval)
├── 📂 data/manuals/        # Input: Drop your PDFs here
├── 📂 db/chroma_db/        # Storage: Persistent Vector Database
├── 📂 csv_test/            # Logs: Benchmark results
├── 📂 notebooks/           # Lab: Prototyping and Data Analysis
│
├── main.py                 # 🎮 Entry Point: Interactive CLI
└── requirements.txt        # Dependencies

## How to use it

1. Prerequisites

Ensure you have Ollama installed and download the models:

ollama pull llama3 
ollama pull llama3.2 
ollama pull granite3-dense:2b

2. Installation

Clone the repository and install dependencies:

pip install -r requirements.txt

3. Startup

Open the notebook and follow the cells:

jupyter notebook ChatrAIn_bot_gen_ai_injection_vector_db_0.0.2.ipynb

After db creation ---> python main.py

* Project created by Raffaele Ciccarone for the Generative AI & LLM course held by IBM     

##Enjoy your local ChatrAIn assistant!##

-------------------------------------------------------------------------
# Copyright (c) 2025 Raffaele Ciccarone
# This code is the intellectual property of Raffaele Ciccarone.
# It is strictly prohibited to copy, distribute, or modify this file
# without distinct written permission from the owner.
# -------------------------------------------------------------------------
