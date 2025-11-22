# ChatrAIn      

/Users/raffaeleciccarone/Desktop/ASSISTENTE MANUTENTORE/1.jpg     

> In the first file, you can find the code to populate the vector DB and the first version of the chatbot.

> In the second version, after the creation of the vector DB, you will find the final and most stable version.

## Project Description

This project implements an intelligent assistant designed to support railway maintenance workers. Unlike generic chatbots, ChatrAIn answers technical questions based exclusively on the Maintenance Manual used to create the vector DB.

The entire architecture is designed to run offline (Air-Gapped) on consumer hardware (Apple Silicon M1/M2), ensuring:

* Total Privacy: 
No data is sent to cloud servers.

* Zero Hallucinations: 
The model is constrained to answer only if it finds evidence in the manuals.

## Key Features 

Advanced Ingestion: Technical PDF parsing with text cleaning and semantic Chunking.

Embedding: Utilization of the BGE-M3 model for high-precision vectorization.

* Optimized Inference Engine: 
Integration with Ollama (Llama 3.2 / Granite) with response streaming.

* Benchmarking: 
Automatic logging system on Pandas to compare latency and response quality across different models.   

/Users/raffaeleciccarone/Desktop/ASSISTENTE MANUTENTORE/2.jpg   

## Technologies Used

Language: Python 3.11+

LLM Runtime: Ollama (Local)

Vector Database: ChromaDB (Persistent)

Embedding Model: FlagEmbedding (BAAI/bge-m3)

PDF Parsing: PyMuPDF (fitz)

Data Analysis: Pandas, Tqdm

Hardware Acceleration: Native support for Apple Metal (MPS)

## File Structure

* ChatrAIn_assistant_v1.ipynb: 
The main notebook with the entire RAG pipeline.

* '...'.pdf: 
The technical manual (Knowledge Base).

* requirements.txt: 
List of Python dependencies.

* test_risposte.csv: 
Performance log (Latency/Responses) generated automatically.

* chroma_db_data/: 
Folder containing the persistent vector database.

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

jupyter notebook ChatrAIn_assistant_v1.ipynb

* Project created by Raffaele Ciccarone for the Generative AI & LLM course held by IBM     


/Users/raffaeleciccarone/Desktop/ASSISTENTE MANUTENTORE/3.jpg
##Enjoy your local ChatrAIn assistant!##
