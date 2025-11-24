# -------------------------------------------------------------------------
# Copyright (c) 2025 Raffaele Ciccarone
# This code is the intellectual property of Raffaele Ciccarone.
# It is strictly prohibited to copy, distribute, or modify this file
# without distinct written permission from the owner.
# -------------------------------------------------------------------------

import os
import glob
import fitz #to import pdf
import re
import chromadb #for vectordb
import time
import torch
import ollama
import pandas as pd
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter



def device():
    if torch.backends.mps.is_available():
        device = 'mps'
        print('mps ready')
    else:       
        device = 'cpu'
        print('=( no mps')
    return device

def load_embedding_model():
    this_device = device()
    if this_device == 'mps':
        use_fp16 = True
    else:
        use_fp16 = False
    print(f'{this_device} : {use_fp16}')

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=use_fp16, devices = this_device)
    print("Embedding model: BGE-M3")
    print("Ready!")
    return model

def path_extraction(path_manuals):
    #the return is not a sorted list
    return glob.glob(os.path.join(path_manuals, '*.pdf')) #if you want: sorted(glob.gl.....)

def cleaning(text):
    #minimal pre-cleaning
    text = text.replace('\x00', '') #ok for db
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def chunkin(text):
    clean_text = cleaning(text)
    #langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len, 
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    
    return text_splitter.split_text(clean_text)

def text_preprocessing(path_manuals):
    pdf_files = path_extraction(path_manuals)
    all_chunks = []
    all_metadatas = []

    print(f"Processing {len(pdf_files)} files...")

    for pdf_path in tqdm(pdf_files, desc="Lettura PDF"):
        full_text = ''
        file_name = os.path.basename(pdf_path)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text()
            #BLOCK CHUNCKING
            doc_chunks = chunkin(full_text)
            #METADATA SAVE
        for c in doc_chunks:
            all_chunks.append(c)
            all_metadatas.append({'source': file_name})
            #lock loop and divide documents
    

    if len(all_chunks) != len(all_metadatas):
        print(f"WARNING! Chunks: {len(all_chunks)}, Metas: {len(all_metadatas)}")
        return [], []

    print(f"Extraction {len(all_chunks)} total chunks from {len(pdf_files)} files COMPLETE.")
    return all_chunks, all_metadatas

def create_vector_db(db_path, collec_name, embedding_model, chunks, metadatas, batch_size=500):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collec_name)

    tot_chunk_n = len(chunks)
    print(f"Generating embeddings and saving {tot_chunk_n} chunks...")

    #process chunks in batches for Out Of Memory
    for i in range(0, tot_chunk_n, batch_size):
        
        end_index = min(i + batch_size, tot_chunk_n)

        batch_chunks = chunks[i : end_index]
        batch_metas = metadatas[i : end_index]
        batch_ids = [f"id_{k}" for k in range(i, end_index)] # generate unique id
        
        encoded_output = embedding_model.encode(batch_chunks)
        batch_embeddings = encoded_output['dense_vecs']

        # Append to the database
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_chunks,
            metadatas=batch_metas
        )
        
    print(f"Done! {len(chunks)} vettors saved into: '{db_path}'.")  

def retrieve_documents(query, embedding_model, db_path, collection_name, n_results=3):
    print(f"\nUser query: '{query}'")

    client = chromadb.PersistentClient(path=db_path)
    db_collection = client.get_collection(name=collection_name)

    encoded_output = embedding_model.encode(query)
    if isinstance(encoded_output, dict):
        query_vector = encoded_output['dense_vecs']
    else:
        query_vector = encoded_output

    # WARNING: query_embeddings want a list
    # So: [query_vector]
    results = db_collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"] 
    )

    print(f"\n=== TOP {n_results} RESULTS ===")
    
    retrieved_docs = results['documents'][0]
    retrieved_metas = results['metadatas'][0]
    retrieved_dists = results['distances'][0]

    for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metas, retrieved_dists)):
        print(f"\nCIUUUF CIUUUUUUF n. #{i+1} (distance: {dist:.4f})")
        print(f"From: {meta.get('source', 'Unknown')}") 
        print(f"Chunk text: {doc[:200]}...") 
        print("=D" * 50)

    # Entire pack for llm
    return results

def add_row_csv(m, elapsed_time, query_text, response,csv_path):

    new_row = {
        'model': m,
        'time': round(elapsed_time, 2),
        'query': query_text,   
        'response': response
    }

    df_new = pd.DataFrame([new_row])

    if not os.path.exists(csv_path):
        df_new.to_csv(csv_path, index=False)
        print(f"New log in: {csv_path}")
    else:
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
        print('Row added to csv log!')


def final_test(query_text, model, db_path, collection_name, model_list, csv_path, n_results=5):
    #global df_risposte
    print("="*50)
    print(f"Il tuo aiutante manutentore sta pensando: '{query_text}'...\n\nCIUUUUUF CIUUUUUUF!")

    results = retrieve_documents(query_text, model, db_path, collection_name, n_results=n_results)
    retrieved_chunks = results['documents'][0]
    context_text = "\n\n---\n\n".join(retrieved_chunks)

    #define model prompt with summary of all retrieved chunks plus user query
    prompt_template = f"""
    Sei un esperto manutentore ferroviario specializzato in manualistica. Basati ESCLUSIVAMENTE sul contesto fornito per rispondere alle domande.
    NON inventare procedure! Se l'informazione non è nel contesto dì che non hai la risposta.

    CONTESTO ESTRATTO DAL MANUALE:
    {context_text}

    DOMANDA UTENTE:
    {query_text}

    RISPOSTA (in italiano tecnico e preciso):
    """
    for m in model_list:
        print(f"\n\n=== MODEL: '{m}' ===")
        #GOOOOOOOOOOOOOOO!!!
        start_time = time.time()

        stream = ollama.chat(model=m, messages=[
            {'role': 'user', 'content': prompt_template},
        ], stream=True)

        response = ''

        for piece in tqdm(stream, desc=f" {m} sta scrivendo...", unit=" token"):
            text_part = piece['message']['content']
            response += text_part

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Tempo di inferenza: {elapsed_time:.2f} secondi')

        add_row_csv(m, elapsed_time, query_text, response, csv_path)

        print(f"""
{'=D'*50}\nRESPONSE:\n{response}
""")