# -------------------------------------------------------------------------
# Copyright (c) 2025 Raffaele Ciccarone
# This code is the intellectual property of Raffaele Ciccarone.
# It is strictly prohibited to copy, distribute, or modify this file
# without distinct written permission from the owner.
# -------------------------------------------------------------------------

import os
import sys
from src.rag_engine import (
        load_embedding_model,
        final_test
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "db", "chroma_db")
COLLECTION_NAME = "manuali_manutenzione_test"

CSV_LOG_DIR = os.path.join(BASE_DIR, "csv_test")
CSV_LOG_FILE = os.path.join(CSV_LOG_DIR, "benchmark_log.csv")


def main():
    print(f"\n--- TEST ---")
    
    model = load_embedding_model()

    llm_models = ['llama3', 'llama3.2'] 

    query = input('\nInserisci la domanda per il test (es. "Come smonto il pantografo?"): ')
    
    if not query.strip():
        print('!QUERY NULL!')
        sys.exit(0) 
        
    final_test(
        query_text=query,
        model=model,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        model_list=llm_models,
        csv_path=CSV_LOG_FILE,
        n_results=1
    )
    print(f"\nTest completed. Log: {CSV_LOG_FILE}")


if __name__ == "__main__":
    main()