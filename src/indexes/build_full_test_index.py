import pandas as pd
import os
from src.bm25_retriever import BM25Manager

def main():
    corpus_path = 'data/test-files/processed/corpus_paragraphs_all.parquet'
    save_dir = 'data/indexes/bm25l_k1_3.5_b_1.0'
    
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found.")
        return

    print(f"Loading full corpus from {corpus_path}...")
    df_corpus = pd.read_parquet(corpus_path)
    print(f"Total paragraphs: {len(df_corpus)}")
    
    manager = BM25Manager()
    retriever, corpus_doc_ids = manager.build_index(df_corpus, method='bm25l', k1=3.5, b=1.0)
    
    print(f"Saving full index to {save_dir}...")
    manager.save_index(retriever, corpus_doc_ids, save_dir)
    print("Full index built and saved successfully.")

if __name__ == "__main__":
    main()
