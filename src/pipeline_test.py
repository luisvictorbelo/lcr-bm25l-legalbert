import pandas as pd
import numpy as np
import time
import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
from src.bm25_retriever import BM25Manager
from src.query_extractor.kli import KLIQueryExtractor
from src.query_extractor.keybert_extractor import KeyBERTQueryExtractor
from src.query_extractor.plm_extractor import PLMQueryExtractor
from src.query_extractor.tfidf_extractor import TFIDFQueryExtractor
from src.query_extractor.proposition_extractor import PropositionQueryExtractor
from src.query_extractor.marked_paragraph_extractor import MarkedParagraphQueryExtractor
import bm25s

def cosine_similarity(A, B):
    """
    Computes cosine similarity between two matrices A and B.
    A shape: (n_q_chunks, emb_dim)
    B shape: (n_c_chunks, emb_dim)
    Returns matrix of shape (n_q_chunks, n_c_chunks)
    """
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    A_norm = np.where(A_norm == 0, 1e-9, A_norm)
    A_normalized = A / A_norm
    
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    B_norm = np.where(B_norm == 0, 1e-9, B_norm)
    B_normalized = B / B_norm
    
    return np.dot(A_normalized, B_normalized.T)

def min_max_normalize(scores):
    scores = np.array(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def main():
    parser = argparse.ArgumentParser(description="Run the retrieval and reranking pipeline.")
    parser.add_argument("--query_method", choices=["KLI", "KeyBERT", "PLM", "TF-IDF", "Proposition", "MarkedParagraph"], default="KLI", help="Method for query extraction.")
    parser.add_argument("--propositions_file", type=str, default='data/test-files/processed/query_propositions_saul.parquet', help="Path to LLM-generated propositions file (Parquet).")
    parser.add_argument("--marked_paragraphs_file", type=str, default='data/test-files/processed/query_marked_paragraphs.parquet', help="Path to raw marked paragraphs file (Parquet).")
    parser.add_argument("--portion", type=float, default=0.5, help="Portion of top terms to select for the query.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 score in hybrid combination.")
    parser.add_argument("--top_k_bm25", type=int, default=100, help="Number of documents to retrieve via BM25.")
    parser.add_argument("--model_name", type=str, default="legal_bert", help="Name of the dense model used for reranking.")
    args = parser.parse_args()

    print(f"Configuration: method={args.query_method}, portion={args.portion}, alpha={args.alpha}")
    print("Loading data...")
    
    # Paths
    corpus_para_path = 'data/test-files/processed/corpus_paragraphs_all.parquet'
    query_para_path = 'data/test-files/processed/query_paragraphs_all.parquet'
    corpus_emb_path = 'data/test-files/processed/chunk_embeddings.parquet'
    query_emb_path = 'data/test-files/processed/query_chunk_embeddings.parquet'
    index_path = "data/indexes/bm25l_k1_3.5_b_1.0"

    # Verify all files exist
    for p in [corpus_para_path, query_para_path, corpus_emb_path, query_emb_path]:
        if not os.path.exists(p):
            print(f"Error: Required file not found: {p}")
            return
    if not os.path.exists(index_path):
        print(f"Error: BM25 Index not found at {index_path}")
        return

    df_corpus = pd.read_parquet(corpus_para_path)
    df_queries = pd.read_parquet(query_para_path)
    df_corpus_emb = pd.read_parquet(corpus_emb_path)
    df_query_emb = pd.read_parquet(query_emb_path)

    # Pick the query docs to test
    unique_query_ids = df_queries['doc_id'].unique()
    
    # Loop through all queries
    all_pipeline_results = {}
    
    print(f"\n--- Starting full pipeline for {len(unique_query_ids)} queries using {args.query_method} ---")
    
    # 1. Initialize Extractor
    if args.query_method == "KLI":
        print("Initializing KLI Extractor (Building background distribution)...")
        extractor = KLIQueryExtractor(df_corpus)
    elif args.query_method == "PLM":
        print("Initializing PLM Extractor (Building background distribution)...")
        extractor = PLMQueryExtractor(df_corpus)
    elif args.query_method == "TF-IDF":
        print("Initializing TF-IDF Extractor...")
        extractor = TFIDFQueryExtractor()
    elif args.query_method == "Proposition":
        print("Initializing Proposition Extractor...")
        extractor = PropositionQueryExtractor(args.propositions_file)
    elif args.query_method == "MarkedParagraph":
        print("Initializing Marked Paragraph Extractor...")
        extractor = MarkedParagraphQueryExtractor(args.marked_paragraphs_file)
    else:
        print("Initializing KeyBERT Extractor...")
        extractor = KeyBERTQueryExtractor()
    
    # 2. Load BM25 Index
    print("Loading BM25 index...")
    bm25_manager = BM25Manager()
    retriever, corpus_doc_ids = bm25_manager.load_index(index_path)

    for query_doc_id in tqdm(unique_query_ids, desc="Processing Queries"):
        # 1. Extract Query / Subqueries
        if args.query_method in ["Proposition", "MarkedParagraph"]:
            subqueries = extractor.get_subqueries(query_doc_id)
            if not subqueries:
                continue
            
            # Aggregate scores for all subqueries
            doc_scores_map = defaultdict(float)
            
            for subquery_text in subqueries:
                # Tokenize and retrieve
                query_tokens = bm25s.tokenize(subquery_text, stopwords='en')[0]
                # Use k=300 for subqueries as per research to ensure coverage
                sub_results, sub_scores = retriever.retrieve([list(query_tokens)], k=300)
                
                for i in range(sub_results.shape[1]):
                    idx = int(sub_results[0, i])
                    doc_id = corpus_doc_ids[idx]
                    score = float(sub_scores[0, i])
                    doc_scores_map[doc_id] += score
            
            # Rank and pick top K
            ranked_docs = sorted(doc_scores_map.items(), key=lambda x: x[1], reverse=True)
            candidate_doc_ids = [doc_id for doc_id, _ in ranked_docs[:args.top_k_bm25]]
            bm25_scores = [score for _, score in ranked_docs[:args.top_k_bm25]]
            
            # For logging/compatibility if needed
            bm25_results = [{'doc_id': doc_id, 'score': score} for doc_id, score in ranked_docs[:args.top_k_bm25]]

        else:
            query_df_subset = df_queries[df_queries['doc_id'] == query_doc_id]
            extracted_queries = extractor.extract_queries(query_df_subset, portion=args.portion)
            query_tokens = extracted_queries.get(query_doc_id, [])
            
            if not query_tokens:
                # print(f"Warning: No query tokens extracted for {query_doc_id}. Skipping.")
                continue

            # 2. BM25 Retrieval
            bm25_results_batch = bm25_manager.retrieve_topk(retriever, corpus_doc_ids, [query_tokens], k=args.top_k_bm25)
            bm25_results = bm25_results_batch[0]
            
            candidate_doc_ids = [res['doc_id'] for res in bm25_results]
            bm25_scores = [res['score'] for res in bm25_results]
        
        # 3. Dense Reranking
        q_embs_df = df_query_emb[df_query_emb['case_id'] == query_doc_id]
        if q_embs_df.empty:
            all_pipeline_results[query_doc_id] = [{"doc_id": r['doc_id'], "score": r['score']} for r in bm25_results]
            continue
            
        q_embs = np.stack(q_embs_df['embedding'].tolist())
        
        dense_scores = []
        for cand_id in candidate_doc_ids:
            cand_embs_df = df_corpus_emb[df_corpus_emb['case_id'] == cand_id]
            if cand_embs_df.empty:
                dense_scores.append(0.0)
                continue
            
            c_embs = np.stack(cand_embs_df['embedding'].tolist())
            sim_matrix = cosine_similarity(q_embs, c_embs)
            max_sim = np.max(sim_matrix)
            dense_scores.append(max_sim)

        # 4. Normalization and Combination
        bm25_norm = min_max_normalize(bm25_scores)
        dense_norm = min_max_normalize(dense_scores)
        
        final_scores = args.alpha * bm25_norm + (1 - args.alpha) * dense_norm
        
        # 5. Store results
        query_results = []
        for i, cand_id in enumerate(candidate_doc_ids):
            query_results.append({
                'doc_id': cand_id,
                'score': float(final_scores[i])
            })
        
        query_results.sort(key=lambda x: x['score'], reverse=True)
        all_pipeline_results[query_doc_id] = query_results

    # Save to JSON
    # Construct dynamic filename
    method_name = args.query_method.lower().replace('-', '_')
    # BM25 params from retriever
    bm25_info = f"{retriever.method}_k1_{retriever.k1}_b_{retriever.b}"
    
    # Optional portion for certain methods
    portion_str = f"_p{args.portion}" if args.query_method not in ["Proposition", "MarkedParagraph"] else ""
    
    dynamic_name = f"{method_name}{portion_str}_{bm25_info}_a{args.alpha}_{args.model_name.lower()}"
    output_dir = 'data/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    output_results_path = os.path.join(output_dir, f'results_{dynamic_name}.json')
    
    print(f"\nSaving all results to {output_results_path}...")
    with open(output_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_pipeline_results, f, indent=2)
    
    print("Full pipeline execution completed.")

if __name__ == "__main__":
    main()
