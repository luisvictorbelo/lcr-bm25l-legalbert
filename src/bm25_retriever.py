import os
import json
import bm25s
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any

class BM25Manager:
    """
    Manages building, saving, loading, and querying BM25 indices using the bm25s library.
    Aggregates paragraph-level tokens into document-level tokens for indexing.
    """

    @staticmethod
    def build_index(corpus_df: pd.DataFrame, method: str = 'bm25l', k1: float = 3.5, b: float = 1.0) -> Tuple[bm25s.BM25, List[str]]:
        """
        Builds a BM25 index from a DataFrame of paragraphs.
        Aggregates tokens by doc_id.
        """
        print(f"Building BM25 index using method={method}, k1={k1}, b={b}...")
        
        doc_tokens_map = defaultdict(list)
        
        # Aggregate processed_tokens by doc_id
        # Expecting 'doc_id' and 'processed_tokens' columns
        for _, row in corpus_df.iterrows():
            doc_id = row['doc_id']
            # processed_tokens is expected to be a list of strings
            tokens = row.get('processed_tokens', [])
            if isinstance(tokens, (list, np.ndarray)):
                doc_tokens_map[doc_id].extend(list(tokens))
            elif isinstance(tokens, str):
                # Fallback if it was stored as a string representation of a list
                try:
                    tokens_list = json.loads(tokens.replace("'", '"'))
                    doc_tokens_map[doc_id].extend(tokens_list)
                except:
                    doc_tokens_map[doc_id].append(tokens)

        corpus_doc_ids = list(doc_tokens_map.keys())
        corpus_tokens = [doc_tokens_map[doc_id] for doc_id in corpus_doc_ids]

        # Initialize and index
        retriever = bm25s.BM25(method=method, k1=k1, b=b)
        retriever.index(corpus_tokens)

        print(f"Indexed {len(corpus_doc_ids)} documents.")
        return retriever, corpus_doc_ids

    @staticmethod
    def save_index(retriever: bm25s.BM25, corpus_doc_ids: List[str], save_dir: str, config_metadata: Dict[str, Any] = None):
        """
        Persists the BM25 index, doc_id mapping, and metadata to disk.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save bm25s native index files into an 'index' subdirectory
        index_path = os.path.join(save_dir, "index")
        retriever.save(index_path)
        
        # Save corpus_doc_ids mapping
        ids_path = os.path.join(save_dir, "corpus_doc_ids.json")
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_doc_ids, f)
            
        # Save metadata config
        config = {
            "method": retriever.method,
            "k1": retriever.k1,
            "b": retriever.b,
            "corpus_size": len(corpus_doc_ids)
        }
        if config_metadata:
            config.update(config_metadata)
            
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            
        print(f"Index and metadata saved to {save_dir}")

    @staticmethod
    def load_index(save_dir: str, mmap: bool = True) -> Tuple[bm25s.BM25, List[str]]:
        """
        Loads a persisted BM25 index and doc_id mapping from disk.
        """
        index_path = os.path.join(save_dir, "index")
        ids_path = os.path.join(save_dir, "corpus_doc_ids.json")
        
        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            raise FileNotFoundError(f"Required index files not found in {save_dir}")
            
        # Load retriever
        retriever = bm25s.BM25.load(index_path, load_corpus=False, mmap=mmap)
        
        # Load doc_ids
        with open(ids_path, 'r', encoding='utf-8') as f:
            corpus_doc_ids = json.load(f)
            
        print(f"Loaded BM25 index from {save_dir} ({len(corpus_doc_ids)} docs).")
        return retriever, corpus_doc_ids

    @staticmethod
    def retrieve_topk(retriever: bm25s.BM25, corpus_doc_ids: List[str], queries: List[List[str]], k: int = 100) -> List[List[Dict[str, Any]]]:
        """
        Performs retrieval for one or more queries and returns results mapped to doc_ids.
        
        Args:
            retriever: The loaded BM25 index.
            corpus_doc_ids: The mapping of internal indices to doc_ids.
            queries: A list of queries, where each query is a list of tokens.
                    Example: [["token1", "token2"], ["token3"]]
            k: Top-k results to return per query.
            
        Returns:
            A list of lists of result dictionaries.
        """
        # Ensure queries is a list of lists
        if not queries:
            return []
            
        # Check if it's a single query passed as a list of strings
        if isinstance(queries[0], str):
            queries = [queries]
            
        results, scores = retriever.retrieve(queries, k=k)
        
        # results shape is (n_queries, k)
        all_results = []
        for q_idx in range(results.shape[0]):
            indices = results[q_idx]
            sim_scores = scores[q_idx]
            
            query_results = []
            for idx, score in zip(indices, sim_scores):
                query_results.append({
                    "doc_id": corpus_doc_ids[idx],
                    "score": float(score)
                })
            all_results.append(query_results)
            
        return all_results
