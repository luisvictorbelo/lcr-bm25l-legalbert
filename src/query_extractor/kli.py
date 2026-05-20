import math
from collections import Counter
import pandas as pd
import numpy as np
import json
from scipy.special import kl_div

class KLIQueryExtractor:
    """
    Extracts queries using the Kullback-Leibler Information (KLI) method.
    It builds a background distribution from the corpus, then for each query document,
    it computes the KLI score for its terms and selects the top portion.
    """
    def __init__(self, corpus_df: pd.DataFrame):
        """
        Initializes the background distribution using the corpus.
        
        Args:
            corpus_df: A pandas DataFrame containing the corpus. 
                       Must have a 'processed_tokens' column.
        """
        print("Building background distribution from corpus...")
        self.background_counter = Counter()
        
        for _, row in corpus_df.iterrows():
            tokens = row.get('processed_tokens', [])
            if isinstance(tokens, (list, np.ndarray)):
                self.background_counter.update(list(tokens))
            elif isinstance(tokens, str):
                try:
                    tokens_list = json.loads(tokens.replace("'", '"'))
                    self.background_counter.update(tokens_list)
                except:
                    self.background_counter.update([tokens])
                    
        self.total_background_terms = sum(self.background_counter.values())
        if self.total_background_terms == 0:
            raise ValueError('Corpus has no processed terms.')
            
        print(f"Background distribution built with {len(self.background_counter)} unique terms.")

    def extract_queries(self, queries_df: pd.DataFrame, portion: float = 0.5) -> dict:
        """
        Extracts queries for each document in the queries_df using the KLI method.
        
        Args:
            queries_df: A pandas DataFrame containing the query paragraphs.
                        Must have 'doc_id' and 'processed_tokens' columns.
            portion: The fraction of top terms to select (e.g., 0.5 for top 50%).
            
        Returns:
            A dictionary mapping doc_id to a list of selected term strings.
        """
        if portion <= 0 or portion > 1:
            raise ValueError('portion must be in (0, 1].')

        print(f"Extracting queries using KLI (portion={portion})...")
        
        # Group paragraphs by doc_id to reconstruct the full query document
        doc_tokens_map = {}
        for _, row in queries_df.iterrows():
            doc_id = row['doc_id']
            tokens = row.get('processed_tokens', [])
            if isinstance(tokens, (list, np.ndarray)):
                tokens = list(tokens)
            elif isinstance(tokens, str):
                try:
                    tokens = json.loads(tokens.replace("'", '"'))
                except:
                    tokens = [tokens]
            
            if doc_id not in doc_tokens_map:
                doc_tokens_map[doc_id] = []
            doc_tokens_map[doc_id].extend(tokens)

        top_terms_by_doc = {}
        for doc_id, doc_tokens in doc_tokens_map.items():
            doc_counter = Counter(doc_tokens)
            total_doc_terms = sum(doc_counter.values())
            
            if total_doc_terms == 0:
                top_terms_by_doc[doc_id] = []
                continue

            term_kli_scores = []
            for term, freq_doc in doc_counter.items():
                p_t_d = freq_doc / total_doc_terms
                
                if term in self.background_counter:
                    p_t_c = self.background_counter[term] / self.total_background_terms
                else:
                    p_t_c = 1.0 / self.total_background_terms
                    
                # KLI(t) = P(t|D) * log(P(t|D) / P(t|C))
                # Using scipy.special.kl_div for the KL divergence of a single term
                kli_score = float(kl_div(p_t_d, p_t_c))
                term_kli_scores.append((term, kli_score))
                
            # Sort terms by KLI score descending
            term_kli_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top K terms based on portion
            k_terms = max(1, int(math.ceil(len(term_kli_scores) * portion)))
            selected_terms = [term for term, _ in term_kli_scores[:k_terms]]
            top_terms_by_doc[doc_id] = selected_terms

        print(f"KLI queries extracted for {len(top_terms_by_doc)} documents.")
        return top_terms_by_doc
