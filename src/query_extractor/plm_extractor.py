import math
from collections import Counter
import pandas as pd
import numpy as np
import json

class PLMQueryExtractor:
    """
    Extracts queries using the Parzen Window Language Model (PLM) method.
    Uses Expectation-Maximization (EM) to estimate a document-specific language model
    by de-noising it against a background corpus distribution.
    """
    def __init__(self, corpus_df: pd.DataFrame):
        """
        Initializes the background distribution using the corpus.
        
        Args:
            corpus_df: A pandas DataFrame containing the corpus. 
                       Must have a 'processed_tokens' column.
        """
        print("Building background distribution for PLM...")
        self.bg_counter = Counter()
        
        for _, row in corpus_df.iterrows():
            tokens = row.get('processed_tokens', [])
            if isinstance(tokens, (list, np.ndarray)):
                self.bg_counter.update(list(tokens))
            elif isinstance(tokens, str):
                try:
                    tokens_list = json.loads(tokens.replace("'", '"'))
                    self.bg_counter.update(tokens_list)
                except:
                    self.bg_counter.update([tokens])
                    
        self.bg_total_terms = sum(self.bg_counter.values())
        if self.bg_total_terms == 0:
            raise ValueError('Corpus has no processed terms.')
            
        self.bg_prob = {term: freq / self.bg_total_terms for term, freq in self.bg_counter.items()}
        print(f"Background distribution built with {len(self.bg_prob)} unique terms.")

    def run_plm_em(self, doc_counter, plm_lambda: float, max_iters: int, eps: float) -> dict:
        """
        Executes the EM algorithm for a single document.
        """
        total_doc_terms = sum(doc_counter.values())
        if total_doc_terms == 0:
            return {}

        # Initialization with MLE: P(t|Qd)
        p_t_qd = {t: freq / total_doc_terms for t, freq in doc_counter.items()}

        for _ in range(max_iters):
            # E-step
            e_t = {}
            for t, tf_t in doc_counter.items():
                p_t_d = p_t_qd.get(t, 0.0)
                p_t_c = self.bg_prob.get(t, 1.0 / self.bg_total_terms)

                denom = (1 - plm_lambda) * p_t_c + plm_lambda * p_t_d
                if denom <= 0:
                    e_t[t] = 0.0
                else:
                    e_t[t] = tf_t * (plm_lambda * p_t_d) / denom

            # M-step
            e_sum = sum(e_t.values())
            if e_sum <= 0:
                break

            new_p_t_qd = {t: e_t[t] / e_sum for t in e_t}

            # Check for convergence
            delta = sum(abs(new_p_t_qd[t] - p_t_qd.get(t, 0.0)) for t in new_p_t_qd)
            p_t_qd = new_p_t_qd
            
            if delta < eps:
                break

        return p_t_qd

    def extract_queries(
        self, 
        queries_df: pd.DataFrame, 
        portion: float = 0.5, 
        plm_lambda: float = 0.5, 
        max_iters: int = 30, 
        eps: float = 1e-9
    ) -> dict:
        """
        Extracts queries for each document in the queries_df using the PLM method.
        
        Args:
            queries_df: A pandas DataFrame containing the query paragraphs.
                        Must have 'doc_id' and 'processed_tokens' columns.
            portion: The fraction of top terms to select.
            plm_lambda: Lambda parameter for PLM (weight of document model).
            max_iters: Maximum iterations for EM.
            eps: Convergence threshold.
            
        Returns:
            A dictionary mapping doc_id to a list of selected term strings.
        """
        if portion <= 0 or portion > 1:
            raise ValueError('portion must be in (0, 1].')

        print(f"Extracting queries using PLM (portion={portion}, lambda={plm_lambda})...")
        
        # Group paragraphs by doc_id
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
            
            p_t_qd = self.run_plm_em(
                doc_counter=doc_counter,
                plm_lambda=plm_lambda,
                max_iters=max_iters,
                eps=eps
            )

            sorted_terms = sorted(p_t_qd.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_terms:
                top_terms_by_doc[doc_id] = []
                continue

            n_selected = max(1, int(math.ceil(len(sorted_terms) * portion)))
            top_terms_by_doc[doc_id] = [t for t, _ in sorted_terms[:n_selected]]

        print(f"PLM queries extracted for {len(top_terms_by_doc)} documents.")
        return top_terms_by_doc
