import math
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFQueryExtractor:
    """
    Extracts queries using the TF-IDF method.
    Ranks terms within the query document set using TfidfVectorizer and selects the top portion.
    """
    def __init__(self):
        pass

    def extract_queries(self, queries_df: pd.DataFrame, portion: float = 0.5) -> dict:
        """
        Extracts queries for each document in the queries_df using the TF-IDF method.
        
        Args:
            queries_df: A pandas DataFrame containing the query paragraphs.
                        Must have 'doc_id' and 'processed_tokens' columns.
            portion: The fraction of top terms to select.
            
        Returns:
            A dictionary mapping doc_id to a list of selected term strings.
        """
        if portion <= 0 or portion > 1:
            raise ValueError('portion must be in (0, 1].')

        print(f"Extracting queries using TF-IDF (portion={portion})...")
        
        # 1. Group paragraphs and reconstruct documents
        doc_ids = []
        docs_text = []
        
        # Grouping
        temp_map = {}
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
            
            if doc_id not in temp_map:
                temp_map[doc_id] = []
            temp_map[doc_id].extend(tokens)
            
        for d_id, d_tokens in temp_map.items():
            if not d_tokens:
                continue
            doc_ids.append(d_id)
            docs_text.append(' '.join(d_tokens))

        if not docs_text:
            print("Warning: No text found to calculate TF-IDF.")
            return {}

        # 2. Compute TF-IDF
        vectorizer = TfidfVectorizer(
            lowercase=False,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm='l2',
        )
        
        tfidf_matrix = vectorizer.fit_transform(docs_text)
        feature_names = np.array(vectorizer.get_feature_names_out())

        # 3. Select top terms
        top_terms_by_doc = {}
        for i, doc_id in enumerate(doc_ids):
            row = tfidf_matrix[i].toarray().ravel()
            
            # Keep terms with positive scores
            non_zero_idx = np.where(row > 0)[0]
            if non_zero_idx.size == 0:
                top_terms_by_doc[doc_id] = []
                continue
                
            # Sort by score descending
            sorted_idx = non_zero_idx[np.argsort(row[non_zero_idx])[::-1]]
            
            # Select top K
            n_selected = max(1, int(math.ceil(len(sorted_idx) * portion)))
            selected_idx = sorted_idx[:n_selected]
            
            top_terms_by_doc[doc_id] = [str(feature_names[j]) for j in selected_idx]

        print(f"TF-IDF queries extracted for {len(top_terms_by_doc)} documents.")
        return top_terms_by_doc
