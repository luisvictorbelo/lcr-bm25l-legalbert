import math
import pandas as pd
from tqdm import tqdm
try:
    from keybert import KeyBERT
except ImportError:
    KeyBERT = None

class KeyBERTQueryExtractor:
    """
    Extracts queries using the KeyBERT method.
    For each paragraph in a query document, it extracts keywords using KeyBERT.
    Then, it aggregates keywords for the whole document, taking the maximum similarity score 
    for repeating keywords, and selects the top portion.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if KeyBERT is None:
            raise ImportError("The 'keybert' library is not installed. Please install it with: pip install keybert")
        print(f"Initializing KeyBERT model: {model_name}...")
        self.kw_model = KeyBERT(model=model_name)

    def extract_queries(
        self, 
        queries_df: pd.DataFrame, 
        portion: float = 0.5,
        top_n_per_paragraph: int = 5,
        diversity: float = 0.6
    ) -> dict:
        """
        Extracts queries for each document in the queries_df using KeyBERT.
        
        Args:
            queries_df: A pandas DataFrame containing the query paragraphs.
                        Must have 'doc_id' and 'cleaned_text' columns.
            portion: The fraction of top terms to select (e.g., 0.5 for top 50%).
            top_n_per_paragraph: Number of keywords to extract per paragraph.
            diversity: Diversity parameter for MMR in KeyBERT.
            
        Returns:
            A dictionary mapping doc_id to a list of selected keyword strings.
        """
        if portion <= 0 or portion > 1:
            raise ValueError('portion must be in (0, 1].')

        print(f"Extracting queries using KeyBERT (portion={portion}, top_n={top_n_per_paragraph}, diversity={diversity})...")
        
        # Group paragraphs by doc_id
        doc_paragraphs_map = {}
        for _, row in queries_df.iterrows():
            doc_id = row['doc_id']
            text = row.get('cleaned_text', '')
            if not isinstance(text, str) or not text.strip():
                continue
                
            if doc_id not in doc_paragraphs_map:
                doc_paragraphs_map[doc_id] = []
            doc_paragraphs_map[doc_id].append(text)

        top_terms_by_doc = {}
        for doc_id, paragraphs in tqdm(doc_paragraphs_map.items(), desc='KeyBERT: processing query docs'):
            best_keyword_score = {}
            
            # Extract keywords for each paragraph
            for text in paragraphs:
                # KeyBERT might fail on extremely short text or if there are no alpha chars
                try:
                    keywords = self.kw_model.extract_keywords(
                        text,
                        stop_words='english',
                        top_n=top_n_per_paragraph,
                        diversity=diversity,
                        use_mmr=True
                    )
                except Exception as e:
                    # print(f"Warning: KeyBERT failed on paragraph in {doc_id}: {e}")
                    continue
                
                # Update best scores
                for keyword, score in keywords:
                    normalized = keyword.strip().lower()
                    if not normalized:
                        continue
                        
                    if normalized not in best_keyword_score or score > best_keyword_score[normalized]:
                        best_keyword_score[normalized] = float(score)

            # Sort keywords by max score descending
            sorted_keywords = sorted(
                best_keyword_score.items(),
                key=lambda item: item[1],
                reverse=True
            )
            
            # Select top portion
            if not sorted_keywords:
                top_terms_by_doc[doc_id] = []
                continue
                
            n_selected = max(1, int(math.ceil(len(sorted_keywords) * portion)))
            selected_terms = [kw for kw, _ in sorted_keywords[:n_selected]]
            top_terms_by_doc[doc_id] = selected_terms

        print(f"KeyBERT queries extracted for {len(top_terms_by_doc)} documents.")
        return top_terms_by_doc
