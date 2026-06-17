import pandas as pd
import os
from typing import List

class MarkedParagraphQueryExtractor:
    """
    Loads raw marked paragraphs and provides them as subqueries for retrieval.
    """
    def __init__(self, paragraphs_path: str):
        """
        Args:
            paragraphs_path: Path to the Parquet file containing marked paragraphs.
        """
        if not os.path.exists(paragraphs_path):
            raise FileNotFoundError(f"Paragraphs file not found at {paragraphs_path}")
        
        print(f"Loading marked paragraphs from {paragraphs_path}...")
        self.df = pd.read_parquet(paragraphs_path)
        
        # Group by doc_id for fast lookup
        self.paragraphs_by_doc = {}
        for doc_id, group in self.df.groupby('doc_id'):
            # Ensure doc_id is string and cleaned
            clean_id = str(doc_id).strip().replace('.txt', '')
            self.paragraphs_by_doc[clean_id] = group['cleaned_text'].tolist()
            
        print(f"Loaded marked paragraphs for {len(self.paragraphs_by_doc)} documents.")

    def get_subqueries(self, doc_id: str) -> List[str]:
        """
        Returns a list of marked paragraphs (subqueries) for a given document.
        """
        clean_id = str(doc_id).strip().replace('.txt', '')
        return self.paragraphs_by_doc.get(clean_id, [])

    def get_all_doc_ids(self) -> List[str]:
        """
        Returns all document IDs for which marked paragraphs are available.
        """
        return list(self.paragraphs_by_doc.keys())
