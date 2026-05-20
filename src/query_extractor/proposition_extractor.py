import pandas as pd
import os
from typing import List, Dict

class PropositionQueryExtractor:
    """
    Loads LLM-generated propositions and provides them as subqueries for retrieval.
    """
    def __init__(self, propositions_path: str):
        """
        Args:
            propositions_path: Path to the Parquet file containing generated propositions.
        """
        if not os.path.exists(propositions_path):
            raise FileNotFoundError(f"Propositions file not found at {propositions_path}")
        
        print(f"Loading propositions from {propositions_path}...")
        self.df = pd.read_parquet(propositions_path)
        
        # Group by doc_id for fast lookup
        self.propositions_by_doc = {}
        for doc_id, group in self.df.groupby('doc_id'):
            # Ensure doc_id is string and cleaned
            clean_id = str(doc_id).strip().replace('.txt', '')
            self.propositions_by_doc[clean_id] = group['proposition'].tolist()
            
        print(f"Loaded propositions for {len(self.propositions_by_doc)} documents.")

    def get_subqueries(self, doc_id: str) -> List[str]:
        """
        Returns a list of propositions (subqueries) for a given document.
        """
        clean_id = str(doc_id).strip().replace('.txt', '')
        return self.propositions_by_doc.get(clean_id, [])

    def get_all_doc_ids(self) -> List[str]:
        """
        Returns all document IDs for which propositions are available.
        """
        return list(self.propositions_by_doc.keys())
