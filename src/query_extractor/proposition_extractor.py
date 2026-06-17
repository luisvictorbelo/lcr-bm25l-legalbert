import pandas as pd
import os
import re
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
            # Sanitize each proposition
            propositions = [self._sanitize(p) for p in group['proposition'].tolist() if p]
            self.propositions_by_doc[clean_id] = propositions

        print(f"Loaded propositions for {len(self.propositions_by_doc)} documents.")

    def _sanitize(self, text: str) -> str:
        """
        Removes residual LLM template tags and artifacts.
        """
        if not isinstance(text, str):
            return ""

        # Remove common instruction tags
        tags = [
            r'\[/INST\]', 
            r'\[INST\]', 
            r'### Assistant:', 
            r'### System:', 
            r'### User:'
        ]

        cleaned = text
        for tag in tags:
            cleaned = re.sub(tag, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

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
