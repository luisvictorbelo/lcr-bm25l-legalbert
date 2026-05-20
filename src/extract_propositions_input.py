import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.preprocess import LegalDocumentPreprocessor

def main():
    # Paths
    DOCUMENTS_FOLDER = 'data/test-files/docs'
    LABELS_FILE = 'data/labels/task1_test_labels_2025.json'
    OUTPUT_FILE = 'data/test-files/processed/query_marked_paragraphs.parquet'
    
    print(f"Loading labels from {LABELS_FILE}...")
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # Normalize query IDs
    query_ids = {k.strip().replace('.txt', '') for k in labels.keys()}
    print(f"Found {len(query_ids)} query IDs.")

    folder = Path(DOCUMENTS_FOLDER)
    if not folder.exists():
        print(f"Error: Folder {DOCUMENTS_FOLDER} not found.")
        return

    files = sorted(list(folder.glob('*.txt')))
    print(f"Processing {len(files)} files to find query documents...")

    preprocessor = LegalDocumentPreprocessor(use_spacy=True)
    all_marked_paragraphs = []

    for file_path in tqdm(files, desc="Extracting marked paragraphs"):
        doc_id = file_path.stem
        if doc_id not in query_ids:
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Process document to extract ONLY paragraphs with markers
            # Note: process_document in src/preprocess.py has an only_marked parameter
            # but it combines with subsequent text. 
            # The notebook extract_marked_paragraphs.ipynb had extract_paragraphs_with_markers_only
            # which we should probably replicate if we want just the marked paragraph.
            
            # For now, let's use the preprocessor's existing process_document with only_marked=True
            # which returns ProcessedParagraph objects.
            paragraphs = preprocessor.process_document(
                doc_id, 
                content, 
                filter_french=True, 
                only_marked=True
            )
            
            for p in paragraphs:
                all_marked_paragraphs.append(p.to_dict())
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_marked_paragraphs:
        print("No marked paragraphs found.")
        return

    print(f"Extracted {len(all_marked_paragraphs)} marked paragraphs.")
    df = pd.DataFrame(all_marked_paragraphs)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == '__main__':
    main()
