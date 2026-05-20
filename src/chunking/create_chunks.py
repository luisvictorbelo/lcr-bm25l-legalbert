import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import argparse

def chunk_document(doc_id, paragraphs, tokenizer, chunk_size=384, stride=192, min_chunk=80):
    # Join paragraphs with spaces to create a single document text
    full_text = " ".join(paragraphs)
    
    # Tokenize the entire document
    # We use add_special_tokens=False because we want to manage them ourselves (CLS/SEP)
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    chunks = []
    if not tokens:
        return chunks
        
    # Sliding window parameters
    # effective_chunk_size is the space available for tokens (384 - 2 special tokens)
    eff_size = chunk_size - 2
    
    start = 0
    while start < len(tokens):
        end = start + eff_size
        chunk_tokens = tokens[start:end]
        
        # Check min_chunk for the last chunk
        if len(chunk_tokens) < min_chunk and len(chunks) > 0:
            # If the last chunk is too small, we skip it unless it's the only one
            break
            
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        
        chunks.append({
            "chunk_id": f"{doc_id}_{len(chunks):03d}",
            "case_id": doc_id,
            "text": chunk_text
        })
        
        if end >= len(tokens):
            break
            
        # Move start by (eff_size - stride)
        start += (eff_size - stride)
        
    return chunks

def load_data(file_path):
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path).to_dict('records')
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Chunk documents into smaller pieces.")
    parser.add_argument("--inputs", nargs="+", help="Input files (JSON or Parquet).")
    parser.add_argument("--output", default="data/processed/chunks.parquet", help="Output Parquet file.")
    args = parser.parse_args()

    # Default values if no args provided (for legacy compatibility)
    if not args.inputs:
        args.inputs = [
            'data/processed/paragraphs/corpus_paragraphs_all.json',
            'data/processed/paragraphs/query_paragraphs_all.json'
        ]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    
    all_chunks = []
    
    for file_path in args.inputs:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        print(f"Processing {file_path}...")
        data = load_data(file_path)
        
        # Group by doc_id while preserving order
        docs = {}
        doc_order = []
        for item in data:
            doc_id = item['doc_id']
            if doc_id not in docs:
                docs[doc_id] = []
                doc_order.append(doc_id)
            # Use 'cleaned_text' as in the preprocessor output
            docs[doc_id].append(item.get('cleaned_text', ''))
            
        for doc_id in tqdm(doc_order):
            paragraphs = docs[doc_id]
            all_chunks.extend(chunk_document(doc_id, paragraphs, tokenizer))
            
    if not all_chunks:
        print("No chunks generated.")
        return

    print(f"Creating DataFrame for {len(all_chunks)} chunks...")
    df = pd.DataFrame(all_chunks)
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving to {args.output}...")
    df.to_parquet(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
