import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os

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

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    
    input_files = [
        'data/processed/paragraphs/corpus_paragraphs_all.json',
        'data/processed/paragraphs/query_paragraphs_all.json'
    ]
    
    all_chunks = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        print(f"Processing {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Group by doc_id while preserving order
        docs = {}
        doc_order = []
        for item in data:
            doc_id = item['doc_id']
            if doc_id not in docs:
                docs[doc_id] = []
                doc_order.append(doc_id)
            docs[doc_id].append(item['cleaned_text'])
            
        for doc_id in tqdm(doc_order):
            paragraphs = docs[doc_id]
            all_chunks.extend(chunk_document(doc_id, paragraphs, tokenizer))
            
    if not all_chunks:
        print("No chunks generated.")
        return

    print(f"Creating DataFrame for {len(all_chunks)} chunks...")
    df = pd.DataFrame(all_chunks)
    
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'chunks.parquet')
    
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
