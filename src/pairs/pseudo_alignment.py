import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import numpy as np

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the token embeddings, ignoring padding tokens.
    """
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_pairs_path = 'data/processed/train_pairs.json'
    chunks_path = 'data/processed/chunks.parquet'
    output_path = 'data/processed/chunk_pairs.parquet'

    print("Loading training pairs...")
    with open(train_pairs_path, 'r', encoding='utf-8') as f:
        train_pairs = json.load(f)

    print("Loading chunks...")
    df_chunks = pd.read_parquet(chunks_path)

    # 2. Extract unique case_ids involved in training pairs
    query_ids = {p['query_id'] for p in train_pairs}
    positive_ids = {p['positive_id'] for p in train_pairs}
    involved_ids = query_ids.union(positive_ids)

    print(f"Number of unique documents involved: {len(involved_ids)}")

    # Filter chunks to only those involved
    df_filtered = df_chunks[df_chunks['case_id'].isin(involved_ids)].copy()
    print(f"Number of chunks to embed: {len(df_filtered)}")

    # 3. Model Initialization
    model_name = 'nlpaueb/legal-bert-base-uncased'
    print(f"Initializing {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 4. Pre-compute Chunk Embeddings
    chunk_embeddings = {}
    
    texts = df_filtered['text'].tolist()
    chunk_ids = df_filtered['chunk_id'].tolist()
    
    batch_size = 32
    print("Computing embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_ids = chunk_ids[i:i+batch_size]
            
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            model_output = model(**encoded_input)
            
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = embeddings.cpu().numpy()
            
            for chunk_id, emb in zip(batch_ids, embeddings):
                chunk_embeddings[chunk_id] = emb

    # 5. Compute Pseudo-Alignments
    print("Aligning query chunks with positive chunks...")
    # Group chunks by case_id for faster retrieval
    case_to_chunks = df_filtered.groupby('case_id')['chunk_id'].apply(list).to_dict()

    pseudo_pairs = []

    for pair in tqdm(train_pairs):
        q_id = pair['query_id']
        p_id = pair['positive_id']
        
        if q_id not in case_to_chunks or p_id not in case_to_chunks:
            continue
            
        q_chunk_ids = case_to_chunks[q_id]
        p_chunk_ids = case_to_chunks[p_id]
        
        # Get embeddings for all query chunks and positive chunks
        q_embs = np.stack([chunk_embeddings[cid] for cid in q_chunk_ids])
        p_embs = np.stack([chunk_embeddings[cid] for cid in p_chunk_ids])
        
        # Normalize for cosine similarity
        q_embs_norm = q_embs / np.linalg.norm(q_embs, axis=1, keepdims=True)
        p_embs_norm = p_embs / np.linalg.norm(p_embs, axis=1, keepdims=True)
        
        # Compute similarity matrix (nq x np)
        sim_matrix = np.matmul(q_embs_norm, p_embs_norm.T)
        
        # For each query chunk, find the best positive chunk
        best_p_indices = np.argmax(sim_matrix, axis=1)
        
        for q_idx, p_idx in enumerate(best_p_indices):
            pseudo_pairs.append({
                'query_chunk_id': q_chunk_ids[q_idx],
                'positive_chunk_id': p_chunk_ids[p_idx]
            })

    # 6. Save Output
    print(f"Generated {len(pseudo_pairs)} pseudo-pairs.")
    df_output = pd.DataFrame(pseudo_pairs)
    print(f"Saving to {output_path}...")
    df_output.to_parquet(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
