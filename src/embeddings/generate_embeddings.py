import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import numpy as np
import argparse

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the token embeddings, ignoring padding tokens.
    """
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks.")
    parser.add_argument("--input", default="data/test-files/processed/test_chunks.parquet", help="Input chunks parquet file.")
    parser.add_argument("--output", default="data/test-files/processed/chunk_embeddings.parquet", help="Output embeddings parquet file.")
    parser.add_argument("--model", default="models/legalbert-finetuned/legalbert-finetuned", help="Path to the fine-tuned model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = args.model
    chunks_path = args.input
    output_path = args.output

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please ensure you've downloaded it from Colab.")
        return

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Loading chunks from {chunks_path}...")
    df_chunks = pd.read_parquet(chunks_path)
    print(f"Found {len(df_chunks)} chunks.")

    texts = df_chunks['text'].tolist()
    chunk_ids = df_chunks['chunk_id'].tolist()
    case_ids = df_chunks['case_id'].tolist()

    all_embeddings = []
    all_token_counts = []

    batch_size = 32
    print("Generating embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            model_output = model(**encoded_input)
            
            # Mean Pooling
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = embeddings.cpu().numpy()
            
            # Token counts (excluding padding)
            token_counts = encoded_input['attention_mask'].sum(dim=1).cpu().numpy()
            
            for emb, count in zip(embeddings, token_counts):
                all_embeddings.append(emb.tolist())
                all_token_counts.append(int(count))

    print("Creating output DataFrame...")
    df_output = pd.DataFrame({
        'chunk_id': chunk_ids,
        'case_id': case_ids,
        'embedding': all_embeddings,
        'token_count': all_token_counts
    })

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to {output_path}...")
    df_output.to_parquet(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
