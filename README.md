# Legal Case Retrieval System

This repository implements a high-performance retrieval and reranking pipeline for legal documents, combining sparse retrieval (BM25) with dense reranking (Legal-BERT).

## Project Structure

```text
.
├── data/
│   ├── labels/                 # Ground truth labels (Qrels)
│   ├── test-files/
│   │   ├── docs/               # Raw .txt documents
│   │   └── processed/          # Parquet files and evaluation results
│   └── indexes/                # Persisted BM25 indices
├── models/
│   └── legalbert-finetuned/    # Fine-tuned Legal-BERT model
├── notebooks/                  # Original notebooks for KLI, BM25, and Fine-tuning
└── src/                        # Modular source code
    ├── chunking/               # Document chunking logic
    ├── debuggers/              # Statistics and debugging scripts
    ├── embeddings/             # Dense embedding generation
    ├── indexes/                # BM25 index building scripts
    ├── query_extractor/        # Query building methods (KLI, etc.)
    ├── preprocess.py           # Text cleaning and tokenization
    ├── bm25_retriever.py       # BM25 API (build, save, load, retrieve)
    ├── pipeline_test.py        # End-to-end retrieval pipeline
    └── evaluate.py             # Performance measurement (COLIEE & IR metrics)
```

## Setup

Ensure you have the required dependencies installed:

```bash
pip3 install transformers pandas pyarrow nltk spacy langdetect bm25s ranx scipy tqdm torch keybert --break-system-packages
python3 -m spacy download en_core_web_sm
```

## Pipeline Execution

Follow these steps in order to process the test set and evaluate performance.

### 1. Preprocessing
Clean and tokenize the raw documents into paragraphs.
```bash
python3 src/preprocess.py
```
*Inputs: `data/test-files/docs/*.txt`*  
*Outputs: `query_paragraphs_all.parquet`, `corpus_paragraphs_all.parquet`*

### 2. Chunking
Split paragraphs into overlapping chunks to fit Legal-BERT context limits.
```bash
# Chunk the corpus
python3 src/chunking/create_chunks.py \
  --inputs data/test-files/processed/corpus_paragraphs_all.parquet \
  --output data/test-files/processed/test_chunks.parquet

# Chunk the queries
python3 src/chunking/create_chunks.py \
  --inputs data/test-files/processed/query_paragraphs_all.parquet \
  --output data/test-files/processed/test_query_chunks.parquet
```

### 3. Build BM25 Index
Aggregate chunks into document-level tokens and build the BM25 index.
```bash
python3 src/indexes/build_full_test_index.py
```
*Outputs: `data/indexes/bm25l_k1_3.5_b_1.0/`*

### 4. Generate Dense Embeddings
Generate 768-dim embeddings using the fine-tuned Legal-BERT model.
```bash
# Embed corpus chunks
python3 src/embeddings/generate_embeddings.py \
  --input data/test-files/processed/test_chunks.parquet \
  --output data/test-files/processed/chunk_embeddings.parquet

# Embed query chunks
python3 src/embeddings/generate_embeddings.py \
  --input data/test-files/processed/test_query_chunks.parquet \
  --output data/test-files/processed/query_chunk_embeddings.parquet
```

### 5. Run Retrieval Pipeline
Execute the full pipeline. You can choose between different query extraction methods:
```bash
# Using KLI (Default)
PYTHONPATH=. python3 src/pipeline_test.py --query_method KLI

# Using PLM
PYTHONPATH=. python3 src/pipeline_test.py --query_method PLM

# Using TF-IDF
PYTHONPATH=. python3 src/pipeline_test.py --query_method TF-IDF
```
*Options:*
- `--query_method`: `KLI` (statistical), `KeyBERT` (embedding-based), `PLM` (probabilistic), or `TF-IDF` (vector space model).
- `--portion`: Fraction of top terms to select (default: `0.5`).
- `--alpha`: BM25 weight in hybrid score (default: `0.5`).

### 6. Evaluation
Generate a comprehensive evaluation report with COLIEE (Prec, Recall, F1) and IR metrics (MRR, NDCG, MAP).
```bash
python3 src/evaluate.py \
  --ft_epochs 3 \
  --ft_lr 2e-5 \
  --query_method KLI \
  --query_portion 0.5 \
  --bm25_method bm25l \
  --bm25_k1 3.5 \
  --bm25_b 1.0
```
*Outputs: `data/test-files/processed/evaluation_report.json`*

## Evaluation Metrics Summary

The evaluation script reports metrics at $k \in \{1, 5, 10, 15, 20, 25\}$.
- **COLIEE Metrics**: Micro-averaged Precision, Recall, and F1.
- **IR Metrics**: MRR, NDCG, and MAP calculated via `ranx`.

## Debugging Tools

- **Case Stats**: `python3 src/debuggers/case_stats.py` - Get statistics on candidate cases per query.
- **Check Docs**: `python3 src/debuggers/check-docs-covered.py` - Verify all documents are present in processed files.
