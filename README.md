# Legal Case Retrieval System

This repository implements a high-performance retrieval and reranking pipeline for legal documents, combining sparse retrieval (BM25) with dense reranking (Legal-BERT).

## Project Structure

```text
.
├── notebooks/                  # Jupyter notebooks for EDA, query extraction & fine-tuning
│   ├── bm25-all-paragraphs-returns-document-id.ipynb
│   ├── create_propositions_with_llm.ipynb
│   ├── extract_marked_paragraphs.ipynb
│   ├── finetuning.ipynb
│   ├── keybert_query.ipynb
│   ├── kli_query.ipynb
│   ├── plm_query.ipynb
│   ├── proposition-query.ipynb
│   ├── pseudo_alignment.ipynb
│   ├── saul_proposition_generator.ipynb
│   └── tf_idf_query.ipynb
├── src/                        # Modular source code
│   ├── chunking/               # Document chunking logic
│   │   └── create_chunks.py
│   ├── debuggers/              # System analysis and sanity check scripts
│   │   ├── case_stats.py
│   │   ├── check-docs-covered.py
│   │   └── keys-are-values.py
│   ├── embeddings/             # Dense embedding generation
│   │   └── generate_embeddings.py
│   ├── indexes/                # Sparse index building scripts
│   │   └── build_full_test_index.py
│   ├── pairs/                  # Training pair creation & pseudo-alignment
│   │   ├── create-pairs.py
│   │   └── pseudo_alignment.py
│   ├── query_extractor/        # Query extraction modules
│   │   ├── keybert_extractor.py
│   │   ├── kli.py
│   │   ├── marked_paragraph_extractor.py
│   │   ├── plm_extractor.py
│   │   ├── proposition_extractor.py
│   │   └── tfidf_extractor.py
│   ├── bm25_retriever.py       # BM25 manager API
│   ├── evaluate.py             # Ranx & COLIEE evaluation script
│   ├── extract_propositions_input.py # Extraction of marked paragraphs
│   ├── generate_plots.py       # Performance visualization and plot generator
│   ├── pipeline_test.py        # End-to-end retrieval & reranking pipeline
│   └── preprocess.py           # Legal document text cleaning & tokenization
└── run_all_methods.py          # Automated benchmark runner across all 6 query methods
```

## Setup

Ensure you have the required dependencies installed:

```bash
pip3 install transformers pandas pyarrow nltk spacy langdetect bm25s ranx scipy tqdm torch keybert matplotlib seaborn --break-system-packages
python3 -m spacy download en_core_web_sm
```

## Pipeline Execution

Follow these steps in order to process documents and evaluate pipeline performance.

### 1. Preprocessing
Clean and tokenize the raw documents into paragraphs.
```bash
python3 src/preprocess.py
```
*Inputs (local): `data/test-files/docs/*.txt`*  
*Outputs (local): `data/test-files/processed/query_paragraphs_all.parquet`, `data/test-files/processed/corpus_paragraphs_all.parquet`*

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
*Outputs (local): `data/indexes/bm25l_k1_3.5_b_1.0/`*

### 4. Generate Dense Embeddings
Generate 768-dim embeddings using the fine-tuned Legal-BERT model.
```bash
# Embed corpus chunks
python3 src/embeddings/generate_embeddings.py \
  --input data/test-files/processed/test_chunks.parquet \
  --output data/test-files/processed/chunk_embeddings.parquet \
  --model models/legalbert-finetuned/legalbert-finetuned

# Embed query chunks
python3 src/embeddings/generate_embeddings.py \
  --input data/test-files/processed/test_query_chunks.parquet \
  --output data/test-files/processed/query_chunk_embeddings.parquet \
  --model models/legalbert-finetuned/legalbert-finetuned
```

### 5. Run Retrieval Pipeline
Execute the full retrieval and reranking pipeline. You can choose between 6 different query extraction methods:
```bash
# Using KLI (Default)
PYTHONPATH=. python3 src/pipeline_test.py --query_method KLI --alpha 0.5 --portion 0.6

# Using KeyBERT
PYTHONPATH=. python3 src/pipeline_test.py --query_method KeyBERT --alpha 0.5 --portion 0.6

# Using PLM
PYTHONPATH=. python3 src/pipeline_test.py --query_method PLM --alpha 0.5 --portion 0.6

# Using TF-IDF
PYTHONPATH=. python3 src/pipeline_test.py --query_method TF-IDF --alpha 0.5 --portion 0.6

# Using Proposition
PYTHONPATH=. python3 src/pipeline_test.py --query_method Proposition --alpha 0.5

# Using MarkedParagraph
PYTHONPATH=. python3 src/pipeline_test.py --query_method MarkedParagraph --alpha 0.5
```
*Options:*
- `--query_method`: `KLI` (statistical), `KeyBERT` (embedding-based), `PLM` (probabilistic), `TF-IDF` (vector space model), `Proposition` (LLM-based propositions), or `MarkedParagraph` (raw marked paragraphs).
- `--portion`: Fraction of top terms to select (default: `0.6`).
- `--alpha`: BM25 weight in hybrid score combination (default: `0.5`).
- `--top_k_bm25`: Number of candidate documents to retrieve via BM25 (default: `100`).
- `--model_name`: Dense model identifier for output metadata (default: `legal_bert`).

### 6. Evaluation
Generate a comprehensive evaluation report with COLIEE metrics (Precision, Recall, F1) and IR metrics (MRR, NDCG, MAP).
```bash
PYTHONPATH=. python3 src/evaluate.py \
  --labels data/labels/task1_test_labels_2025.json \
  --query_method KLI \
  --query_portion 0.6 \
  --bm25_method bm25l \
  --bm25_k1 3.5 \
  --bm25_b 1.0 \
  --alpha 0.5
```
*Outputs (local): `data/evaluation/report_<dynamic_name>.json`*

---

## Automated Multi-Method Benchmark

To run the pipeline and evaluation automatically across all 6 query extraction methods and output a comparison table:

```bash
python3 run_all_methods.py
```

---

## Visualization & Plotting

To generate performance plots (such as F1@5 vs. Alpha weight, Latency vs. F1 trade-offs, and Precision-Recall curves) based on evaluation summaries:

```bash
python3 src/generate_plots.py
```
*Outputs (local): `data/evaluation/plots/*.png`*

---

## Auxiliary Utilities

- **Marked Paragraph Extraction**: `python3 src/extract_propositions_input.py` - Extract marked paragraphs from raw query documents.
- **Training Pair Generation**: `python3 src/pairs/create-pairs.py` - Create ground truth query-positive document pairs for fine-tuning.
- **Chunk Alignment**: `python3 src/pairs/pseudo_alignment.py` - Align query chunks with positive document chunks for dense model training.
- **Debugging & Analysis**:
  - `python3 src/debuggers/case_stats.py`: Statistics on candidate cases per query.
  - `python3 src/debuggers/check-docs-covered.py`: Verification that all documents are present in processed files.
  - `python3 src/debuggers/keys-are-values.py`: Inspect key-value mappings.

---

## Evaluation Metrics Summary

The evaluation script reports metrics at $k \in \{1, 5, 10, 15, 20, 25\}$.
- **COLIEE Metrics**: Micro-averaged Precision, Recall, and F1.
- **IR Metrics**: MRR, NDCG, and MAP calculated via `ranx`.
