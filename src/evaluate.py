import json
import argparse
import os
from datetime import datetime
from ranx import Qrels, Run, evaluate
from typing import List, Dict

def normalize_doc_id(doc_id: str) -> str:
    """Removes .txt extension if present and strips whitespace."""
    doc_id = str(doc_id).strip()
    if doc_id.lower().endswith('.txt'):
        return doc_id[:-4]
    return doc_id

def calculate_coliee_metrics(qrels_dict: Dict[str, List[str]], results_dict: Dict[str, List[Dict]], k: int):
    """
    Calculates Precision, Recall, and F1 based on COLIEE formulas.
    Micro-averaged across all queries.
    """
    total_retrieved = 0
    total_relevant = 0
    total_correct = 0

    for q_id, relevant_docs in qrels_dict.items():
        if q_id not in results_dict:
            total_relevant += len(relevant_docs)
            continue
        
        retrieved_docs = [r['doc_id'] for r in results_dict[q_id][:k]]
        correct = set(retrieved_docs).intersection(set(relevant_docs))
        
        total_retrieved += len(retrieved_docs)
        total_relevant += len(relevant_docs)
        total_correct += len(correct)

    precision = total_correct / total_retrieved if total_retrieved > 0 else 0
    recall = total_correct / total_relevant if total_relevant > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval results and save a full report.")
    parser.add_argument("--labels", default="data/labels/task1_test_labels_2025.json", help="Path to ground truth labels JSON.")
    parser.add_argument("--results", help="Path to pipeline results JSON.")
    parser.add_argument("--output", help="Path to save the evaluation report.")
    
    # Metadata arguments
    parser.add_argument("--ft_epochs", type=int, default=3, help="Fine-tuning epochs.")
    parser.add_argument("--ft_lr", type=float, default=2e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--ft_batch_size", type=int, default=16, help="Fine-tuning batch size.")
    parser.add_argument("--ft_warmup", type=float, default=0.1, help="Fine-tuning warmup ratio.")
    parser.add_argument("--query_method", default="KLI", help="Query extraction method.")
    parser.add_argument("--query_portion", type=float, default=0.5, help="Query extraction portion.")
    parser.add_argument("--bm25_method", default="bm25l", help="BM25 method.")
    parser.add_argument("--bm25_k1", type=float, default=3.5, help="BM25 k1.")
    parser.add_argument("--bm25_b", type=float, default=1.0, help="BM25 b.")
    parser.add_argument("--model_name", default="legal_bert", help="Dense model name.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight used.")
    
    args = parser.parse_args()

    # Dynamic paths if not provided
    if not args.results or not args.output:
        method_name = args.query_method.lower().replace('-', '_')
        portion_str = f"_p{args.query_portion}" if args.query_method not in ["Proposition", "MarkedParagraph"] else ""
        dynamic_name = f"{method_name}{portion_str}_{args.bm25_method}_k1_{args.bm25_k1}_b_{args.bm25_b}_a{args.alpha}_{args.model_name.lower()}"
        
        if not args.results:
            args.results = f'data/evaluation/results_{dynamic_name}.json'
        
        if not args.output:
            args.output = f'data/evaluation/report_{dynamic_name}.json'

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Load Data
    if not os.path.exists(args.results):
        print(f"Error: Results file not found at {args.results}")
        return

    qrels_dict = {normalize_doc_id(q_id): [normalize_doc_id(d) for d in pos_list] for q_id, pos_list in labels_raw.items()}
    results_dict = {normalize_doc_id(q_id): [{"doc_id": normalize_doc_id(r['doc_id']), "score": r['score']} for r in res_list] for q_id, res_list in results_raw.items()}

    # 2. Prepare ranx objects
    ranx_qrels_data = {q_id: {doc_id: 1 for doc_id in docs} for q_id, docs in qrels_dict.items()}
    qrels = Qrels(ranx_qrels_data)
    ranx_run_data = {q_id: {r['doc_id']: r['score'] for r in docs} for q_id, docs in results_dict.items()}
    run = Run(ranx_run_data)

    # 3. Evaluation
    cutoffs = [1, 5, 10, 15, 20, 25]
    metrics = ["mrr", "ndcg", "map"]
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "fine_tuning": {
                "backbone": "nlpaueb/legal-bert-base-uncased",
                "epochs": args.ft_epochs,
                "learning_rate": args.ft_lr,
                "batch_size": args.ft_batch_size,
                "warmup_ratio": args.ft_warmup
            },
            "query_extraction": {
                "method": args.query_method,
                "portion": args.query_portion
            },
            "bm25": {
                "method": args.bm25_method,
                "k1": args.bm25_k1,
                "b": args.bm25_b
            },
            "reranking": {
                "method": "max_chunk_similarity",
                "alpha": args.alpha,
                "model": args.model_name
            }
        },
        "results": {}
    }

    print("\n" + "="*80)
    print(f"{'K':<4} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'MRR':<7} | {'NDCG':<7} | {'MAP':<7}")
    print("-" * 80)

    for k in cutoffs:
        coliee = calculate_coliee_metrics(qrels_dict, results_dict, k)
        ranx_metrics = [f"{m}@{k}" for m in metrics]
        ranx_scores = evaluate(qrels, run, ranx_metrics, make_comparable=True)
        
        row_results = {
            "coliee_precision": coliee['precision'],
            "coliee_recall": coliee['recall'],
            "coliee_f1": coliee['f1'],
            "mrr": ranx_scores[f'mrr@{k}'],
            "ndcg": ranx_scores[f'ndcg@{k}'],
            "map": ranx_scores[f'map@{k}']
        }
        report_data["results"][f"k_{k}"] = row_results

        print(f"{k:<4} | {coliee['precision']:.4f} | {coliee['recall']:.4f} | {coliee['f1']:.4f} | "
              f"{ranx_scores[f'mrr@{k}']:.4f} | {ranx_scores[f'ndcg@{k}']:.4f} | {ranx_scores[f'map@{k}']:.4f}")

    print("="*80)
    
    # Save Report
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)
    print(f"Full evaluation report saved to: {args.output}\n")

if __name__ == "__main__":
    main()
