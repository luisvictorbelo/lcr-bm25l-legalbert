import subprocess
import json
import os
import sys

def main():
    methods = ["KLI", "KeyBERT", "PLM", "TF-IDF", "Proposition", "MarkedParagraph"]
    alpha = 0.3
    
    results_summary = {}

    for method in methods:
        print(f"\n========================================================")
        print(f"RUNNING METHOD: {method} with alpha={alpha}")
        print(f"========================================================")
        
        # 1. Run pipeline
        pipeline_cmd = [
            sys.executable, "src/pipeline_test.py",
            "--query_method", method,
            "--alpha", str(alpha)
        ]
        print(f"Running pipeline command: {' '.join(pipeline_cmd)}")
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        try:
            # We run it and let the output stream to the console
            subprocess.run(pipeline_cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline for {method}: {e}")
            continue

        # 2. Run evaluation
        eval_cmd = [
            sys.executable, "src/evaluate.py",
            "--query_method", method,
            "--alpha", str(alpha)
        ]
        print(f"Running evaluation command: {' '.join(eval_cmd)}")
        try:
            subprocess.run(eval_cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation for {method}: {e}")
            continue

        # 3. Read generated report
        method_name_file = method.lower().replace('-', '_')
        portion_str = f"_p0.5" if method not in ["Proposition", "MarkedParagraph"] else ""
        dynamic_name = f"{method_name_file}{portion_str}_bm25l_k1_3.5_b_1.0_a{alpha}_legal_bert"
        report_path = f"data/evaluation/report_{dynamic_name}.json"
        
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
                results_summary[method] = report_data["results"]
        else:
            print(f"Warning: Expected report file not found at {report_path}")

    # Generate Markdown summary table
    print("\n\n========================================================")
    print("EVALUATION RESULTS COMPARISON (alpha = 0.7)")
    print("========================================================\n")
    
    # We will print tables for different cutoffs k = 5, 10, 20
    cutoffs = ["k_5", "k_10", "k_20"]
    for cutoff in cutoffs:
        print(f"### Performance at {cutoff.replace('_', ' ').upper()}:")
        print(f"| Method | Precision | Recall | F1 | MRR | NDCG | MAP |")
        print(f"|---|---|---|---|---|---|---|")
        for method in methods:
            if method in results_summary and cutoff in results_summary[method]:
                metrics = results_summary[method][cutoff]
                print(f"| {method} | {metrics['coliee_precision']:.4f} | {metrics['coliee_recall']:.4f} | {metrics['coliee_f1']:.4f} | {metrics['mrr']:.4f} | {metrics['ndcg']:.4f} | {metrics['map']:.4f} |")
            else:
                print(f"| {method} | N/A | N/A | N/A | N/A | N/A | N/A |")
        print()

if __name__ == "__main__":
    main()
