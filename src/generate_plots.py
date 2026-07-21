import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

def main():
    # 1. Load data
    summary_path = 'data/evaluation/summary_report.csv'
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df = pd.read_csv(summary_path)
    os.makedirs('data/evaluation/plots', exist_ok=True)
    
    # 2. Plot 1: F1@5 vs. Alpha (Line plot)
    plt.figure(figsize=(9, 5.5))
    sns.lineplot(
        data=df, 
        x="Alpha", 
        y="F1@5", 
        hue="Method", 
        style="Method", 
        markers=True, 
        dashes=False, 
        linewidth=2, 
        markersize=8
    )
    plt.title("Impacto do Peso da Busca Híbrida (Alfa) na Pontuação F1@5")
    plt.xlabel("Alfa (Peso Léxico)")
    plt.ylabel("F1 Score em K=5")
    plt.xticks([0.3, 0.5, 0.7])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Método Query")
    plt.tight_layout()
    plt.savefig('data/evaluation/plots/f1_vs_alpha.png', dpi=300)
    plt.close()
    print("Saved f1_vs_alpha.png")
    
    # 3. Plot 2: F1@5 vs. Avg Query Time (Scatter plot)
    # We select alpha=0.3 since it is the best performing alpha for almost all methods
    df_best = df[df["Alpha"] == 0.3]
    
    plt.figure(figsize=(8.5, 5.5))
    scatter = sns.scatterplot(
        data=df_best,
        x="Avg Query Time (ms)",
        y="F1@5",
        hue="Method",
        style="Method",
        s=120,
        legend=False
    )
    
    # Label each point
    for i in range(df_best.shape[0]):
        row = df_best.iloc[i]
        plt.text(
            row["Avg Query Time (ms)"] + 3, 
            row["F1@5"] - 0.0005, 
            f"{row['Method']}\n({row['Avg Query Time (ms)']:.1f}ms, F1:{row['F1@5']:.4f})",
            fontsize=9.5,
            verticalalignment='center'
        )
        
    plt.title("Trade-off: F1@5 Score vs. Average Query Latency (Alpha = 0.3)")
    plt.xlabel("Average Query Processing Time (ms)")
    plt.ylabel("F1 Score at K=5")
    plt.xlim(100, 480)
    plt.ylim(0.27, 0.32)
    plt.tight_layout()
    plt.savefig('data/evaluation/plots/f1_vs_latency.png', dpi=300)
    plt.close()
    print("Saved f1_vs_latency.png")
    
    # 4. Plot 3: Precision & Recall vs. K for the best configuration
    # Load all raw reports to find the one for MarkedParagraph with alpha=0.3
    best_report_path = 'data/evaluation/report_markedparagraph_bm25l_k1_3.5_b_1.0_a0.3_legal_bert.json'
    if os.path.exists(best_report_path):
        with open(best_report_path, 'r') as f:
            best_report = json.load(f)
            
        k_values = []
        precision_vals = []
        recall_vals = []
        
        for k_str, metrics in best_report["results"].items():
            k = int(k_str.split('_')[1])
            k_values.append(k)
            precision_vals.append(metrics["coliee_precision"])
            recall_vals.append(metrics["coliee_recall"])
            
        plt.figure(figsize=(8.5, 5))
        plt.plot(k_values, precision_vals, marker='o', color='#d62728', label='Precisão@K', linewidth=2)
        plt.plot(k_values, recall_vals, marker='s', color='#1f77b4', label='Revocação@K', linewidth=2)
        
        plt.title("Curva de Precisão vs. Revocação para diferentes valores de K\n(ParágrafosMarcados, Alfa = 0.3)")
        plt.xlabel("K (Número de Documentos Recuperados)")
        plt.ylabel("Pontuação da Métrica")
        plt.xticks(k_values)
        plt.ylim(0.0, 0.7)
        plt.legend(loc='center right')
        plt.tight_layout()
        plt.savefig('data/evaluation/plots/precision_recall_vs_k.png', dpi=300)
        plt.close()
        print("Saved precision_recall_vs_k.png")
    else:
        print(f"Warning: Best report not found at {best_report_path}, skipping plot 3.")

if __name__ == "__main__":
    main()
