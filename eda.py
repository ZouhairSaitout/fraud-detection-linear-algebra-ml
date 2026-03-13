"""
eda.py : Analyse exploratoire des données pour le jeu de transactions frauduleuses.

Produit une figure avec quatre panneaux :
  1. Répartition des classes (fraude vs. légitime)
  2. Distribution des normes L2 des transactions
  3. Distribution du nombre de non-zéros par transaction (sparsité des lignes)
  4. Distribution du nombre de non-zéros par feature (sparsité des colonnes)

Exécution autonome : python eda.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_simulation import generate_transactions


def compute_eda_stats(X, y):
    """Retourne un dictionnaire de statistiques pour affichage et visualisation."""
    n_transactions, n_features = X.shape
    nnz = X.nnz
    density = nnz / (n_transactions * n_features)

    n_fraud = int(y.sum())
    n_legit = n_transactions - n_fraud
    fraud_rate = n_fraud / n_transactions

    row_nnz = np.diff(X.indptr)          # non-zéros par transaction
    col_nnz = np.diff(X.tocsc().indptr)  # non-zéros par feature

    row_norms = np.array(X.power(2).sum(axis=1)).ravel() ** 0.5  # norme L2 par transaction

    return {
        "n_transactions": n_transactions,
        "n_features": n_features,
        "nnz": nnz,
        "density": density,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_rate": fraud_rate,
        "row_nnz": row_nnz,
        "col_nnz": col_nnz,
        "row_norms": row_norms,
    }


def print_report(stats):
    w = 52
    print("=" * w)
    print("  RAPPORT DU JEU DE DONNÉES")
    print("=" * w)
    print(f"  Transactions : {stats['n_transactions']:>10,}")
    print(f"  Features     : {stats['n_features']:>10,}")
    print(f"  Non-zéros    : {stats['nnz']:>10,}")
    print(f"  Densité matrice      : {stats['density']:.4%}")
    print("-" * w)
    print(f"  Transactions légitimes : {stats['n_legit']:>8,}  ({1-stats['fraud_rate']:.1%})")
    print(f"  Transactions frauduleuses : {stats['n_fraud']:>8,}  ({stats['fraud_rate']:.1%})")
    print("-" * w)
    row = stats["row_nnz"]
    print(f"  Non-zéros / transaction : moyenne={row.mean():.1f}  "
          f"écart-type={row.std():.1f}  max={row.max()}")
    col = stats["col_nnz"]
    print(f"  Non-zéros / feature     : moyenne={col.mean():.1f}  "
          f"écart-type={col.std():.1f}  max={col.max()}")
    print("=" * w)


def plot_eda(stats, save_path="eda.png"):
    BLUE   = "#3a7bd5"
    ORANGE = "#e07b3a"
    GREEN  = "#3aab5e"
    GRAY   = "#888888"

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Analyse exploratoire — Jeu de transactions frauduleuses",
                 fontsize=13, fontweight="normal", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # 1-Répartition des classes 
  
    ax1 = fig.add_subplot(gs[0, 0])
    labels  = ["Légitime", "Fraude"]
    counts  = [stats["n_legit"], stats["n_fraud"]]
    colors  = [BLUE, ORANGE]
    bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor="none")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + stats["n_transactions"] * 0.01,
                 f"{count:,}\n({count/stats['n_transactions']:.1%})",
                 ha="center", va="bottom", fontsize=10)
    ax1.set_title("Répartition des classes", fontsize=11)
    ax1.set_ylabel("Nombre de transactions")
    ax1.set_ylim(0, max(counts) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    # 2-Distribution des normes L2 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(stats["row_norms"], bins=50, color=BLUE, edgecolor="none", alpha=0.85)
    ax2.axvline(stats["row_norms"].mean(), color=ORANGE, linewidth=1.5,
                linestyle="--", label=f"moyenne = {stats['row_norms'].mean():.2f}")
    ax2.set_title("Norme L2 des features par transaction", fontsize=11)
    ax2.set_xlabel("‖x‖₂")
    ax2.set_ylabel("Nombre de transactions")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    # 3-Non-zéros par transaction
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(stats["row_nnz"], bins=40, color=GREEN, edgecolor="none", alpha=0.85)
    ax3.axvline(stats["row_nnz"].mean(), color=ORANGE, linewidth=1.5,
                linestyle="--", label=f"moyenne = {stats['row_nnz'].mean():.1f}")
    ax3.set_title("Non-zéros par transaction (sparsité des lignes)", fontsize=11)
    ax3.set_xlabel("Nombre de features non-nuls")
    ax3.set_ylabel("Nombre de transactions")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.spines[["top", "right"]].set_visible(False)

    # 4-Non-zéros par feature
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(stats["col_nnz"], bins=40, color=GRAY, edgecolor="none", alpha=0.75)
    ax4.axvline(stats["col_nnz"].mean(), color=ORANGE, linewidth=1.5,
                linestyle="--", label=f"moyenne = {stats['col_nnz'].mean():.1f}")
    ax4.set_title("Non-zéros par feature (sparsité des colonnes)", fontsize=11)
    ax4.set_xlabel("Nombre de transactions utilisant cette feature")
    ax4.set_ylabel("Nombre de features")
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.spines[["top", "right"]].set_visible(False)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure EDA sauvegardée dans {save_path}")
    return fig


if __name__ == "__main__":
    print("Génération des transactions …")
    X, y = generate_transactions()

    stats = compute_eda_stats(X, y)
    print_report(stats)
    plot_eda(stats)
