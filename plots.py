"""
Deux figures :

1. Scalabilité   — temps d'exécution en fonction de la dimension du problème pour les trois solveurs
2. Convergence   — norme du résidu par itération (montre la rapidité de convergence de chaque solveur),
                   plus la courbe de log-perte de la descente de gradient du modèle de régression
                   logistique pour comparaison.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from experiment import build_linear_system
from solvers import jacobi_sparse, gauss_seidel, sor
from logistic_regression import train
from data_simulation import generate_transactions
from sklearn.model_selection import train_test_split

COLORS = {
    "Jacobi":       "#3a7bd5",
    "Gauss-Seidel": "#e07b3a",
    "SOR (w=1.2)":  "#3aab5e",
    "DG log-perte": "#9b59b6",
}

# Figure 1: Scalabilité

def plot_scalability(save_path="scalabilite.png"):
    sizes = [50, 100, 200, 500]
    results = {name: [] for name in COLORS if name != "DG log-perte"}

    solver_fns = {
        "Jacobi":       lambda A, b: jacobi_sparse(A.data, A.indices, A.indptr, b),
        "Gauss-Seidel": lambda A, b: gauss_seidel(A.data, A.indices, A.indptr, b),
        "SOR (w=1.2)":  lambda A, b: sor(A.data, A.indices, A.indptr, b, w=1.2),
    }

    for s in sizes:
        print(f"  scalabilité — n_features={s} …")
        A, b = build_linear_system(n_features=s)
        for name, fn in solver_fns.items():
            t0 = time.perf_counter()
            fn(A, b)
            results[name].append(time.perf_counter() - t0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, times in results.items():
        ax.plot(sizes, times, marker="o", label=name,
                color=COLORS[name], linewidth=2)

    ax.set_xlabel("Dimension du problème (n_features)", fontsize=12)
    ax.set_ylabel("Temps d'exécution (s)", fontsize=12)
    ax.set_title("Scalabilité des solveurs itératifs", fontsize=13)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    return fig


# Figure 2: Courbes de convergence

def _solver_residuals(solver_fn, A, b, max_iter=80):
    """Exécute un solveur itération par itération en enregistrant ‖Ax-b‖ à chaque étape."""
    residuals = []
    # démarrage à chaud avec 1 itération à la fois en remplaçant max_iter à chaque appel
    x = np.zeros(len(b))
    for it in range(1, max_iter + 1):
        x, _ = solver_fn(A.data, A.indices, A.indptr, b, max_iter=it, tol=0.0)
        res = np.linalg.norm(A.dot(x) - b)
        residuals.append(res)
        if res < 1e-6:
            break
    return residuals


def plot_convergence(save_path="convergence.png"):
    print("  construction du système pour le graphique de convergence …")
    A, b = build_linear_system(n_features=200)

    solver_fns = {
        "Jacobi":       jacobi_sparse,
        "Gauss-Seidel": gauss_seidel,
        "SOR (w=1.2)":  lambda data, idx, ptr, b, **kw: sor(
                            data, idx, ptr, b, w=1.2, **kw),
    }

    print("  calcul des résidus des solveurs …")
    solver_residuals = {}
    for name, fn in solver_fns.items():
        solver_residuals[name] = _solver_residuals(fn, A, b)

    print("  entraînement de la régression logistique pour la courbe de perte …")
    X, y = generate_transactions(n_transactions=5000, n_features=200)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    _, _, loss_history = train(X_tr, y_tr, lr=0.1, max_iter=150)

    # tracé
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analyse de convergence", fontsize=13)

    # Gauche : résidus des solveurs
    for name, res in solver_residuals.items():
        if max(res) > 0:
            ax1.semilogy(range(1, len(res) + 1), res,
                         label=name, color=COLORS[name], linewidth=2)
    ax1.set_title("Solveur linéaire — résidu ‖Ax−b‖ par itération", fontsize=11)
    ax1.set_xlabel("Itération")
    ax1.set_ylabel("Norme du résidu (échelle log)")
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    # Droite : log-perte de la descente de gradient
    ax2.plot(loss_history, color=COLORS["DG log-perte"], linewidth=2)
    ax2.set_title("Descente de gradient — log-perte par époque", fontsize=11)
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Perte d'entropie croisée binaire")
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    return fig


# Point d'entrée

if __name__ == "__main__":
    plot_scalability()
    plot_convergence()
    try:
        plt.show()
    except Exception:
        pass
