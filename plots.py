"""
Figures produites :
  1. scalabilite.png           - temps d'exécution vs dimension pour les trois solveurs
  2. convergence.png           - résidus solveurs (log) + log-perte descente de gradient
  3. structure_creuse.png      - visualisation de la structure CSR de la matrice X
  4. convergence_solveurs.png  - résidu ‖Aw−b‖ par itération (Gauss-Seidel vs SOR)
  5. sensibilite_omega.png     - nombre d'itérations SOR en fonction de ω
  6. courbe_roc.png            - courbe ROC train/test standalone
  7. courbe_perte.png          - log-perte par époque standalone
  8. matrice_confusion.png     - matrice de confusion sur le jeu de test
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score  

from experiment import build_linear_system
from solvers import jacobi_sparse, gauss_seidel, sor
from logistic_regression import train, _predict_proba, evaluate 
from data_simulation import generate_transactions


# Palette commune
BLEU    = "#3a7bd5"
ORANGE  = "#e07b3a"
VERT    = "#3aab5e"
VIOLET  = "#9b59b6"
GRIS    = "#888888"

COULEURS = {
    "Jacobi":       BLEU,
    "Gauss-Seidel": ORANGE,
    "SOR (w=1.2)":  VERT,
    "DG log-perte": VIOLET,
}


# Utilitaires internes

def residus_par_iteration(solver_fn, A, b, max_iter=80): 
    """
    Exécute un solveur itération par itération et enregistre
    le résidu ‖Ax−b‖ à chaque étape.
    """
    residus = []
    for it in range(1, max_iter + 1):
        x, _ = solver_fn(A.data, A.indices, A.indptr, b, max_iter=it, tol=0.0)
        r = np.linalg.norm(A.dot(x) - b)
        residus.append(r)
        if r < 1e-6:
            break
    return residus


def entrainer_modele(): 
    """
    Entraîne la régression logistique et retourne tout ce dont
    les fonctions de visualisation ont besoin.
    """
    X, y = generate_transactions(n_transactions=5000, n_features=200)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    w, b, historique_perte = train(X_tr, y_tr, lr=0.1, max_iter=300)

    # Seuil optimal par maximisation du F1 sur l'entraînement
    proba_tr = _predict_proba(X_tr, w, b)
    seuils   = np.linspace(0.01, 0.5, 200)
    f1s      = [f1_score(y_tr, (proba_tr >= s).astype(int), zero_division=0)
                for s in seuils]
    seuil    = float(seuils[np.argmax(f1s)])

    metriques_tr = evaluate(X_tr, y_tr, w, b, threshold=seuil)
    metriques_te = evaluate(X_te, y_te, w, b, threshold=seuil)

    return dict(
        w=w, b=b, seuil=seuil,
        historique_perte=historique_perte,
        metriques_tr=metriques_tr,
        metriques_te=metriques_te,
        y_tr=y_tr, y_te=y_te,
    )

# Figure 1: Scalabilité

def plot_scalabilite(save_path="scalabilite.png"):
    tailles   = [50, 100, 200, 500]                         
    resultats = {nom: [] for nom in COULEURS if nom != "DG log-perte"}

    solveurs = {
        "Jacobi":       lambda A, b: jacobi_sparse(A.data, A.indices, A.indptr, b),
        "Gauss-Seidel": lambda A, b: gauss_seidel(A.data, A.indices, A.indptr, b),
        "SOR (w=1.2)":  lambda A, b: sor(A.data, A.indices, A.indptr, b, w=1.2),
    }

    for s in tailles:
        print(f"  scalabilité  - n_features={s} …")
        A, b = build_linear_system(n_features=s)
        for nom, fn in solveurs.items():                    
            t0 = time.perf_counter()
            fn(A, b)
            resultats[nom].append(time.perf_counter() - t0) 

    fig, ax = plt.subplots(figsize=(8, 5))
    for nom, temps in resultats.items():                     
        ax.plot(tailles, temps, marker="o", label=nom,      
                color=COULEURS[nom], linewidth=2)

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

def plot_convergence(save_path="convergence.png"):
    print("  construction du système pour le graphique de convergence …")
    A, b = build_linear_system(n_features=200)

    solveurs = {                                             
        "Jacobi":       jacobi_sparse,
        "Gauss-Seidel": gauss_seidel,
        "SOR (w=1.2)":  lambda data, idx, ptr, b, **kw: sor(
                            data, idx, ptr, b, w=1.2, **kw),
    }

    print("  calcul des résidus des solveurs …")
    residus_solveurs = {nom: residus_par_iteration(fn, A, b)  
                        for nom, fn in solveurs.items()}        

    print("  entraînement de la régression logistique pour la courbe de perte …")
    X, y = generate_transactions(n_transactions=5000, n_features=200)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    _, _, loss_history = train(X_tr, y_tr, lr=0.1, max_iter=150)

    # tracé
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analyse de convergence", fontsize=13)

    # Gauche : résidus des solveurs
    for nom, res in residus_solveurs.items():              
        if max(res) > 0:
            ax1.semilogy(range(1, len(res) + 1), res,
                         label=nom, color=COULEURS[nom], linewidth=2)
    ax1.set_title("Solveur linéaire  - résidu ‖Ax−b‖ par itération", fontsize=11)
    ax1.set_xlabel("Itération")
    ax1.set_ylabel("Norme du résidu (échelle log)")
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    # Droite : log-perte de la descente de gradient
    ax2.plot(loss_history, color=COULEURS["DG log-perte"], linewidth=2)
    ax2.set_title("Descente de gradient  - log-perte par époque", fontsize=11)
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Perte d'entropie croisée binaire")
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    return fig


# Figure 3: Structure creuse de la matrice X

def plot_structure_creuse(save_path="structure_creuse.png"):
    """
    Visualise la position des valeurs non nulles dans un sous-ensemble de X
    (spy plot). Illustre concrètement ce que signifie une matrice creuse
    avec une densité de 1%.
    """
    print("  génération de la structure creuse …")
    X, _ = generate_transactions(n_transactions=80, n_features=60)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spy(X, markersize=3, color=BLEU, alpha=0.8)
    ax.set_xlabel("Features (colonnes)", fontsize=11)
    ax.set_ylabel("Transactions (lignes)", fontsize=11)
    ax.set_title("Structure creuse de la matrice X (80×60, densité=1%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    plt.close(fig)


# Figure 4: Convergence détaillée - Gauss-Seidel vs SOR

def plot_convergence_solveurs(save_path="convergence_solveurs.png"):
    """
    Résidu ‖Aw−b‖ itération par itération pour Gauss-Seidel et SOR uniquement
    (Jacobi exclu car il diverge et écrase l'échelle).
    Permet de comparer finement la vitesse de décroissance des deux solveurs convergents.
    """
    print("  calcul des résidus Gauss-Seidel / SOR …")
    A, b = build_linear_system(n_features=200)

    solveurs = {
        "Gauss-Seidel": gauss_seidel,
        "SOR (w=1.2)":  lambda data, idx, ptr, b, **kw: sor(
                            data, idx, ptr, b, w=1.2, **kw),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for nom, fn in solveurs.items():
        residus = residus_par_iteration(fn, A, b)          
        ax.semilogy(range(1, len(residus) + 1), residus,
                    marker="o", markersize=4,
                    color=COULEURS[nom], linewidth=2, label=nom)

    ax.set_xlabel("Itération", fontsize=12)
    ax.set_ylabel("Résidu ‖Aw − b‖ (échelle log)", fontsize=12)
    ax.set_title("Convergence des solveurs itératifs", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    plt.close(fig)



# Figure 5: Sensibilité de SOR au paramètre ω

def plot_sensibilite_omega(save_path="sensibilite_omega.png"):
    """
    Nombre d'itérations nécessaires à SOR en fonction de ω ∈ (0, 2).
    Justifie le choix ω=1.2 et illustre le théorème de convergence SOR.
    La ligne verticale ω=1 correspond à Gauss-Seidel.
    """
    print("  calcul de la sensibilité à omega …")
    A, b = build_linear_system(n_features=200)

    omegas      = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9]
    iterations  = []
    for w_val in omegas:
        _, it = sor(A.data, A.indices, A.indptr, b, w=w_val, max_iter=500)
        iterations.append(it)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(omegas, iterations, marker="o", color=BLEU, linewidth=2)
    ax.axvline(1.0, color=ORANGE, linestyle="--", alpha=0.7,
               label="Gauss-Seidel (ω=1)")
    ax.axvline(1.2, color=VERT, linestyle="--", alpha=0.7,
               label="ω=1.2 (valeur choisie)")
    ax.set_xlabel("Facteur de relaxation ω", fontsize=12)
    ax.set_ylabel("Nombre d'itérations", fontsize=12)
    ax.set_title("Sensibilité de SOR au paramètre ω", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    plt.close(fig)


# Figure 6: Courbe ROC standalone

def plot_courbe_roc(modele, save_path="courbe_roc.png"):
    """
    Courbe ROC (Receiver Operating Characteristic) pour les jeux d'entraînement
    et de test. L'aire sous la courbe (AUC) mesure la capacité de discrimination
    du modèle indépendamment du seuil de décision.
    """
    metriques_tr = modele["metriques_tr"]
    metriques_te = modele["metriques_te"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(metriques_te["fpr"], metriques_te["tpr"],
            color=BLEU, linewidth=2,
            label=f"Test  AUC={metriques_te['roc_auc']:.3f}")
    ax.plot(metriques_tr["fpr"], metriques_tr["tpr"],
            color=VERT, linewidth=1.5, linestyle="--",
            label=f"Entraînement AUC={metriques_tr['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#cccccc", linewidth=1, linestyle=":")
    ax.fill_between(metriques_te["fpr"], metriques_te["tpr"],
                    alpha=0.08, color=BLEU)
    ax.set_xlabel("Taux de faux positifs", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs", fontsize=12)
    ax.set_title("Courbe ROC  - Régression Logistique", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    plt.close(fig)


# Figure 7: Courbe de perte standalone

def plot_courbe_perte(modele, save_path="courbe_perte.png"):
    """
    Log-perte (entropie croisée binaire) par époque d'entraînement.
    Montre la convergence de la descente de gradient mini-batch
    et permet de détecter une éventuelle divergence ou sur-apprentissage.
    """
    historique = modele["historique_perte"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(historique, color=VIOLET, linewidth=2)
    ax.fill_between(range(len(historique)), historique,
                    alpha=0.1, color=VIOLET)
    ax.set_xlabel("Époque", fontsize=12)
    ax.set_ylabel("Log-perte (entropie croisée)", fontsize=12)
    ax.set_title("Convergence de la descente de gradient", fontsize=13)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    plt.close(fig)


# Figure 8: Matrice de confusion

def plot_matrice_confusion(modele, save_path="matrice_confusion.png"):
    """
    Matrice de confusion sur le jeu de test.
    Les quatre cellules  VP, FP, FN, VN  permettent d'évaluer
    l'équilibre précision/rappel du classifieur.
    """
    cm = modele["metriques_te"]["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prédit Légitime", "Prédit Fraude"], fontsize=10)
    ax.set_yticklabels(["Réel Légitime", "Réel Fraude"], fontsize=10)

    # Valeurs dans les cellules
    for i in range(2):
        for j in range(2):
            couleur_texte = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color=couleur_texte)

    ax.set_title("Matrice de confusion (jeu de test)", fontsize=12)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → sauvegardé {save_path}")
    plt.close(fig)


# Génèration de toutes les figures

if __name__ == "__main__":

    # Figures indépendantes du modèle ML (solveurs + données)
    print("\n Figures des solveurs et des données")
    plot_scalabilite()
    plot_structure_creuse()
    plot_convergence_solveurs()
    plot_sensibilite_omega()
    plot_convergence() 

    # Figures du modèle ML; un seul entraînement pour toutes
    print("\n Entraînement du modèle (une seule fois)")
    modele = entrainer_modele()                            

    print("\n Figures du modèle ML")
    plot_courbe_roc(modele)
    plot_courbe_perte(modele)
    plot_matrice_confusion(modele)

    print("\nToutes les figures ont été générées.")
    try:
        plt.show()
    except Exception:
        pass
