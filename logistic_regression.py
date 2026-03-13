"""
logistic_regression.py : Classifieur de fraude entraîné par descente de gradient.

Ce module relie les trois thèmes du projet :
  • Apprentissage automatique — régression logistique comme classifieur de fraude
  • Optimisation             — descente de gradient par mini-lots minimisant la log-vraisemblance
  • Algèbre linéaire         — le gradient est  Xᵀ(ŷ - y) / n,  un produit matrice–vecteur creux

------------
train(X_train, y_train, ...)  →  poids, biais, historique des pertes
evaluate(X, y, poids, biais)  →  dictionnaire de métriques
plot_training(loss_history, roc_data, save_path)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from data_simulation import generate_transactions


# Fonctions mathématiques de base

def _sigmoid(z):
    """Sigmoïde numériquement stable."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _log_loss(y, y_hat):
    """Entropie croisée binaire, écrêtée pour la stabilité numérique."""
    y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def _predict_proba(X, w, b):
    """Retourne P(fraude=1) pour chaque ligne de X."""
    return _sigmoid(X.dot(w) + b)



# Entraînement


def train(X_train, y_train,
          lr=0.1,
          lambda_reg=1e-3,
          max_iter=300,
          batch_size=256,
          tol=1e-5,
          class_weight="balanced",
          random_state=42):
    """
    Entraîne un classifieur par régression logistique avec descente de gradient par mini-lots.

    Le gradient de la log-perte régularisée et pondérée par classe par rapport à w est :
        ∇_w L = Xᵀ(poids_échantillons ⊙ (ŷ - y)) / n  +  λ w
    — un produit matrice–vecteur creux, gardant la mémoire proportionnelle au nombre de valeurs non nulles.

    La pondération par classe amplifie la contribution au gradient de la classe minoritaire
    (fraude), empêchant l'optimiseur de converger vers "toujours prédire légitime".

    Args:
        X_train      : matrice CSR creuse, forme (n, p)
        y_train      : vecteur d'étiquettes binaires, forme (n,)
        lr           : taux d'apprentissage
        lambda_reg   : intensité de la régularisation L2
        max_iter     : nombre maximum d'époques de descente de gradient
        batch_size   : taille des mini-lots
        tol          : arrêt si le changement de perte < tol
        class_weight : "balanced" (automatique) ou dict {0: w0, 1: w1}
        random_state : graine du générateur aléatoire

    Returns:
        w            : vecteur de poids appris, forme (p,)
        b            : biais scalaire appris
        loss_history : liste des valeurs de perte d'entraînement par époque
    """
    rng = np.random.default_rng(random_state)
    n, p = X_train.shape

    # calcul des poids par échantillon pour corriger le déséquilibre des classes
    n_pos = y_train.sum()
    n_neg = n - n_pos
    if class_weight == "balanced":
        w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
    elif isinstance(class_weight, dict):
        w_neg, w_pos = class_weight[0], class_weight[1]
    else:
        w_pos = w_neg = 1.0

    sample_weights = np.where(y_train == 1, w_pos, w_neg)  # forme (n,)

    w = np.zeros(p)
    b = 0.0
    loss_history = []

    for epoch in range(max_iter):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            X_b = X_train[batch]
            y_b = y_train[batch]
            sw  = sample_weights[batch]
            n_b = len(batch)

            y_hat = _predict_proba(X_b, w, b)
            # pondération de l'erreur par le poids de classe avant le calcul du gradient
            weighted_err = sw * (y_hat - y_b)

            grad_w = X_b.T.dot(weighted_err) / n_b + lambda_reg * w
            grad_b = weighted_err.mean()

            w = w - lr * grad_w
            b = b - lr * grad_b

            epoch_loss += _log_loss(y_b, y_hat)
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if epoch > 0 and abs(loss_history[-2] - loss_history[-1]) < tol:
            print(f"  Convergence à l'époque {epoch + 1}  (perte={avg_loss:.5f})")
            break

    return w, b, loss_history


# Évaluation

def evaluate(X, y, w, b, threshold=0.5):
    """
    Calcule les métriques de classification sur un jeu de données.

    Retourne un dictionnaire avec : accuracy, precision, recall, f1, roc_auc,
    confusion_matrix, fpr, tpr (pour le tracé de la courbe ROC).
    """
    proba = _predict_proba(X, w, b)
    preds = (proba >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y, proba)

    return {
        "accuracy":         (preds == y).mean(),
        "precision":        precision_score(y, preds, zero_division=0),
        "recall":           recall_score(y, preds, zero_division=0),
        "f1":               f1_score(y, preds, zero_division=0),
        "roc_auc":          roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, preds),
        "fpr":              fpr,
        "tpr":              tpr,
        "proba":            proba,
    }


def print_metrics(metrics, split_name="Test"):
    w = 44
    print("=" * w)
    print(f"  ÉVALUATION — {split_name}")
    print("=" * w)
    print(f"  Exactitude  : {metrics['accuracy']:.4f}")
    print(f"  Précision   : {metrics['precision']:.4f}")
    print(f"  Rappel      : {metrics['recall']:.4f}")
    print(f"  Score F1    : {metrics['f1']:.4f}")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print("-" * w)
    cm = metrics["confusion_matrix"]
    print(f"  Matrice de confusion :")
    print(f"    VN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  VP={cm[1,1]:>6}")
    print("=" * w)


# Graphiques

def plot_training(loss_history, train_metrics, test_metrics, save_path="entrainement.png"):
    BLEU   = "#3a7bd5"
    ORANGE = "#e07b3a"
    VERT   = "#3aab5e"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Régression Logistique — Entraînement & Évaluation", fontsize=13)

    # 1-Convergence de la perte 
    ax = axes[0]
    ax.plot(loss_history, color=BLEU, linewidth=2)
    ax.set_title("Convergence de la descente de gradient", fontsize=11)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Log-perte (ensemble d'entraînement)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # 2-Courbe ROC
    ax = axes[1]
    ax.plot(test_metrics["fpr"], test_metrics["tpr"],
            color=BLEU, linewidth=2,
            label=f"Test  AUC={test_metrics['roc_auc']:.3f}")
    ax.plot(train_metrics["fpr"], train_metrics["tpr"],
            color=VERT, linewidth=1.5, linestyle="--",
            label=f"Entraînement AUC={train_metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#cccccc", linewidth=1, linestyle=":")
    ax.set_title("Courbe ROC", fontsize=11)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # 3-Diagramme en barres des métriques
    ax = axes[2]
    metric_names = ["Exactitude", "Précision", "Rappel", "F1", "ROC-AUC"]
    train_vals = [train_metrics[k] for k in
                  ["accuracy", "precision", "recall", "f1", "roc_auc"]]
    test_vals  = [test_metrics[k]  for k in
                  ["accuracy", "precision", "recall", "f1", "roc_auc"]]

    x = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x - width / 2, train_vals, width, label="Entraînement", color=VERT, alpha=0.85)
    ax.bar(x + width / 2, test_vals,  width, label="Test",         color=BLEU, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Métriques de classification", fontsize=11)
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Graphique d'entraînement sauvegardé dans {save_path}")
    return fig


# Point d'entrée

if __name__ == "__main__":
    print("Génération des données …")
    X, y = generate_transactions(n_transactions=5000, n_features=200)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Entraînement : {X_train.shape[0]} échantillons  |  Test : {X_test.shape[0]} échantillons")
    print(f"Taux de fraude — entraînement : {y_train.mean():.2%}  |  test : {y_test.mean():.2%}\n")

    print("Entraînement de la régression logistique par descente de gradient …")
    w, b, loss_history = train(X_train, y_train, lr=0.1, max_iter=300)

    # Recherche du seuil qui maximise le F1 sur l'ensemble d'entraînement.
    # Cela évite les deux extrêmes : 0.5 (manque toute la fraude à 5% de prévalence)
    # et le taux de fraude brut (signale tout).
    train_proba = _predict_proba(X_train, w, b)
    from sklearn.metrics import f1_score as _f1
    thresholds  = np.linspace(0.01, 0.5, 200)
    f1_scores   = [_f1(y_train, (train_proba >= t).astype(int), zero_division=0)
                   for t in thresholds]
    threshold   = float(thresholds[np.argmax(f1_scores)])
    print(f"\nSeuil optimal (F1 max sur l'entraînement) = {threshold:.3f}\n")

    train_metrics = evaluate(X_train, y_train, w, b, threshold=threshold)
    test_metrics  = evaluate(X_test,  y_test,  w, b, threshold=threshold)

    print_metrics(train_metrics, "Entraînement")
    print_metrics(test_metrics,  "Test")

    plot_training(loss_history, train_metrics, test_metrics)
