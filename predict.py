"""
predict.py : Pipeline d'inférence pour le classifieur de fraude entraîné.

Fournit :
  • Classe FraudDetector    : encapsule les poids entraînés, évalue de nouvelles transactions
  • save_model / load_model : sauvegarde / charge les poids dans un fichier .npz
  • demo()                  : exemple bout en bout : entraînement → sauvegarde → rechargement → évaluation

"""

import numpy as np
import scipy.sparse as sparse
from data_simulation import generate_transactions
from logistic_regression import train, evaluate, print_metrics
from sklearn.model_selection import train_test_split


# Persistance du modèle


def save_model(w, b, path="fraud_model.npz"):
    """Sauvegarde le vecteur de poids et le biais dans une archive NumPy compressée."""
    np.savez_compressed(path, w=w, b=np.array([b]))
    print(f"Modèle sauvegardé dans {path}")


def load_model(path="fraud_model.npz"):
    """Charge le vecteur de poids et le biais depuis une archive .npz."""
    data = np.load(path)
    return data["w"], float(data["b"][0])


# Classe du détecteur


class FraudDetector:
    """
    Enveloppe légère autour des poids de la régression logistique entraînée.

    Utilisation:
    
    detector = FraudDetector(w, b, threshold=0.5)
    scores   = detector.score(X_new)        # probabilité de fraude par transaction
    flags    = detector.predict(X_new)      # binaire : 1 = fraude, 0 = légitime
    report   = detector.score_report(X_new) # dictionnaire structuré par transaction
    """

    def __init__(self, w, b, threshold=0.5):
        self.w = w
        self.b = b
        self.threshold = threshold

    @classmethod
    def from_file(cls, path="fraud_model.npz", threshold=0.5):
        w, b = load_model(path)
        return cls(w, b, threshold)

    def _sigmoid(self, z):
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def score(self, X):
        """Retourne la probabilité de fraude pour chaque transaction. Forme : (n,)"""
        return self._sigmoid(X.dot(self.w) + self.b)

    def predict(self, X):
        """Retourne l'indicateur binaire de fraude pour chaque transaction. Forme : (n,)"""
        return (self.score(X) >= self.threshold).astype(int)

    def score_report(self, X, transaction_ids=None):
        """
        Retourne une liste de dictionnaires, un par transaction, contenant :
          id, fraud_probability, predicted_label, risk_level
        """
        probs = self.score(X)
        preds = (probs >= self.threshold).astype(int)
        n = X.shape[0]
        ids = transaction_ids if transaction_ids is not None else list(range(n))

        def niveau_risque(p):
            if p >= 0.8:  return "ÉLEVÉ"
            if p >= 0.5:  return "MOYEN"
            if p >= 0.2:  return "FAIBLE"
            return "MINIMAL"

        return [
            {
                "id":                  ids[i],
                "fraud_probability":   round(float(probs[i]), 4),
                "predicted_label":     int(preds[i]),
                "risk_level":          niveau_risque(probs[i]),
            }
            for i in range(n)
        ]

    def summary(self, X, y_true=None):
        """Affiche un résumé concis de l'évaluation pour un lot de transactions."""
        probs  = self.score(X)
        preds  = (probs >= self.threshold).astype(int)
        n      = X.shape[0]
        n_flag = preds.sum()

        print("=" * 46)
        print("  DÉTECTION DE FRAUDE — RÉSUMÉ D'ÉVALUATION")
        print("=" * 46)
        print(f"  Transactions évaluées     : {n:>8,}")
        print(f"  Signalées comme fraude    : {n_flag:>8,}  ({n_flag/n:.2%})")
        print(f"  Seuil de décision         : {self.threshold:.2f}")
        print("-" * 46)
        print(f"  Probabilité de fraude moy. : {probs.mean():.4f}")
        print(f"  Probabilité de fraude max. : {probs.max():.4f}")
        print(f"  Probabilité de fraude min. : {probs.min():.4f}")

        if y_true is not None:
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            tn = int(((preds == 0) & (y_true == 0)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            print("-" * 46)
            print(f"  Vrais positifs  (fraudes détectées)  : {tp:>6,}")
            print(f"  Faux positifs   (fausses alarmes)    : {fp:>6,}")
            print(f"  Faux négatifs   (fraudes manquées)   : {fn:>6,}")
            print(f"  Vrais négatifs  (légitimes, validés) : {tn:>6,}")
            print(f"  Précision : {precision:.4f}   Rappel : {recall:.4f}")

        print("=" * 46)


# Démonstration


def demo():
    print("Étape 1 : génération et découpage des données")
    X, y = generate_transactions(n_transactions=5000, n_features=200)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Étape 2 : entraînement")
    w, b, _ = train(X_train, y_train, lr=0.1, max_iter=300)

    print("\n Étape 3 : sauvegarde et rechargement du modèle")
    save_model(w, b)
    detector = FraudDetector.from_file()

    print("\n Étape 4 : évaluation de l'ensemble de test")
    detector.summary(X_test, y_true=y_test)

    print("\n Étape 5 : rapport sur un échantillon de transactions")
    sample = X_test[:10]
    sample_ids = [f"TXN-{i:04d}" for i in range(10)]
    report = detector.score_report(sample, transaction_ids=sample_ids)

    print(f"\n{'ID':<12} {'P(fraude)':>10} {'Étiquette':>10} {'Risque':<10}")
    print("-" * 46)
    for row in report:
        label_str = "FRAUDE" if row["predicted_label"] else "légitime"
        print(f"{row['id']:<12} {row['fraud_probability']:>10.4f} "
              f"{label_str:>10} {row['risk_level']:<10}")

    print("\nÉtape 6 : évaluation de nouvelles transactions inédites")
    X_new, _ = generate_transactions(n_transactions=500, n_features=200)
    detector.summary(X_new)


if __name__ == "__main__":
    demo()
