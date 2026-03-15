import numpy as np
import scipy.sparse as sparse


def generate_transactions(n_transactions=5000, n_features=200, density=0.01):
    
    """
    Simule une matrice creuse de caractéristiques de transactions
    et des labels binaires de fraude.

    Args:
        n_transactions: nombre de transactions (lignes)
        n_features:     nombre de caractéristiques par transaction (colonnes)
        density:        fraction des entrées non nulles dans X

    Returns:
        X: matrice creuse CSR de dimension (n_transactions, n_features)
        y: tableau de labels binaires de fraude de dimension (n_transactions,)
    """
  
    X = sparse.random(n_transactions, n_features, density=density, format="csr",
                      random_state=42)

    rng = np.random.default_rng(42)
    true_weights = rng.standard_normal(n_features)

    scores = X.dot(true_weights)

    threshold = np.percentile(scores, 95)
    y = (scores > threshold).astype(float)   # float pour l'algèbre linéaire utilisée ensuite

    return X, y
