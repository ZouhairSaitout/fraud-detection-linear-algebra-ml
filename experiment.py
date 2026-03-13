import scipy.sparse as sparse
from data_simulation import generate_transactions


def build_linear_system(n_features=200, n_transactions=5000, lambda_reg=1e-3):
      
    """
    Construit le système d'équations normales régularisé (X^T X + λI) w = X^T y.

    Args:
        n_features:     dimension de l'espace des features
        n_transactions: nombre de transactions simulées
        lambda_reg:     intensité de la régularisation L2

    Returns:
        A: matrice creuse CSR de dimension (n_features, n_features)
        b: tableau dense de dimension (n_features,)
    """
  
    X, y = generate_transactions(n_transactions=n_transactions,
                                 n_features=n_features)

    XtX = (X.T @ X).tocsc().tocsr()
    A = XtX + lambda_reg * sparse.eye(n_features, format="csr")

    b = X.T.dot(y)

    return A.tocsr(), b
