import numpy as np


# Jacobi

def jacobi_sparse_numba(data, indices, indptr, b, max_iter=500, tol=1e-6):
   
    """
    Solveur Jacobi opérant sur des matrices au format CSR.

    À chaque itération, toutes les composantes sont mises à jour simultanément
    en utilisant les valeurs de l'étape précédente. Aucune modification en place
    n'est effectuée avant que le balayage complet soit terminé.

    Args:
        data, indices, indptr: représentation CSR de la matrice A
        b:                     vecteur côté droit
        max_iter:              nombre maximal d'itérations
        tol:                   seuil de convergence (norme L2 sur la mise à jour)

    Returns:
        x: solution
        k: nombre d'itérations effectuées
    """
  
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            row = slice(indptr[i], indptr[i + 1])
            cols = indices[row]
            vals = data[row]

            diag_mask = (cols == i)
            diag = vals[diag_mask][0]
            off_sum = vals[~diag_mask] @ x[cols[~diag_mask]]

            x_new[i] = (b[i] - off_sum) / diag

        delta = x_new - x
        if np.linalg.norm(delta) < tol:
            return x_new.copy(), k

        x[:] = x_new

    return x, max_iter


# Gauss-Seidel

def gauss_seidel_numba(data, indices, indptr, b, max_iter=500, tol=1e-6):
    
    """
    Solveur Gauss-Seidel opérant sur des matrices au format CSR.

    Les composantes sont mises à jour EN PLACE à chaque balayage, 
    de sorte que les valeurs nouvellement calculées sont utilisées 
    immédiatement pour les lignes suivantes. Cela converge généralement 
    plus rapidement que Jacobi pour le même problème.

    Args:
        data, indices, indptr: représentation CSR de la matrice A
        b:                     vecteur côté droit
        max_iter:              nombre maximal d'itérations
        tol:                   seuil de convergence (norme L2 sur la mise à jour)

    Returns:
        x: solution
        k: nombre d'itérations effectuées
    """
  
    n = len(b)
    x = np.zeros(n)

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            row = slice(indptr[i], indptr[i + 1])
            cols = indices[row]
            vals = data[row]

            diag_mask = (cols == i)
            diag = vals[diag_mask][0]
            off_sum = vals[~diag_mask] @ x[cols[~diag_mask]]

            x[i] = (b[i] - off_sum) / diag

        if np.linalg.norm(x - x_old) < tol:
            return x, k

    return x, max_iter


# Successive Over-Relaxation (SOR)


def sor_numba(data, indices, indptr, b, w=1.2, max_iter=500, tol=1e-6):
   
    """
    Solveur SOR (Successive Over-Relaxation) opérant sur des matrices au format CSR.

    Étend Gauss-Seidel avec un facteur de relaxation w :
        x[i] ← w * x_GS[i] + (1 - w) * x_old[i]

    w = 1.0 → identique à Gauss-Seidel.
    1 < w < 2 → sur-relaxation, réduit souvent le nombre d'itérations.
    0 < w < 1 → sous-relaxation, utile lorsque G-S divergerait.

    Args:
        data, indices, indptr: représentation CSR de la matrice A
        b:                     vecteur côté droit
        w:                     facteur de relaxation (par défaut 1.2)
        max_iter:              nombre maximal d'itérations
        tol:                   seuil de convergence (norme L2 sur la mise à jour)

    Returns:
        x: solution
        k: nombre d'itérations effectuées
    """
  
    n = len(b)
    x = np.zeros(n)

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            row = slice(indptr[i], indptr[i + 1])
            cols = indices[row]
            vals = data[row]

            diag_mask = (cols == i)
            diag = vals[diag_mask][0]
            off_sum = vals[~diag_mask] @ x[cols[~diag_mask]]

            x_gs = (b[i] - off_sum) / diag
            x[i] = w * x_gs + (1 - w) * x_old[i]

        if np.linalg.norm(x - x_old) < tol:
            return x, k

    return x, max_iter
