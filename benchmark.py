"""
Évalue les trois solveurs itératifs sur un même système linéaire
et affiche le nombre d'itérations, le temps d'exécution et le résidu pour chacun.
"""

import time
import numpy as np
from experiment import build_linear_system
from solvers import jacobi_sparse, gauss_seidel, sor


def residual(A, x, b):
    """Résidu relatif  ‖Ax - b‖ / ‖b‖  (plus petit = meilleur)."""
    r = np.linalg.norm(A.dot(x) - b)
    return r / (np.linalg.norm(b) + 1e-15)


A, b = build_linear_system()

# Chaque lambda reçoit la matrice creuse A et le vecteur b, puis gère
# les composantes CSR elle-même.L'appel reste simple.
solveurs = {
    "Jacobi":       lambda A, b: jacobi_sparse(A.data, A.indices, A.indptr, b),
    "Gauss-Seidel": lambda A, b: gauss_seidel(A.data, A.indices, A.indptr, b),
    "SOR (w=1.2)":  lambda A, b: sor(A.data, A.indices, A.indptr, b, w=1.2),
}

# En-tête du tableau de résultats
print(f"{'Solveur':<18} {'Itérations':>12} {'Temps (s)':>10} {'Résidu relatif':>15}")
print("-" * 58)

for nom, solveur in solveurs.items():
    debut = time.perf_counter()
    x, iters = solveur(A, b)

    temps = time.perf_counter() - debut
    rel_res = residual(A, x, b)

    print(f"{nom:<18} {iters:>12} {temps:>10.3f} {rel_res:>15.2e}")
