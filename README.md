# Détection de Fraude Bancaire par Algèbre Linéaire et Machine Learning

> Simulation, analyse et classification de transactions bancaires à grande dimension,
> en exploitant les systèmes linéaires creux, les solveurs itératifs et la régression logistique
> entraînée par descente de gradient.

---

## Objectifs du projet

La fraude bancaire est un problème de **classification binaire à fort déséquilibre** : dans un
flux de transactions, la grande majorité sont légitimes, et une minorité (ici ~5 %) sont
frauduleuses. Détecter cette minorité sans déclencher trop de fausses alarmes est le défi central.

Ce projet l'aborde sous trois angles complémentaires :

1. **Algèbre linéaire & matrices creuses** — les données transactionnelles sont naturellement
   creuses (chaque transaction n'active qu'une fraction des 200 features disponibles). On
   exploite cette structure pour résoudre efficacement le système linéaire des équations normales
   `(XᵀX + λI)w = Xᵀy` sans jamais convertir la matrice en format dense.

2. **Solveurs itératifs & optimisation** — trois algorithmes classiques (Jacobi, Gauss-Seidel,
   SOR) sont comparés sur leur vitesse de convergence et leur coût calculatoire. La descente de
   gradient mini-batch est introduite comme pont vers le Machine Learning.

3. **Machine Learning** — une régression logistique est entraînée from scratch, avec gestion du
   déséquilibre de classes, sélection du seuil de décision optimal, et évaluation complète
   (précision, rappel, F1, ROC-AUC).

---

## Architecture du projet

```
fraud-detection-linear-systems/
│
├── data_simulation.py       # Génération des données transactionnelles simulées
├── experiment.py            # Construction du système linéaire (équations normales)
├── solvers.py               # Solveurs itératifs : Jacobi, Gauss-Seidel, SOR
├── benchmark.py             # Comparaison des solveurs (temps, itérations, résidu)
├── eda.py                   # Analyse exploratoire des données (EDA)
├── logistic_regression.py   # Modèle ML : entraînement, évaluation, visualisation
├── predict.py               # Pipeline d'inférence : scoring de nouvelles transactions
└── plots.py                 # Courbes de scalabilité et de convergence
```

---

## Description détaillée de chaque module

### `data_simulation.py` — Génération des données

Ce module crée un jeu de données synthétique qui imite la structure réelle de données
transactionnelles bancaires.

**Ce qu'il fait :**
- Génère une matrice `X` de forme `(n_transactions, n_features)` au format **CSR** (Compressed
  Sparse Row). Avec une densité de 1 %, la grande majorité des entrées valent zéro — exactement
  comme en pratique, où une transaction n'active qu'un sous-ensemble restreint de features
  (montant, pays, heure, type de marchand, etc.).
- Calcule un score par transaction via `X · w_vrai`, où `w_vrai` est un vecteur de poids aléatoires.
- Classe les 5 % de transactions avec le score le plus élevé comme frauduleuses (`y = 1`), les
  autres comme légitimes (`y = 0`).

**Pourquoi des données simulées ?** Cela permet de contrôler exactement la structure du problème
(dimensionnalité, sparsité, taux de fraude) et de reproduire les expériences de manière
déterministe via une graine aléatoire fixe.

---

### `experiment.py` — Construction du système linéaire

Ce module traduit le problème de régression en un **système linéaire** de la forme `Aw = b`.

**La formule centrale :** `(XᵀX + λI) w = Xᵀy`

C'est l'équation normale de la régression linéaire régularisée (régularisation L2, aussi appelée
Ridge). Voici ce que représente chaque terme :

| Terme | Dimension | Signification |
|-------|-----------|---------------|
| `X` | `(n, p)` | Matrice des features transactionnelles |
| `Xᵀ X` | `(p, p)` | Corrélations entre features |
| `λI` | `(p, p)` | Régularisation Ridge — évite le surapprentissage |
| `A = XᵀX + λI` | `(p, p)` | Matrice du système à résoudre |
| `b = Xᵀy` | `(p,)` | Projection des labels sur les features |
| `w` | `(p,)` | Vecteur de poids solution |

La matrice `A` est **symétrique définie positive** et **creuse** — propriété exploitée par les
solveurs itératifs du module suivant.

---

### `solvers.py` — Solveurs itératifs

Ce module implémente trois solveurs classiques pour les systèmes linéaires creux `Aw = b`. Tous
opèrent directement sur les tableaux CSR (`data`, `indices`, `indptr`) sans jamais convertir `A`
en matrice dense — ce qui maintient la consommation mémoire proportionnelle au nombre
d'éléments non nuls.

**Jacobi**

À chaque itération, chaque composante `w[i]` est mise à jour en isolant le terme diagonal :

```
w_new[i] = (b[i] - Σ_{j≠i} A[i,j] · w[j]) / A[i,i]
```

Toutes les mises à jour utilisent les valeurs de l'itération précédente. C'est simple et
parallélisable, mais la convergence est lente — voire inexistante sur ce problème (le rayon
spectral de la matrice d'itération dépasse 1).

**Gauss-Seidel**

Identique à Jacobi, mais les mises à jour sont appliquées **en place** : dès que `w[i]` est
recalculé, sa nouvelle valeur est immédiatement utilisée pour les composantes suivantes. Cela
accélère significativement la convergence (12 itérations contre 500+ pour Jacobi sur ce
problème).

**SOR (Successive Over-Relaxation)**

Extension de Gauss-Seidel avec un facteur de relaxation `ω` :

```
w[i] ← ω · w_GS[i] + (1 - ω) · w_old[i]
```

Avec `ω = 1` : identique à Gauss-Seidel. Avec `1 < ω < 2` : sur-relaxation qui peut accélérer
la convergence. Ici `ω = 1.2` est utilisé par défaut.

---

### `benchmark.py` — Comparaison des solveurs

Lance les trois solveurs sur le même système linéaire et affiche un tableau comparatif :

```
Solver               Iterations   Time (s)   Rel. residual
----------------------------------------------------------
Jacobi                      500      0.533        4.16e+88   ← diverge
Gauss-Seidel                 12      0.019        6.16e-08   ← converge
SOR (w=1.2)                  18      0.021        7.04e-08   ← converge
```

La divergence de Jacobi n'est pas un bug : elle illustre un résultat théorique important —
la convergence de Jacobi n'est garantie que si la matrice est à diagonale strictement dominante,
ce qui n'est pas le cas ici.

---

### `eda.py` — Analyse exploratoire des données

Avant tout modélisation, ce module analyse la structure brute du dataset et produit un rapport
en quatre panneaux :

1. **Équilibre des classes** — visualise le ratio 95 %/5 % légitimes/frauduleux. Ce fort
   déséquilibre est un paramètre clé : un modèle naïf qui prédit "toujours légitime" obtient
   95 % d'accuracy sans rien détecter.

2. **Distribution des normes L2 des transactions** — mesure l'amplitude de chaque vecteur de
   features `‖x‖₂`. Une distribution homogène indique que les transactions sont comparables en
   magnitude.

3. **Sparsité par ligne (transactions)** — nombre de features non nulles par transaction. Avec
   une densité de 1 % et 200 features, chaque transaction active en moyenne 2 features.

4. **Sparsité par colonne (features)** — nombre de transactions qui activent chaque feature.
   Une distribution uniforme signifie que les features sont équitablement utilisées.

---

### `logistic_regression.py` — Modèle de classification

C'est le cœur du projet Machine Learning. Ce module implémente une **régression logistique**
entraînée from scratch par **descente de gradient mini-batch**.

**Pourquoi la régression logistique ?**
Elle est directement connectée à l'algèbre linéaire : le score de fraude est `σ(Xw + b)` où
`σ` est la fonction sigmoïde, et le gradient est `Xᵀ(ŷ - y) / n` — exactement un produit
matrice-vecteur creux.

**La fonction de perte** est l'entropie croisée binaire (log-loss) :

```
L(w) = -1/n · Σ [ y · log(ŷ) + (1-y) · log(1-ŷ) ] + (λ/2) · ‖w‖²
```

**La mise à jour des poids** à chaque mini-batch :

```
w ← w - α · [Xᵀ(sw ⊙ (ŷ - y)) / n + λw]
b ← b - α · mean(sw ⊙ (ŷ - y))
```

où `sw` sont les poids par échantillon qui corrigent le déséquilibre de classes.

**Gestion du déséquilibre de classes** — avec 5 % de fraudes, un seuil de décision à 0.5 ne
déclencherait jamais d'alerte. Deux mécanismes sont mis en place :
- Les erreurs sur les transactions frauduleuses sont multipliées par `n/(2·n_fraude) ≈ 10`
  dans le gradient (class weighting).
- Le seuil de décision optimal est sélectionné en maximisant le F1-score sur le jeu
  d'entraînement.

**Résultats obtenus :**

| Métrique | Train | Test |
|----------|-------|------|
| Accuracy | 0.960 | 0.954 |
| Précision | 0.558 | 0.526 |
| Rappel | 0.935 | 0.800 |
| F1 | 0.699 | 0.635 |
| ROC-AUC | 0.989 | 0.968 |

Le ROC-AUC de 0.968 signifie que le modèle distingue très bien les fraudes des transactions
légitimes dans l'espace des probabilités, même si le F1 est modéré — ce qui est attendu avec
un tel déséquilibre de classes.

---

### `predict.py` — Pipeline d'inférence

Ce module expose la classe `FraudDetector` : une interface propre qui encapsule les poids
entraînés et permet de scorer de nouvelles transactions en production.

**Fonctionnalités :**
- `score(X)` — retourne la probabilité de fraude `P(fraude=1)` pour chaque transaction.
- `predict(X)` — retourne le label binaire selon le seuil de décision.
- `score_report(X)` — retourne un rapport structuré par transaction avec identifiant, probabilité,
  label, et niveau de risque (MINIMAL / LOW / MEDIUM / HIGH).
- `summary(X, y_true)` — affiche un tableau de bord complet : nombre de fraudes détectées, TP,
  FP, FN, TN, précision, rappel.
- `save_model` / `load_model` — persistance des poids au format `.npz` compressé.

Exemple de rapport de scoring :

```
ID             P(fraud)    Label Risk
--------------------------------------------
TXN-0000         0.4257    legit LOW
TXN-0004         0.5681    FRAUD MEDIUM
TXN-0009         0.4496    legit LOW
```

---

### `plots.py` — Visualisations

Produit deux figures :

**`scalability.png`** — temps de calcul des trois solveurs en fonction de la dimension `p`
(nombre de features). Montre comment le coût croît avec la taille du problème.

**`convergence.png`** — deux sous-graphes côte à côte :
- Résidu `‖Aw - b‖` par itération (échelle logarithmique) pour les solveurs itératifs.
  Illustre la divergence de Jacobi et la convergence rapide de Gauss-Seidel et SOR.
- Log-loss par époque pour la descente de gradient. Montre la décroissance régulière de
  l'erreur d'entraînement jusqu'à convergence.

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/fraud-detection-linear-systems.git
cd fraud-detection-linear-systems

# Installer les dépendances
pip install numpy scipy matplotlib scikit-learn
```

---

## Utilisation

Chaque module peut être exécuté indépendamment :

```bash
# 1. Analyse exploratoire des données
python eda.py

# 2. Comparer les solveurs itératifs
python benchmark.py

# 3. Entraîner et évaluer le modèle ML
python logistic_regression.py

# 4. Pipeline complet d'inférence (entraînement → sauvegarde → scoring)
python predict.py

# 5. Générer les courbes de scalabilité et de convergence
python plots.py
```

---

## Concepts mathématiques clés

| Concept | Rôle dans le projet |
|---------|---------------------|
| Matrice CSR | Stockage efficace des données transactionnelles creuses |
| Équations normales `(XᵀX + λI)w = Xᵀy` | Formulation algébrique du problème de régression |
| Rayon spectral | Détermine si un solveur itératif converge ou diverge |
| Sigmoïde `σ(z) = 1/(1+e⁻ᶻ)` | Transforme un score réel en probabilité de fraude |
| Gradient `Xᵀ(ŷ-y)/n` | Produit creux matrice-vecteur — cœur de la descente de gradient |
| ROC-AUC | Mesure la qualité de discrimination du classifieur, indépendante du seuil |
| Class weighting | Corrige le déséquilibre 95/5 dans la fonction de perte |

---

## Dépendances

| Bibliothèque | Usage |
|-------------|-------|
| `numpy` | Algèbre linéaire dense, vecteurs, calculs numériques |
| `scipy.sparse` | Matrices creuses au format CSR |
| `scikit-learn` | Découpage train/test, métriques (F1, ROC-AUC, matrice de confusion) |
| `matplotlib` | Visualisations (EDA, convergence, ROC, scalabilité) |
