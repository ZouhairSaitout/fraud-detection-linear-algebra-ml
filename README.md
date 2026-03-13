# Détection de fraude - Algèbre linéaire & ML

Détection de transactions bancaires frauduleuses par systèmes linéaires creux, solveurs itératifs et régression logistique entraînée from scratch.

---

## C'est quoi ce projet

Les données transactionnelles sont naturellement creuses : chaque transaction n'active qu'une fraction des features disponibles (montant, pays, heure, type de marchand...). Ce projet exploite explicitement cette structure plutôt que de la déléguer à une librairie.

Concrètement, on formule d'abord le problème comme un système linéaire creux `(XᵀX + λI)w = Xᵀy`, on compare des solveurs itératifs dessus, puis on construit une régression logistique from scratch par-dessus. Le taux de fraude est fixé à 5% dans les données simulées, volontairement élevé pour faciliter les expériences. Les vrais datasets sont souvent bien en dessous.

Trois questions ont guidé le projet :
- Est-ce que des solveurs classiques (Jacobi, Gauss-Seidel, SOR) tiennent la route sur les équations normales avec des données creuses ?
- Que se passe-t-il quand Jacobi diverge et pourquoi exactement ?
- Comment construire une régression logistique from scratch qui fonctionne réellement sur des données déséquilibrées ?

---

## Structure

```
├── data_simulation.py     # matrice de transactions synthétiques (format CSR, densité 1%)
├── experiment.py          # construction du système linéaire (XᵀX + λI)w = Xᵀy
├── solvers.py             # Jacobi, Gauss-Seidel, SOR opèrent directement sur les tableaux CSR
├── benchmark.py           # comparaison des solveurs côte à côte
├── eda.py                 # analyse de sparsité, déséquilibre des classes, distribution des normes L2
├── logistic_regression.py # entraînement from scratch avec class weighting + sélection de seuil
├── predict.py             # classe FraudDetector (score, predict, save/load)
└── plots.py               # courbes de convergence et de scalabilité
```

---

## La partie algèbre linéaire

L'idée centrale : plutôt que de passer directement à un optimiseur, on écrit le problème de régression sous forme matricielle :

```
(XᵀX + λI) w = Xᵀy
```

C'est la régression ridge en forme matricielle. La matrice A = XᵀX + λI est symétrique définie positive et creuse ce qui rend les solveurs itératifs naturellement adaptés.

Les trois solveurs de `solvers.py` opèrent tous directement sur les tableaux CSR (`data`, `indices`, `indptr`) sans jamais passer en dense. Les résultats du benchmark sont honnêtes :

```
Solveur          Itérations   Temps (s)   Résidu
-------------------------------------------------
Jacobi                  500      0.533    4.16e+88   ← diverge
Gauss-Seidel             12      0.019    6.16e-08
SOR (ω=1.2)              18      0.021    7.04e-08
```

La divergence de Jacobi n'est pas un bug, c'est le résultat attendu. La convergence exige une dominance diagonale stricte, que cette matrice n'a pas. Gauss-Seidel et SOR convergent rapidement parce qu'ils utilisent les valeurs mises à jour en place.

---

## La partie ML

Une fois le système linéaire résolu, on passe à la régression logistique comme classifieur. Le gradient s'écrit :

```
∇L = Xᵀ(ŷ - y) / n + λw
```

Ce qui est encore une fois un produit matrice-vecteur creux, le lien direct entre les deux moitiés du projet.

Deux choses qui changent vraiment le résultat sur des données déséquilibrées :

1. **Class weighting** : les erreurs sur les fraudes sont pondérées ~10× dans le gradient. Sans ça, le modèle apprend à prédire "légitime" pour tout et obtient 95 % d'accuracy en ne faisant rien.
2. **Sélection du seuil** : le seuil par défaut à 0.5 est inutile ici. On balaye les seuils possibles et on choisit celui qui maximise le F1 sur le train.

Résultats sur le jeu de test :

| Métrique | Score |
|----------|-------|
| Accuracy | 0.954 |
| Précision | 0.526 |
| Rappel | 0.800 |
| F1 | 0.635 |
| ROC-AUC | 0.968 |

Le rappel compte plus que la précision en détection de fraude.Rater une fraude est pire qu'une fausse alarme. Le ROC-AUC de 0.968 signifie que le modèle classe bien les transactions frauduleuses au-dessus des légitimes en termes de probabilité, même si le F1 brut reste modéré.

---

## Limites

Les données sont synthétiques. La sparsité, le taux de fraude et la structure des features sont tous contrôlés. Ça rend les expériences reproductibles et les maths propres, mais les chiffres ne se transfèrent pas directement à de vraies données transactionnelles. Le dataset Kaggle sur la fraude par carte de crédit serait l'étape suivante évidente.

Les solveurs itératifs (Jacobi/GS/SOR) sont ici principalement pédagogiques. En pratique, on utiliserait le gradient conjugué (`scipy.sparse.linalg.cg`) pour ce type de système. L'intérêt est de comprendre pourquoi certains solveurs convergent et d'autres non, pas de les défendre pour un usage en production.

---

## Installation

```bash
git clone https://github.com/ZouhairSaitout/fraud-detection-linear-algebra-ml.git
cd fraud-detection-linear-algebra-ml
pip install numpy scipy matplotlib scikit-learn
```

```bash
python eda.py                  # analyse exploratoire
python benchmark.py            # comparaison des solveurs
python logistic_regression.py  # entraînement et évaluation
python predict.py              # pipeline d'inférence complet
python plots.py                # courbes de convergence et de scalabilité
```
