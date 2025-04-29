# Structure du projet
1) `Metropolis.py` consiste en le coeur du simulateur d'Ising en 2D avec l'algorithme Metropolis-Hastings. Correspond à une classe.
2) `functions.py` Recèle plein de fonctions utiles au développement du code. La version rapide de l'algorithme, accélérée par Numba, s'y trouve également.
3) `analyse_v2.ipynb` est le notebook servant à lancer des simulations ainsi qu'à analyser les résultats. Les figures du rapport ont aussi été produites grâce à ce notebook.
4) Le dossier `Vieux code` contient des traces de code développé au courant du projet, mais qui a été supplanté par de nouvelles versions.

# Exploration numérique du modèle d'Ising

Ce projet implémente une simulation du modèle d'Ising en 2 dimensions. Le modèle d'Ising est un modèle utilisé en physique statistique pour décrire les changements de phases et certaines propriétés magnétiques, telles que le ferromagnétisme, le paramagnétisme et l'hystérisis.

## Aperçu

Le modèle d'Ising consiste en un treillis de noeuds dotés de spin soit « up » (+1), soit « down » (-1). Les interactions entre noeuds adjacents ainsi que la température du système influencent toutes deux le comportement global. Ce projet vise à simuler des changements de phase et à trouver les seuils critiques afférents de manière numérique, afin de les comparer aux prédictions analytiques provenant de la littérature scientifique.


## Installation

Pour installer ce projet, il faut cloner le repos, créer un environnement virtuel puis insaller les librairies nécessairement au bon fonctionnement.

```bash
git clone <repository-url>
cd "dossier global du projet"
python -m venv venv
source ../venv/bin/activate
cd physnum25
```

Si vous êtes sur VScode, il faut ensuite cliquer manuellement sur "Python: Select Interpreter".
