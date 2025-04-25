
# TODO
* Automatiser les simulations multiples pour une boucle complète d'hystérésis
* BONUS : Système de plusieurs modèles d'Ising interagissant ensemble


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
pip install -r requirements.txt
```

Si vous êtes sur VScode, il faut ensuite cliquer manuellement sur "Python: Select Interpreter".
