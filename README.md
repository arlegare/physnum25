# Exploration numérique du modèle d'Ising

Ce projet implémente une simulation du modèle d'Ising en 3 dimensions. Le modèle d'Ising est un modèle utilisé en physique statistique pour décrire les changements de phases et les propriétés magnétiques.

## Aperçu

Le modèle d'Ising consiste en un treillis de noeuds dotés de spin soit « up » (+1), soit « down » (-1). Les interactions entre noeuds adjacents ainsi que la température du système influencent toutes deux le comportement global. Ce projet vise à simuler des changements de phase et à trouver les seuils critiques afférents de manière numérique, afin de les comparer aux prédictions analytiques provenant de la littérature scientifique.


## Installation

Pour installer ce projet, il faut cloner le repos et insaller les dépendances.

```bash
git clone <repository-url>
cd ising-spin-glass
pip install -r requirements.txt
```

## Utilisation

Pour lancer les simulations, il faut utiliser la classe `IsingModel` définie `src/ising_model.py`. En voici un exemple :

```python
from src.ising_model import IsingModel

model = IsingModel(size=10, temperature=1.0)
model.initialize_lattice()
model.run_simulation(steps=1000)
```

## Lancement de simulations

Pour s'assurer que l'implémentation soit bien effectuée, on doit lancer les unités de test qui se trouvent dans le dossier `tests`. La commande suivante s'avère utile :

```bash
pytest tests/test_ising_model.py
```

## Contribution

N'oubliez pas de push/pull et de commenter vos contributions!

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
