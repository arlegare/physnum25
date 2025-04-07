import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import scipy.ndimage as sc
import scipy.constants as cte
import time
import os
import sys

@nb.njit(nopython=True)
def microstate_energy(lattice, h, betaJ, size):

    """
    Calcule l'énergie totale d'un micro-état donné (lattice : configuration de spins; h : composante Z du champ magnétique).

    Args:
        lattice (np.ndarray): Configuration de spins.
        h (float): Composante Z du champ magnétique.
        betaJ (float): Constante de couplage.
        size (int): Taille de la grille.

    Returns:
        float: Énergie totale du micro-état.
    """

    energie_totale = 0.0

    for i in range(size):
        for j in range(size):
            spin = lattice[i, j]
            # Conditions périodiques (Pac-Man)
            voisinage = (
                lattice[(i + 1) % size, j] +  # Voisin du bas
                lattice[(i - 1) % size, j] +  # Voisin du haut
                lattice[i, (j + 1) % size] +  # Voisin de droite
                lattice[i, (j - 1) % size]    # Voisin de gauche
            )
            # Contribution de l'énergie du spin
            energie_totale -= betaJ *  spin * voisinage  # Interaction avec les voisins

            # Contribution de l'énergie du champ magnétique
            energie_totale -= h * spin  # Signe (-) car si spin est dans la même direction que le champ, l'énergie est minimisée.

    return energie_totale

@nb.njit(nopython=True)
def convolution_2d_periodic(lattice, kernel):
        """
        Effectue une convolution 2D avec conditions aux limites périodiques.

        Args:
            lattice (np.ndarray): Grille de spins.
            kernel (np.ndarray): Noyau de convolution.

        Returns:
            np.ndarray: Résultat de la convolution.
        """

        N, M = lattice.shape
        kh, kw = kernel.shape
        kh2 = kh // 2
        kw2 = kw // 2

        result = np.zeros_like(lattice)

        for i in range(N):
            for j in range(M):
                acc = 0.0
                for u in range(-kh2, kh2 + 1):
                    for v in range(-kw2, kw2 + 1):
                        ni = (i + u) % N
                        nj = (j + v) % M
                        acc += lattice[ni, nj] * kernel[u + kh2, v + kw2]
                result[i, j] = acc

        return result

@nb.njit(nopython=True)
def interaction_energy(lattice, row, col, h, betaJ, size):
    """
    Calcule l'énergie d'interaction pour un spin donné.

    Args :
        lattice (np.ndarray): Grille de spins.
        row (int): Indice de la ligne du spin.
        col (int): Indice de la colonne du spin.
        h (float): Champ magnétique externe.
        J (float): Constante de couplage.
        size (int): Taille de la grille.

    Renvoie :
        float: Énergie d'interaction pour le spin donné.
    """

    spin = lattice[row, col]
    # Conditions périodiques pour les voisins
    voisinage = (
        lattice[(row + 1) % size, col] +
        lattice[(row - 1) % size, col] +
        lattice[row, (col + 1) % size] +
        lattice[row, (col - 1) % size]
    )
    # Énergie due au champ magnétique et aux interactions avec les voisins
    return -h * spin - betaJ * spin * voisinage

@nb.njit(nopython=True)
def find_equilibrium(lattice, n_iter, betaJ, h, size):
    """
    Trouve l'état d'équilibre en utilisant l'algorithme de Metropolis.

    Args:
        betaJ (float): Valeur de beta * J (normalisée).
        h (float): Champ magnétique externe.

    Returns:
        tuple: Liste des grilles, énergie finale, liste des moyennes des spins, liste des énergies.
    """
    energy = microstate_energy(lattice, h, betaJ, size) # Énergie initiale du système
    spin_mean_list = [np.mean(lattice)]
    energy_list = [energy]

    for _ in range(n_iter):
        row = np.random.randint(0, size)
        col = np.random.randint(0, size)

        #old_spin = lattice[row, col]
        lattice[row, col] *= -1  # Essai de flip

        new_lattice = lattice.copy()
        new_lattice[row][col] *= -1 # Flippage d'un spin au hasard (teneur Monte Carlo du problème...)
        
        # Terme dû au champ + terme de corrélation avec conditions frontières périodiques.
        # On calcule seulement l'énergie du spin concerné puisque les autres ne changent pas.
        E_i = interaction_energy(lattice, row, col, h, betaJ, size) # Avant le flip.
        E_f = interaction_energy(new_lattice, row, col, h, betaJ, size) # Après le flip.

        DeltaE = E_f - E_i  # Variation d'énergie

        # Si l'énergie du nouveau microétat est plus grande, on flip seulement avec la probabilité donnée par l'équation avec l'exponentielle
        if DeltaE > 0 and np.random.random() < np.exp(-betaJ * DeltaE):  
            lattice = new_lattice
            energy += DeltaE
            
        # Si l'énergie est plus petite, on flippe (100% de chance)
        elif DeltaE <= 0:
            lattice = new_lattice  
            energy += DeltaE

        spin_mean_list.append(np.mean(lattice))
        energy_list.append(energy)

    return lattice, spin_mean_list, energy_list

class Metropolis():
    def __init__(self, n_iter, lattice_size, magnetic_field, temperature, J, previous_lattice = None, pourcentage_up=0.80):
        """
        Initialise les paramètres de la simulation de Metropolis.

        Args:
            n_iter (int): Nombre d'itérations pour la simulation.
            lattice_size (int): Taille de la grille de spins.
            magnetic_field (float): Champ magnétique externe.
            temperature (float): Température du système.
            J (float): Constante de couplage (positive pour ferromagnétisme, négative pour antiferromagnétisme).
            previous_lattice (np.ndarray, optional): Grille de spins initiale. Si None, une grille sera générée.
        """
    
        self.n_iter = n_iter  # Nb d'itérations/steps pour la simulation. Par ex. choisir assez long pour atteindre l'équilibre pour une valeur de (T,B).
        self.size = lattice_size  # Dimension de la grille (carrée) de spins. 
        self.B = magnetic_field  # Champ magnétique externe.
        self.T = temperature # Température du système. On peut l'utiliser pour calculer betaJ.
        self.J = J  # Constante de couplage. >0 pour ferromagnétisme et <0 pour antiferromagnétisme.
        self.betaJ = J / (cte.physical_constants["Boltzmann constant"][0] * temperature)
        self.up_perc = pourcentage_up  # Pourcentage de spins orienté up dans la grille initiale (entre 0 et 1)

        if previous_lattice is not None:
            self.lattice = previous_lattice
        else:
            self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        """
        Initialise une grille avec un certain pourcentage de spins orienté up ou down (1 ou -1).

        Renvoie :
            np.ndarray : Grille de spins initialisée.
        """

        init_lattice = np.random.random((self.size, self.size))

        return np.where(init_lattice < self.up_perc, 1, -1).astype("int8")
    
   
    def run(self, betaJ):
        """
        Exécute l'algorithme de Metropolis.

        Args:
            betaJ (float): Valeur de beta * J (normalisée).

        Returns:
            tuple: Liste des grilles, liste des moyennes des spins, liste des énergies.
        """
        return find_equilibrium(self.lattice, self.n_iter, self.betaJ, self.B, self.size)



# ----------Exemple d'utilisation de la classe Metropolis----------
# L'idéal serait d'importer cette classe dans un autre fichier pour l'utiliser.


start_time = time.time()

# Créer une instance de la classe Metropolis avec les paramètres souhaités
metropolis = Metropolis(n_iter=100000, lattice_size=100, magnetic_field=0, temperature=1.0, J=1.0)

# Trouver l'état d'équilibre en utilisant la méthode `run`
lattice, spin_means, energy_list = metropolis.run(betaJ=0.7)

step_algo = np.arange(0, len(spin_means), 1)

print("Temps d'exécution : ", time.time() - start_time)

plt.figure(1)
plt.plot(step_algo, spin_means)
plt.xlabel("Étape")
plt.ylabel(r"$\langle M \rangle $")

plt.figure(2)
plt.plot(step_algo, energy_list)
plt.xlabel("Étape")
plt.ylabel(r"$E/J$")

plt.figure(3)
plt.imshow(lattice, vmin=-1, vmax=1, cmap="seismic")
plt.title("Grille finale")
plt.xticks([])
plt.yticks([])
plt.show()