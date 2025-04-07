import numpy as np
import matplotlib.pyplot as plt
import numba as num
import scipy.ndimage as sc
import scipy.constants as cte

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
    
        self.n_iter = n_iter  # Nb d'itérations/steps pour la simulation (choisir assez long pour atteindre l'équilibre pour une valeur de h,T etc)
        self.size = lattice_size  # Dimensions de la grille de spins
        self.B = magnetic_field  
        self.T = temperature
        self.J = J  # Constante de couplage. >0 pour ferromagnétisme et <0 pour antiferromagnétisme.
        self.up_perc = pourcentage_up  # Pourcentage de spins orienté up dans la grille initiale (entre 0 et 1)

        if previous_lattice:
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
    
    def convolution_2d_periodic(self, lattice, kernel):
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

    def microstate_energy(self, lattice, h):

        """
        Calcule l'énergie totale d'un micro-état donné (lattice : configuration de spins; h : composante Z du champ magnétique).

        Args:
            lattice (np.ndarray): Configuration de spins.
            h (float): Composante Z du champ magnétique.

        Returns:
            float: Énergie totale du micro-état.
        """

        N = lattice.shape[0]
        energie_totale = 0.0

        # Contribution du champ magnétique externe
        for i in range(N):
            for j in range(N):
                energie_totale -= h * lattice[i, j]  # Signe (-) car si spin est dans la même direction que le champ, l'énergie est minimisée.

        # Contribution des interactions entre voisins
        for i in range(N):
            for j in range(N):
                spin = lattice[i, j]
                # Conditions périodiques (Pac-Man)
                voisinage = (
                    lattice[(i + 1) % N, j] +  # Voisin du bas
                    lattice[(i - 1) % N, j] +  # Voisin du haut
                    lattice[i, (j + 1) % N] +  # Voisin de droite
                    lattice[i, (j - 1) % N]    # Voisin de gauche
                )
                energie_totale -= spin * voisinage  # Interaction avec les voisins

        return energie_totale
    
    def interaction_energy(self, lattice, row, col, h):
        """
        Calcule l'énergie d'interaction pour un spin donné.

        Args :
            lattice (np.ndarray): Grille de spins.
            row (int): Indice de la ligne du spin.
            col (int): Indice de la colonne du spin.
            h (float): Champ magnétique externe.

        Renvoie :
            float: Énergie d'interaction pour le spin donné.
        """

        spin = lattice[row, col]
        # Conditions périodiques pour les voisins
        voisinage = (
            lattice[(row + 1) % self.size, col] +
            lattice[(row - 1) % self.size, col] +
            lattice[row, (col + 1) % self.size] +
            lattice[row, (col - 1) % self.size]
        )
        # Énergie due au champ magnétique et aux interactions avec les voisins
        return -h * spin - self.J * spin * voisinage

    
    def find_equilibrium(self, betaJ, h):
        """
        Trouve l'état d'équilibre en utilisant l'algorithme de Metropolis.

        Args:
            betaJ (float): Valeur de beta * J (normalisée).
            h (float): Champ magnétique externe.

        Returns:
            tuple: Liste des grilles, énergie finale, liste des moyennes des spins, liste des énergies.
        """
        lattice = self.lattice.copy()
        energy = self.microstate_energy(lattice, h)  # Énergie initiale du système
        # betaJ = J / (cte.physical_constants["Boltzmann constant"][0] * T)  # BetaJ = J / kT

        list_lattices = [lattice.copy()] # Probably une meilleure façon de le faire mais je met une liste de lattices pour faire l'animation plus tard. On peut pas mettre des trucs de matplotlib dans une foncion s'il y a numba
        spin_mean_list = [np.mean(lattice)]
        energy_list = [energy]

        for _ in range(self.n_iter):
            new_lattice = lattice.copy()
            row, col = np.random.randint(0, self.size), np.random.randint(0, self.size)
            new_lattice[row][col] *= -1 # Flip un spin au hasard
            
            # Terme dû au champ + terme de corrélation avec conditions frontières périodiques.
            # On calcule seulement l'énergie du spin concerné puisque les autres ne changent pas.
            # Calcul de l'énergie avant et après le flip
            E_i = self.interaction_energy(lattice, row, col, h)
            E_f = self.interaction_energy(new_lattice, row, col, h)

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
            list_lattices.append(lattice)

        return list_lattices, spin_mean_list, energy_list

# ------------------------------------------------------------------------
# On crée l'instance de la classe Metropolis avec les paramètres souhaités.
metropolis = Metropolis(n_iter=1000, lattice_size=50, magnetic_field=0.5, temperature=1.0, J=1.0)

"""
On trouve la configuration à l'équilibre pour cesdits paramètres gràce à l'algorithme Monte Carlo de Metropolis.
On peut aussi choisir de passer une grille de spins initiale en paramètre (previous_lattice).
Ceci peut s'avérer utile pour faire varier la température et/ou le champ magnétique
puis observer l'évolution conséquente de la grille de spins.
"""

lattices, spin_means, energies = metropolis.find_equilibrium(betaJ=0.1, h=0.5)
print("Énergie finale :", energies[-1])