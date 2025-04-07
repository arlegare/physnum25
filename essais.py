import numpy as np
import matplotlib.pyplot as plt
import numba as num
import scipy.ndimage as sc
import scipy.constants as cte

class Metropolis():
    def __init__(self, n_iter, lattice_size, magnetic_field, temperature, J, previous_lattice=None):
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
        self.n_iter = n_iter
        self.size = lattice_size
        self.B = magnetic_field
        self.T = temperature
        self.J = J
        self.up_perc = pourcentage_up

        if previous_lattice:
            self.lattice = previous_lattice
        else:
            self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        """
        Initialise une grille de spins avec des valeurs aléatoires (1 ou -1).

        Returns:
            np.ndarray: Grille de spins initialisée.
        """
        lattice = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if np.random.random() > 0.5:
                    lattice[i][j] = 1
                else:
                    lattice[i][j] = -1
        return lattice

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
        Calcule l'énergie totale d'un micro-état donné.

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
                energie_totale -= h * lattice[i, j]

        # Contribution des interactions entre voisins
        for i in range(N):
            for j in range(N):
                spin = lattice[i, j]
                voisinage = (
                    lattice[(i + 1) % N, j] +
                    lattice[(i - 1) % N, j] +
                    lattice[i, (j + 1) % N] +
                    lattice[i, (j - 1) % N]
                )
                energie_totale -= spin * voisinage

        return energie_totale

    def find_equilibrium(betaJ, h, lattice, n_iter, energy):
        """
        Trouve l'état d'équilibre en utilisant l'algorithme de Metropolis.

        Args:
            betaJ (float): Valeur de beta * J (normalisée).
            h (float): Champ magnétique externe.
            lattice (np.ndarray): Grille de spins initiale.
            n_iter (int): Nombre d'itérations.
            energy (float): Énergie initiale du système.

        Returns:
            tuple: Liste des grilles, énergie finale, liste des moyennes des spins, liste des énergies.
        """
        list_lattices = [lattice.copy()]
        spin_mean_list = [np.mean(lattice)]
        energy_list = [energy]

        for _ in range(n_iter):
            new_lattice = lattice.copy()
            row, col = np.random.randint(0, len(lattice[0])), np.random.randint(0, len(lattice[0]))
            new_lattice[row][col] *= -1

            E_i = -h * lattice[row, col] - lattice[row, col] * (
                lattice[(row + 1) % len(lattice), col] +
                lattice[(row - 1) % len(lattice), col] +
                lattice[row, (col + 1) % len(lattice)] +
                lattice[row, (col - 1) % len(lattice)]
            )
            E_f = -h * new_lattice[row, col] - new_lattice[row, col] * (
                new_lattice[(row + 1) % len(lattice), col] +
                new_lattice[(row - 1) % len(lattice), col] +
                new_lattice[row, (col + 1) % len(lattice)] +
                new_lattice[row, (col - 1) % len(lattice)]
            )

            DeltaE = E_f - E_i

            if DeltaE > 0 and np.random.random() < np.exp(-betaJ * DeltaE):
                lattice = new_lattice
                energy += DeltaE
            elif DeltaE <= 0:
                lattice = new_lattice
                energy += DeltaE

            spin_mean_list.append(np.mean(lattice))
            energy_list.append(energy)
            list_lattices.append(lattice)

        return list_lattices, energy, spin_mean_list, energy_list
