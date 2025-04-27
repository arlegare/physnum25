import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sc
import time
from numba import njit
from functions import metropolis_kernel

class Metropolis():
    def __init__(self, lattice_size, betaJ, magnetic_field, energy=None, previous_lattice=None, pourcentage_up=0.60, run_max=True, convol="scipy", n_iter_max=100000, fluct_eq=0.003, buffer=5000, seed=None):
        """
        Initialise les paramètres de la simulation de Metropolis.

        Entrée :
            n_iter (int): Nombre d'itérations pour la simulation.
            lattice_size (int): Taille de la grille de spins.
            magnetic_field (float): Champ magnétique externe.
            betaJ (float): Ratio de la constante de couplage J sur k_BT (positif pour ferromagnétisme, négatif pour antiferromagnétisme).
            previous_lattice (np.ndarray, optional): Grille de spins initiale. Si None, une grille sera générée.
        """
    
        self.n_iter_max = n_iter_max  # Nb d'itérations/steps pour la simulation. Par ex. choisir assez long pour atteindre l'équilibre pour une valeur de (T,h).
        self.size = lattice_size  # Dimension de la grille (carrée) de spins. 
        self.h = magnetic_field  # Champ magnétique externe.
        self.run_max = run_max  # Si True, on fait un while True (n_iter prescrit), sinon on vérifie la variation d'énergie.
        self.betaJ = betaJ
        self.up_perc = pourcentage_up  # Pourcentage de spins orienté up dans la grille initiale (entre 0 et 1)
        self.convol = convol # Méthode de convolution. Comparer les méthdodes devient un élément de discussion intéressant.
        self.n_iter_max = n_iter_max # Si ça converge pas, permet d'éviter le freeze.
        self.fluct_eq = fluct_eq # Différence d'énergie (petite) comme critère de convergence.
        self.buffer = buffer # Taille du buffer contenant les valeurs d'énergie pour la variation considérée.
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)  # Générateur de seed pseudo-aléatoire indépendant. 

        self.energy_list = []  # Liste pour stocker les énergies à chaque itération.
        self.spin_mean_list = []  # Liste pour stocker la moyenne des spins à chaque itération.
        self.list_lattices = []  # Liste pour stocker les grilles de spins à chaque itération.


        if previous_lattice is not None:
            self.lattice = previous_lattice
            self.energy = energy
        else:
            self.lattice = self.initialize_lattice()
            self.energy = self.microstate_energy()


    def initialize_lattice(self):
        """
        Initialise une grille avec un certain pourcentage de spins orienté up ou down (1 ou -1).

        Renvoie :
            np.ndarray : Grille de spins initialisée.
        """

        init_lattice = self.rng.random((self.size, self.size))

        return np.where(init_lattice < self.up_perc, 1, -1).astype("int8")


    def microstate_energy(self):
        """
        Calcule l'énergie totale d'un micro-état donné (lattice : configuration de spins; h : composante Z du champ magnétique normalisée avec J).

        On doit tenir compte de deux contributions : 
            1) les voisins immédiats;
            2) le champ magnétique externe.
        """
        energie_mag = 0
        energy_array = 0

        lattice = self.lattice
        h = self.h
        # Contribution du champ magnétique externe
        energie_mag -= h * np.sum(lattice)  # Utilisation de la somme vectorisée pour accélérer le calcul.
        # Contribution des interactions entre voisins
        mask = sc.generate_binary_structure(2,1)  # Matrice 2D avec True seulement aux voisins plus proche (connectivité=1)
        mask[1,1] = False  # On veut pas compter le spin lui même dans la somme
        energy_array = -lattice * sc.convolve(lattice, mask, mode='wrap')  # On applique les conditions frontières périodiques avec l'argument wrap. La convolution revient à faire la somme sur les s_j en prenant compte du fait que j correspond aux plus proches voisins
        
        return energie_mag + np.sum(energy_array)  # On retourne l'énergie totale du micro-état.


    def find_equilibrium(self, fast=True):
        h = self.h
        size = self.size
        betaJ = self.betaJ
        lattice = self.lattice.copy()
        energy = self.energy
        n_iter = self.n_iter_max
        spin_mean_list = [np.mean(lattice)]
        list_lattices = [lattice.copy()]
        energy_list = [energy]
        
        rng = self.rng
        fluct_eq = self.fluct_eq
        run_max = self.run_max
        buffer = self.buffer
        energy_fluctuation = 1e6  # Start with large fluctuation

        if fast:
            lattice, energy, spin_mean_list, energy_list, list_lattices = metropolis_kernel(lattice, h, betaJ, n_iter)
            self.lattice = lattice
            self.energy = energy
            self.spin_mean_list = spin_mean_list
            self.energy_list = energy_list
            return list_lattices, energy, spin_mean_list, energy_list

        else:
            for iter in range(n_iter):
                new_lattice = lattice.copy()
                if iter % 1000 == 0:
                    print(f"h = {h:.2f}, iter = {iter}, E = {energy:.2f}, ΔE_fluct = {energy_fluctuation:.2e}")

                # On flip un spin aléatoire
                row, col = rng.integers(0, size), rng.integers(0, size)
                new_lattice[row, col] *= -1 

                # On calcul l'énergie du spin concerné puisque les autres ne changent pas
                E_i = -h * lattice[row, col] - lattice[row, col] * (
                    lattice[(row+1) % size, col] +
                    lattice[(row-1) % size, col] +
                    lattice[row, (col+1) % size] +
                    lattice[row, (col-1) % size]
                )

                E_f = -h * new_lattice[row, col] - new_lattice[row, col] * (
                    new_lattice[(row+1) % size, col] +
                    new_lattice[(row-1) % size, col] +
                    new_lattice[row, (col+1) % size] +
                    new_lattice[row, (col-1) % size]
                )

                DeltaE = E_f - E_i
                if DeltaE <= 0 or rng.random() < np.exp(-betaJ * DeltaE):
                    lattice = new_lattice
                    energy += DeltaE

                spin_mean_list.append(np.mean(lattice))
                energy_list.append(energy)
                list_lattices.append(lattice.copy()) # On peut mettre condition avec un modulo pour accélérer un peu

                # On calcul la fluctuation en énergie si run_max=False, de sorte que la simulation s'arrête si la fluctuation d'énergie est suffisamment petite.
                if not run_max:
                    if iter < 2*buffer: # Choix arbitraire pour éviter de faire la moyenne sur trop peu de points
                        energy_fluctuation = 1e6
                    else:
                        energy_fluctuation = np.std(energy_list[-buffer:]) / np.abs(np.mean(energy_list[-buffer:])) # Calcul des fluctuations en énergie autour de la moyenne

                    if energy_fluctuation < fluct_eq:
                        break
            
            # Actualisation des attributs de la classe avec les résultats de la simulation
            self.lattice = lattice
            self.energy = energy
            self.spin_mean_list = spin_mean_list
            self.energy_list = energy_list
            self.list_lattices = list_lattices
            return list_lattices, energy, spin_mean_list, energy_list


    def summary(self):
        return {
            "seed": self.seed,
            "lattice_shape": self.lattice.shape,
            "initial_magnetization": np.mean(self.lattice),
            "magnetic_field": self.h,
            "betaJ": self.betaJ,
            "convol": self.convol
        }


    def plot_lattice(self, lattice=None):
        if lattice is None:
            lattice = self.lattice
        plt.imshow(lattice, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Grille de spins")
        plt.colorbar(label="Valeur du spin")
        plt.xticks([])
        plt.yticks([])
        plt.show()


    def plot_energy(self):
        plt.plot(self.energy_list)
        plt.xlabel("Itération")
        plt.ylabel("E/J")
        plt.title("Energie au cours de la simulation (normalisée)")
        plt.show()
    

    def plot_spin_mean(self):
        plt.plot(self.spin_mean_list)
        plt.xlabel("Itération")
        plt.ylabel(r"$\langle M \rangle $")
        plt.title("Moment magnétique moyen au cours de la simulation")
        plt.show()


    def plot_hysteresis(self, h_low=-1, h_high=1, resolution=0.05, fast=True):
        h_list = np.concatenate((np.arange(h_low, h_high, resolution), np.arange(h_high, h_low, -resolution)))
        spin_step_list = []
        for i in range(len(h_list)):
            self.h = h_list[i]  # On change le champ magnétique pour la prochaine itération
            lattices, energy, spin_means, energy_list = metro.find_equilibrium(fast=True)
            spin_step_list.append(spin_means[-1])
            metro.lattice = lattices[-1]  # On change le champ magnétique pour la prochaine itération
        plt.plot(h_list, spin_step_list)
        plt.xlabel("Champ magnétique normalisé (h/J)")
        plt.ylabel(r"$\langle M \rangle $")
        plt.title("Courbe d'hystérèse")
        plt.show()


metro = Metropolis(lattice_size=64, betaJ=0.5, magnetic_field=0.0, pourcentage_up=-1.0, n_iter_max=30000, seed=None, run_max=True, buffer=5000, fluct_eq=0.0015)

metro.plot_hysteresis(h_low=-1, h_high=1, resolution=0.05, fast=True)
