import numpy as np
import matplotlib.pyplot as plt
import numba as num
import scipy.ndimage as sc
import scipy.constants as cte
from matplotlib.animation import FuncAnimation

class Metropolis():
    # Changer pour mettre les commentaires format PEP-8
    def __init__(self, n_iter, lattice_size, magnetic_field, temperature, J, previous_lattice = None):
        self.n_iter = n_iter  # Nb d'itérations/steps pour la simulation (choisir assez long pour atteindre l'équilibre pour une valeur de B,T etc)
        self.size = lattice_size  # Dimensions de la grille de spins
        self.B = magnetic_field  
        self.T = temperature
        self.J = J  # Constante de couplage. Positif pour ferromagnétisme et négatif pour antiferromagnétisme
        if previous_lattice:
            self.lattice = previous_lattice
        else:
            self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        #  Initialise une grille avec un certain pourcentage de spins orienté up ou down
        #  Peut-être donner un argument dans innit pour choisir? Sinon on peut juste mettre par défaut une certaine valeur genre 50/50
        #  Je mets 50/50 for now
        lattice = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if np.random.random() > 0.5:
                    lattice[i][j] = 1
                else:
                    lattice[i][j] = -1
        return lattice

    def microstate_energy(self, lattice):
        #  Utile pour déterminer si un spin doit être flippé ou non dans la fonction principale
        #  Faut additionner la somme des voisins les plus proches et prendre en compte la contribution du champ mag
        tot_energy = 0
        # On commence par celle du champ
        for i in range(self.size):
            for j in range(self.size):
                tot_energy -= self.B * cte.physical_constants["Bohr magneton"][0] * lattice[i][j]
        # Funky business pour faire le terme de corrélations
        mask = sc.generate_binary_structure(2,1)  # Matrice 2D avec True seulement aux voisins plus proche (connectivité=1)
        mask[1,1] = False  # On veut pas compter le spin lui même dans la somme
        energy_array = -lattice * self.J * sc.convolve(lattice, mask, mode='wrap')  # On applique les conditions frontières périodiques avec l'argument wrap. La convolution revient à faire la somme sur les s_j en prenant compte du fait que j correspond aux plus proches voisins
        return tot_energy + energy_array.sum()
    
    #@num.njit(nopython=True, nogil=True)
    def find_equilibrium(self):
        # On commence par définir une nouvelle grille où on a flippé un spin aléatoirement
        # Créer une copie de lattice en premier
        list_lattices = [] # Probably une meilleure façon de le faire mais je met une liste de lattices pour faire l'animation plus tard. On peut pas mettre des trucs de matplotlib dans une foncion s'il y a numba 
        for _ in range(self.n_iter):
            list_lattices.append(self.lattice.copy())
            #current_lattice = self.lattice.copy()
            new_lattice = self.lattice.copy()
            row, col = np.random.randint(0, self.size), np.random.randint(0, self.size)
            new_lattice[row, col] *= -1 # Flip un spin au hasard

            DeltaE = self.microstate_energy(new_lattice) - self.microstate_energy(self.lattice)
            if DeltaE > 0 and np.random.random() < np.exp(-1/(cte.Boltzmann * self.T) * DeltaE):  # Si l'énergie du nouveau microétat est plus grande, on flip seulement avec la probabilité donnée par l'équation avec l'exponentielle
                self.lattice = new_lattice
            elif DeltaE < 0:
                self.lattice = new_lattice  # Si l'énergie est plus petite on flip (100% de chance)
        return list_lattices  # On pourrait peut-être return des paramètres de la lattice finale genre magnétisation, spin moyen, énergie moyenne, etc. sinon faire une autre fonction I guess


