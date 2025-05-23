import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import scipy.ndimage as sc
import scipy.constants as cte
import time
import os
import sys

## Simuler en batch. Parallélisation? -> Chiant à mettre sur windows, donc devra rester sur la tour.
## Quantification de l'aire de la courbe pour constante selon betaJ? 
## Lien avec dimension fractale?
## Permettre des CF ouvertes et fermées pour ajouter un élément de discussion.
## Critère de convergence plutôt qu'un nombre de pas imposé (avec tout même limite "dure" de pas au cas où)

def convolution_2d_fft(lattice, kernel):
    """
    Effectue une convolution 2D en exploitant la FFT. Notion cool de PM2!
    """
    if lattice.shape[0] % 2 == 0:
        raise ValueError("La dimension de la grille doit être impaire.") # Petite contrainte tannante par contre.
    
    # On effectue la transformée de Fourier des deux matrices.
    lattice_fourier = np.fft.fft2(lattice)
    #mask_fourier = np.fft.fft2(np.flipud(np.fliplr(mask))) 
    kernel_fourier = np.fft.fft2(np.fft.ifftshift(kernel), s=lattice.shape)

   # Multiplication dans le domaine fréquentiel
    cc_fourier = lattice_fourier * kernel_fourier

    # Transformée inverse pour revenir dans l'espace réel
    cc = np.real(np.fft.ifft2(cc_fourier))

    # On multiplie les transformées dans le domaine fréquentiel.
    #mask_fourier_padded = np.pad(mask_fourier, (lattice.shape[0]-3+1)//2) 
    #cc = np.real(np.fft.ifft2(lattice_fourier * mask_fourier_padded))  # Transformée inverse pour revenir dans l'espace réel

    # On centre le résultat.
    #m, n = lattice.shape
    #cc = np.roll(cc, -m // 2 + 1, axis=0)
    #cc = np.roll(cc, -n // 2 + 1, axis=1)

    return cc

#@numba.njit
def convolution_2d_mitaine(lattice, kernel):
    """
    Effectue une convolution 2D.
    Forme compatible avec une accélération par Numba.
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
                    ni = (i + u) % N # Conditions périodiques en x
                    nj = (j + v) % M # Conditions périodiques en y
                    acc += lattice[ni, nj] * kernel[u + kh2, v + kw2]
            result[i, j] = acc

    return result

def extended_lattice(lattice): 
    """
    Applique les CF périodiques (Pac-Man) à la grille 2D.
    L'astuce repose sur l'ajout de bordures de 1 ligne/colonne autour de la grille. 
    La dimension effective de la grille est donc de (N+2) x (N+2). 
    """
    n = lattice.shape[0] # Taille de la grille

    lattice_extended = np.concatenate([lattice, lattice, lattice], axis=1)  # On applique les CF à gauche et à droite
    lattice_extended = np.concatenate([lattice_extended, lattice_extended, lattice_extended], axis=0)  # On applique les CF en haut et en bas.

    return lattice_extended[n-1:2*n+1, n-1:2*n+1] 

def correlation_energy(lattice, method="scipy"):
    # La convolution revient à faire la somme sur les s_j en prenant compte du fait que j correspond aux plus proches voisins
    match method:
        case "fft":
            mask = np.ones((3,3), dtype=int)  # Matrice 2D avec 1 seulement aux voisins plus proche (connectivité=1).
            mask[1,1] = 0  # Le spin est lui-même exclu de la somme.
            grille_etendue = extended_lattice(lattice)  # On applique les CF périodiques (Pac-Man)
            energy_array = -lattice * convolution_2d_fft(grille_etendue, mask)[1:lattice.shape[0]+1,1:lattice.shape[0]+1]

            return np.sum(energy_array)

        case "mitaine":
            mask = np.ones((3,3), dtype=int)  # Matrice 2D avec 1 seulement aux voisins plus proche (connectivité=1).
            mask[1,1] = 0  # Le spin est lui-même exclu de la somme.
            energy_array = -lattice * convolution_2d_mitaine(lattice, mask)#[1:lattice.shape[0]+1,1:lattice.shape[0]+1]
            return np.sum(energy_array)
        
        case "scipy":
            mask = sc.generate_binary_structure(2,1)  # Matrice 2D avec True seulement aux voisins plus proche (connectivité=1)
            mask[1,1] = False  # On veut pas compter le spin lui même dans la somme

            # On applique les conditions frontières périodiques avec l'argument wrap. 
            energy_array = -lattice * sc.convolve(lattice, mask, mode='wrap')  

            return np.sum(energy_array)
        
        case default: # Méthode SciPy par défaut
            mask = sc.generate_binary_structure(2,1)  # Matrice 2D avec True seulement aux voisins plus proche (connectivité=1)
            mask[1,1] = False  # On veut pas compter le spin lui même dans la somme

            # On applique les conditions frontières périodiques avec l'argument wrap. 
            energy_array = -lattice * sc.convolve(lattice, mask, mode='wrap')  

            return np.sum(energy_array)

    
def microstate_energy(lattice, h, method="scipy"):
    """
    Calcule l'énergie totale d'un micro-état donné (lattice : configuration de spins; h : composante Z du champ magnétique normalisée avec J).

    On doit tenir compte de deux contributions : 
        1) les voisins immédiats;
        2) le champ magnétique externe.
    """
    N = lattice.shape[0]
    energie_mag = 0.0
    energie_corr = 0.0

    # Contribution du champ magnétique externe
    energie_mag -= h * np.sum(lattice)  # Utilisation de la somme vectorisée pour accélérer le calcul.
    # Contribution des interactions entre voisins
    energie_corr = correlation_energy(lattice, method)

    return energie_mag + energie_corr

def find_equilibrium(lattice, betaJ, h, size, energy, run_max = True, n_iter_max=300000, fluct_eq=0.003, buffer=1000, seed=None):
    """
    Trouve l'état d'équilibre en utilisant l'algorithme de Metropolis-Hastings.

    Entrée :
        betaJ (float): Valeur de beta * J (normalisée).
        h (float): Champ magnétique externe normalisé par la constante de couplage (h = H/J).

    Sortie :
        tuple: Liste des grilles, énergie finale, liste des moyennes des spins, liste des énergies.
    """
    rng = np.random.default_rng(seed)  # Générateur indépendant. 

    spin_mean_list = [np.mean(lattice)]
    energy_list = [energy]
    lattice_list = [lattice.copy()]

    energy_fluctuation = 1e6 # Initialisation de la variation d'énergie à une valeur assez élevée
    iter = 0 # Initialisation du compteur


    while run_max or energy_fluctuation >= fluct_eq: # Soit on a un while True (n_iter prescrit), soit on vérifie la variation d'énergie
        print("h : ",h, "Itération : ", iter, "Énergie : ", energy, "Fluctuation d'énergie : ", energy_fluctuation) # Affichage de l'itération et de l'énergie actuelle
        if iter >= n_iter_max: # Sortie de boucle si dépassement du nombre d'itérations imposé
            print("Sortie de boucle")
            break

        row = rng.integers(0, size)
        col = rng.integers(0, size)

        new_lattice = lattice.copy()
        new_lattice[row][col] *= -1 # Flippage d'un spin au hasard (teneur Monte Carlo du problème...)
        
        # Terme dû au champ + terme de corrélation avec conditions frontières périodiques.
        # On calcule seulement l'énergie du spin concerné puisque les autres ne changent pas.
        E_i = -h * lattice[row, col] -  lattice[row, col] * (lattice[(row+1) % len(lattice), col] + lattice[(row-1) % len(lattice), col] + lattice[row, (col+1) % len(lattice)] + lattice[row, (col-1) % len(lattice)])
        E_f = -h * new_lattice[row, col] - new_lattice[row, col] * (new_lattice[(row+1) % len(lattice), col] + new_lattice[(row-1) % len(lattice), col] + new_lattice[row, (col+1) % len(lattice)] + new_lattice[row, (col-1) % len(lattice)])
        DeltaE = E_f - E_i  # Variation d'énergie

        # Si l'énergie du nouveau microétat est plus grande, on flippe seulement avec la probabilité donnée par l'équation avec l'exponentielle
        if DeltaE > 0 and rng.random() < np.exp(-betaJ * DeltaE):  
            lattice = new_lattice
            energy += DeltaE
            
        # Si l'énergie est plus petite, on flippe (100% de chance)
        elif DeltaE <= 0:
            lattice = new_lattice  
            energy += DeltaE
        spin_mean_list.append(np.mean(lattice))
        energy_list.append(energy)
        lattice_list.append(lattice.copy())

        if iter < 2*buffer:
            energy_fluctuation = 1e6
        else:
            # On calcule la fluctuation d'énergie sur le buffer de points précédents
            energy_fluctuation = np.std(energy_list[-buffer:]) / np.abs(np.mean(energy_list[-buffer:]))

        iter += 1

    return lattice_list, spin_mean_list, energy_list, energy_list[-1]

class Metropolis():
    def __init__(self, lattice_size, betaJ, magnetic_field, energy=None, previous_lattice=None, pourcentage_up=0.60, run_max=True, convol="scipy", n_iter_max=300000, fluct_eq=0.003, buffer=1000, seed=None):
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
        #self.temperature = temperature # Température
        self.run_max = run_max  # Si True, on fait un while True (n_iter prescrit), sinon on vérifie la variation d'énergie.
        #self.betaJ = couplage / (cte.k * temperature)  # Calcul de betaJ à partir de la température et du couplage
        self.betaJ = betaJ
        self.up_perc = pourcentage_up  # Pourcentage de spins orienté up dans la grille initiale (entre 0 et 1)
        self.convol = convol # Méthode de convolution. Comparer les méthdodes devient un élément de discussion intéressant.
        self.n_iter_max = n_iter_max # Si ça converge pas, permet d'éviter le freeze.
        self.fluct_eq = fluct_eq # Différence d'énergie (petite) comme critère de convergence.
        self.buffer = buffer # Taille du buffer contenant les valeurs d'énergie pour la variation considérée.
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)  # Générateur de seed pseudo-aléatoire indépendant. 

        if previous_lattice is not None:
            self.lattice = previous_lattice
            self.energy = energy
        else:
            self.lattice = self.initialize_lattice()
            self.energy = microstate_energy(self.lattice, self.h, method=self.convol)

    def initialize_lattice(self):
        """
        Initialise une grille avec un certain pourcentage de spins orienté up ou down (1 ou -1).

        Renvoie :
            np.ndarray : Grille de spins initialisée.
        """

        init_lattice = self.rng.random((self.size, self.size))

        return np.where(init_lattice < self.up_perc, 1, -1).astype("int8")
    
    def summary(self):
        return {
            "seed": self.seed,
            "lattice_shape": self.lattice.shape,
            "initial_magnetization": np.mean(self.lattice),
            "magnetic_field": self.h,
            "betaJ": self.betaJ,
            "convol": self.convol
        }
    
    def run(self):
        """
        Exécute l'algorithme de Metropolis.

        Returns:
            tuple: Liste des grilles, liste des moyennes des spins, liste des énergies.
        """

        return find_equilibrium(self.lattice, self.betaJ, self.h, self.size, run_max=self.run_max, n_iter_max=self.n_iter_max, fluct_eq=self.fluct_eq, buffer=self.buffer, seed=self.seed)



h_list = np.concatenate((np.arange(-1, 1, 0.1), np.arange(1, -1, -0.1)))
spin_mean_list = []
metro = Metropolis(64, betaJ=0.5, magnetic_field=-1.0, energy=None , convol="scipy", pourcentage_up=0.0, run_max=True, n_iter_max=30000, fluct_eq=0.0025, buffer=10000, seed=None)

for i in range(len(h_list)):
    metro.h = h_list[i]  # On change le champ magnétique pour la prochaine itération
    lattices, spin_means, energy_list, energy = metro.run()
    spin_mean_list.append(spin_means[-1])
    metro.lattice = lattices[-1]  # On change le champ magnétique pour la prochaine itération


plt.figure(0)
plt.plot(h_list, spin_mean_list)
plt.xlabel("H/J")
plt.ylabel(r"$\langle M \rangle $")
plt.show()

# ----------Exemple d'utilisation de la classe Metropolis----------
# L'idéal serait d'importer cette classe dans un autre fichier pour l'utiliser.

"""
start_time = time.time()

# Créer une instance de la classe Metropolis avec les paramètres souhaités
metropolis = Metropolis(64, betaJ=0.7, magnetic_field=0.0, convol="scipy", pourcentage_up=0.7, run_max=False, n_iter_max=300000, fluct_eq=0.0018, buffer=10000, seed=None)

# Trouver l'état d'équilibre en utilisant la méthode "run"
lattices, spin_means, energy_list = metropolis.run()




step_algo = np.arange(0, len(spin_means), 1) # Itérations de la "descente" Metropolis


print("Temps d'exécution : ", time.time() - start_time)

plt.figure(1)
plt.plot(step_algo, spin_means)
plt.xlabel("Pas de la simulation")
plt.ylabel(r"$\langle M \rangle $")
plt.title("Magnétisation")

plt.figure(2)
plt.plot(step_algo, energy_list)
plt.xlabel("Pas de la simulation")
plt.ylabel(r"$E/J$")
plt.title("Énergie")

plt.figure(3)
plt.imshow(lattices[0], vmin=-1, vmax=1)
plt.title("Grille initiale")
plt.xticks([])
plt.yticks([])

plt.figure(4)
plt.imshow(lattices[-1], vmin=-1, vmax=1)
plt.title("Grille finale")
plt.xticks([])
plt.yticks([])

plt.show()
"""