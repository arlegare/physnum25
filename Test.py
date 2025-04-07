import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.ndimage as sc
import scipy.constants as cte
#from matplotlib.animation import FuncAnimation
import time


@numba.njit
def initialize_lattice(size, pourcentage_up=0.70):
    lattice = np.random.random((size, size))
    return np.where(lattice < pourcentage_up, 1, -1)

@numba.njit
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

# La convolution revient à faire la somme sur les s_j, où j correspond aux plus proches voisins (du mask).
"""
def convolution_2d_avec_fft(lattice, mask):

    if lattice.shape[0] % 2 == 0:
        raise ValueError("La dimension de la grille doit être impaire.")
    
    # On effectue la transformée de Fourier des deux matrices.
    lattice_fourier = np.fft.fft2(lattice)
    mask_fourier = np.fft.fft2(np.flipud(np.fliplr(mask))) 

    # On multiplie les transformées dans le domaine fréquentiel.
    mask_fourier_padded = np.pad(mask_fourier, (lattice.shape[0]-3+1)//2) 
    cc = np.real(np.fft.ifft2(lattice_fourier * mask_fourier_padded))  # Transformée inverse pour revenir dans l'espace réel

    # On centre le résultat.
    m, n = lattice.shape
    cc = np.roll(cc, -m // 2 + 1, axis=0)
    cc = np.roll(cc, -n // 2 + 1, axis=1)

    return cc
"""

@numba.njit
def convolution_2d_periodic(lattice, kernel):
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
                    ni = (i + u) % N
                    nj = (j + v) % M
                    acc += lattice[ni, nj] * kernel[u + kh2, v + kw2]
            result[i, j] = acc

    return result

"""
@numba.njit
def microstate_energy(lattice, h):

    energie_totale = 0 # Énergie totale nulle!

    # On commence par la contribution du champ magnétique externe.
    for i in range(len(lattice[0])):
        for j in range(len(lattice[0])):
            energie_totale -= h * lattice[i, j] # Signe (-) car si spin est dans la même direction que le champ, l'énergie est minimisée.

    #mask = sc.generate_binary_structure(2,1)  # Matrice 2D avec True seulement aux voisins plus proche (connectivité=1).
    mask = np.ones((3,3), dtype=int)  # Matrice 2D avec 1 seulement aux voisins plus proche (connectivité=1).
    mask[1,1] = 0  # Le spin est lui-même exclu de la somme.

    # Utilisation de la fonction convolution 2D calculer l'énergie des interactions.
    grille_etendue = extended_lattice(lattice)  # On applique les CF périodiques (Pac-Man)
    energies_interaction_vec = -lattice * convolution_2d_periodic(grille_etendue, mask)[1:lattice.shape[0]+1,1:lattice.shape[0]+1]

    return energie_totale + np.sum(energies_interaction_vec)  # 2 contributions à l'énergie : 1) champ magnétique; 2) spins voisins.
"""

@numba.njit
def microstate_energy(lattice, h):
    """
    Calcule l'énergie totale d'un micro-état donné (lattice : configuration de spins; h : composante Z du champ magnétique).

    On doit tenir compte de deux contributions : 
        1) les voisins immédiats;
        2) le champ magnétique externe.
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


@numba.njit(nogil=True)
def find_equilibrium(betaJ, h,  lattice, n_iter, energy):
    # BetaJ vu q'on a normalisé. Revient à diviser par J dans la formule de l'énergie
    # On commence par définir une nouvelle grille où on a flippé un spin aléatoirement
    # Créer une copie de lattice en premier
    list_lattices = [lattice.copy()] # Probably une meilleure façon de le faire mais je met une liste de lattices pour faire l'animation plus tard. On peut pas mettre des trucs de matplotlib dans une foncion s'il y a numba
    spin_mean_list = [np.mean(lattice)]
    energy_list = [energy]

    for _ in range(n_iter):
        new_lattice = lattice.copy()
        row, col = np.random.randint(0, len(lattice[0])), np.random.randint(0, len(lattice[0]))
        new_lattice[row][col] *= -1 # Flip un spin au hasard
        
        # Terme dû au champ + terme de corrélation avec conditions frontières périodiques
        # On calcule seulement l'énergie du spin concerné puisque les autres ne changent pas
        E_i = -h * lattice[row, col] -  lattice[row, col] * (lattice[(row+1) % len(lattice), col] + lattice[(row-1) % len(lattice), col] + lattice[row, (col+1) % len(lattice)] + lattice[row, (col-1) % len(lattice)])
        E_f = -h * new_lattice[row, col] - new_lattice[row, col] * (new_lattice[(row+1) % len(lattice), col] + new_lattice[(row-1) % len(lattice), col] + new_lattice[row, (col+1) % len(lattice)] + new_lattice[row, (col-1) % len(lattice)])

        DeltaE = E_f - E_i # Variation d'énergie.

        if DeltaE > 0 and np.random.random() < np.exp(-betaJ*DeltaE):  # Si l'énergie du nouveau microétat est plus grande, on flip seulement avec la probabilité donnée par l'équation avec l'exponentielle
            lattice = new_lattice
            energy += DeltaE
        elif DeltaE <= 0:
            lattice = new_lattice  # Si l'énergie est plus petite on flip (100% de chance)
            energy += DeltaE

        spin_mean_list.append(np.mean(lattice))
        energy_list.append(energy)
        list_lattices.append(lattice)

    return list_lattices, energy, spin_mean_list, energy_list


start_time = time.time()

initial_lattice = initialize_lattice(100) 
energy = microstate_energy(initial_lattice, 0)
lattices, energy, spin_means, energy_list = find_equilibrium(0.7, 0, initial_lattice, 100000, energy) 
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
plt.imshow(lattices[-1], vmin=-1, vmax=1, cmap="seismic")
plt.title("Grille finale")
plt.xticks([])
plt.yticks([])

plt.figure(4)
plt.imshow(lattices[0], vmin=-1, vmax=1, cmap="seismic")
plt.title("Grille initiale")
plt.xticks([])
plt.yticks([])
plt.show()
