import numpy as np
import matplotlib.pyplot as plt
import numba as num
import scipy.ndimage as sc
import scipy.constants as cte
from matplotlib.animation import FuncAnimation
import time


def initialize_lattice(size, pourcentage_up=0.8):
    #  Initialise une grille avec un certain pourcentage de spins orienté up ou down
    lattice = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if np.random.random() < pourcentage_up:
                lattice[i, j] = 1
            else:
                lattice[i, j] = -1
    return lattice

def microstate_energy(lattice, h):
    #  Faut additionner la somme des voisins les plus proches et prendre en compte la contribution du champ mag
    tot_energy = 0
    # On commence par celle du champ
    for i in range(len(lattice[0])):
        for j in range(len(lattice[0])):
            tot_energy -= h * lattice[i, j] # Signe - car si spin est dans la même direction que le champ, l'énergie est minimisée
    # Funky business pour faire le terme de corrélations
    mask = sc.generate_binary_structure(2,1)  # Matrice 2D avec True seulement aux voisins plus proche (connectivité=1)
    mask[1,1] = False  # On veut pas compter le spin lui même dans la somme
    energy_array = -lattice * sc.convolve(lattice, mask, mode='wrap')  # On applique les conditions frontières périodiques avec l'argument wrap. La convolution revient à faire la somme sur les s_j en prenant compte du fait que j correspond aux plus proches voisins
    return tot_energy + energy_array.sum()

@num.njit(nogil=True)
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
        # On calcul seulement l'énergie du spin concerné puisque les autres ne changent pas
        E_i = -h * lattice[row, col] -  lattice[row, col] * (lattice[(row+1) % len(lattice), col] + lattice[(row-1) % len(lattice), col] + lattice[row, (col+1) % len(lattice)] + lattice[row, (col-1) % len(lattice)])
        E_f = -h * new_lattice[row, col] - new_lattice[row, col] * (new_lattice[(row+1) % len(lattice), col] + new_lattice[(row-1) % len(lattice), col] + new_lattice[row, (col+1) % len(lattice)] + new_lattice[row, (col-1) % len(lattice)])

        DeltaE = E_f - E_i
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

#start_time = time.time()
initial_lattice = initialize_lattice(64, pourcentage_up=0.0)
energy = microstate_energy(initial_lattice, 0)

h_list = np.concatenate((np.arange(-1, 1, 0.05), np.arange(1, -1, -0.05)))
spin_mean_list = []
for i in range(len(h_list)):
    lattices, energy, spin_means, energy_list = find_equilibrium(0.7, h_list[i], initial_lattice, 30000, energy)
    spin_mean_list.append(spin_means[-1])
    initial_lattice = lattices[-1]  # On garde le dernier état comme état initial pour la prochaine itération

plt.figure(0)
plt.plot(h_list, spin_mean_list)
plt.xlabel("h/J")
plt.ylabel("Spin Mean")
plt.show()


#print("Execution time:", time.time() - start_time, "seconds") 
#step_algo = np.arange(0, len(spin_means), 1)

"""
plt.figure(1)
plt.plot(step_algo, spin_means)
plt.xlabel("Step")
plt.ylabel("Spin Mean")

plt.figure(2)
plt.plot(step_algo, energy_list)
plt.xlabel("Step")
plt.ylabel("E/J")

plt.figure(3)
plt.imshow(lattices[-1], vmin=-1, vmax=1)
plt.title("Final Lattice")
plt.xticks([])
plt.yticks([])

plt.figure(4)
plt.imshow(lattices[0], vmin=-1, vmax=1)
plt.title("Initial Lattice")
plt.xticks([])
plt.yticks([])
#plt.show()
"""