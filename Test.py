import numpy as np
import matplotlib.pyplot as plt
import numba as num
import scipy.ndimage as sc
import scipy.constants as cte
from matplotlib.animation import FuncAnimation

magneton = cte.physical_constants["Bohr magneton"][0]
boltzmann = cte.Boltzmann



def initialize_lattice(size):
    #  Initialise une grille avec un certain pourcentage de spins orienté up ou down
    #  Peut-être donner un argument dans innit pour choisir? Sinon on peut juste mettre par défaut une certaine valeur genre 50/50
    lattice = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if np.random.random() > 0.75:
                lattice[i][j] = 1
            else:
                lattice[i][j] = -1
    return lattice

def microstate_energy(lattice, B, J):
    #  Faut additionner la somme des voisins les plus proches et prendre en compte la contribution du champ mag
    tot_energy = 0
    # On commence par celle du champ
    for i in range(len(lattice[0])):
        for j in range(len(lattice[0])):
            tot_energy -= B * cte.physical_constants["Bohr magneton"][0] * lattice[i][j]
    # Funky business pour faire le terme de corrélations
    mask = sc.generate_binary_structure(2,1)  # Matrice 2D avec True seulement aux voisins plus proche (connectivité=1)
    mask[1,1] = False  # On veut pas compter le spin lui même dans la somme
    energy_array = -lattice * J * sc.convolve(lattice, mask, mode='wrap')  # On applique les conditions frontières périodiques avec l'argument wrap. La convolution revient à faire la somme sur les s_j en prenant compte du fait que j correspond aux plus proches voisins
    return tot_energy + energy_array.sum()

@num.njit(nogil=True)
def find_equilibrium(T, B, J, lattice, n_iter, energy):
    # On commence par définir une nouvelle grille où on a flippé un spin aléatoirement
    # Créer une copie de lattice en premier
    list_lattices = [] # Probably une meilleure façon de le faire mais je met une liste de lattices pour faire l'animation plus tard. On peut pas mettre des trucs de matplotlib dans une foncion s'il y a numba
    spin_mean_list = []
    energy_list = []
    for _ in range(n_iter):
        list_lattices.append(lattice.copy())
        new_lattice = lattice.copy()
        row, col = np.random.randint(0, len(lattice[0])), np.random.randint(0, len(lattice[0]))
        new_lattice[row, col] *= -1 # Flip un spin au hasard
        E_i = B * magneton * lattice[row][col]
        E_f = B * magneton * new_lattice[row][col]
        if row == 0:
            E_i -= J * lattice[row][col] *(lattice[row+1][col] + lattice[-1][col])
            E_f -= J * new_lattice[row][col] *(new_lattice[row+1][col] + new_lattice[-1][col])

        if row == (len(lattice)-1):
            E_i -= J * lattice[row][col] * (lattice[row-1][col] + lattice[0][col])
            E_f -= J * new_lattice[row][col] * (new_lattice[row-1][col] + new_lattice[0][col])

        if col == 0:
            E_i -= J * lattice[row][col] * (lattice[row][col+1] + lattice[row][-1])
            E_f -= J * new_lattice[row][col] * (new_lattice[row][col+1] + new_lattice[row][-1])

        if col == (len(lattice)-1):
            E_i -= J * lattice[row][col] * (lattice[row][col-1] + lattice[row][0])
            E_f -= J * new_lattice[row][col] * (new_lattice[row][col-1] + new_lattice[row][0])

        if row != 0 and row != (len(lattice)-1) and col != 0 and col != (len(lattice)-1):   # Je sais pas pourquoi mais si je met juste un else ya un index error
            E_i -= J * lattice[row][col] * (lattice[row][col+1] + lattice[row][col-1] + lattice[row+1][col] + lattice[row-1][col])
            E_f -= J * new_lattice[row][col] * (new_lattice[row][col+1] + new_lattice[row][col-1] + new_lattice[row+1][col] + new_lattice[row-1][col])

        DeltaE = E_f - E_i
        if DeltaE > 0 and np.random.random() < np.exp(-1/(boltzmann * T) * DeltaE):  # Si l'énergie du nouveau microétat est plus grande, on flip seulement avec la probabilité donnée par l'équation avec l'exponentielle
            lattice = new_lattice
            energy += DeltaE
        elif DeltaE < 0:
            lattice = new_lattice  # Si l'énergie est plus petite on flip (100% de chance)
            energy += DeltaE
        spin_mean_list.append(np.mean(lattice))
        energy_list.append(energy)
    return list_lattices, energy, spin_mean_list, energy_list


initial_lattice = initialize_lattice(100)
energy = microstate_energy(initial_lattice, 1, 1)
lattices, energy, spin_means, energy_list = find_equilibrium(1, 0, 1, initial_lattice, 100000, energy) 
time_array = np.arange(0, len(spin_means), 1)

plt.figure(1)
plt.plot(time_array, spin_means)
plt.xlabel("Time")
plt.ylabel("Spin Mean")

plt.figure(2)
plt.plot(time_array, energy_list)
plt.xlabel("Time")
plt.ylabel("Energy")

def animate_lattice(lattices, interval=1):
    fig, ax = plt.subplots()
    ax.set_title("Lattice Animation")
    ax.set_xticks([])
    ax.set_yticks([])

    # Function to update the lattice for each frame
    def update(frame):
        ax.clear()
        ax.set_title("Lattice Animation")
        ax.imshow(lattices[frame], vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(lattices), interval=interval)
    plt.show()

#animate_lattice(lattices)
plt.show()