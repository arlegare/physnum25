import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
from tqdm import tqdm
import sys
import scipy as sp
from numba import njit

@njit(nogil=True)
def pseudo_random_generator(seed, size, offset=0):
    """
    Un générateur de nombres pseudo-aléatoires simple utilisant la méthode du générateur congruentiel linéaire (LCG).
    Cela peut être accéléré avec @njit(nogil=True).

    Entrée :
        seed : int, la valeur initiale de la seed.
        size : int, le nombre de nombres aléatoires à générer.
        offset : int, le nombre de valeurs échantillonnées auparavant pour se situer sur la séquence pseudo-aléatoire.

    Retourne :
        Un tableau numpy 1D de nombres pseudo-aléatoires dans l'intervalle [0, 1).
    """
    a = 1664525  # Multiplicateur
    c = 1013904223  # Incrément
    m = 2**32  # Module
    random_numbers = np.empty(size, dtype=np.float64)
    state = seed

    # Avancer l'état pour atteindre l'offset
    for _ in range(offset):
        state = (a * state + c) % m

    # Générer les nombres pseudo-aléatoires
    for i in range(size):
        state = (a * state + c) % m
        random_numbers[i] = state / m

    return random_numbers

@njit(nogil=True)
def metropolis_fast(lattice, h, betaJ, n_iter, seed=None, seed_offset=0, save_all=False, verbose=False):
    """
    Version optimisée de l'algorithme Metropolis. 
       Celui-ci utilise la fonction njit de numba pour compiler le code en C et l'accélérer. Cependant, celle-ci ne permet pas d'utiliser un seed aléatoire ou des fonctions Scipy.

    Arguments:
        lattice : matrice 2D de spins (1 ou -1)
        h : champ magnétique
        betaJ : beta * J
        n_iter : nombre d'itérations de l'algorithme
        seed : int, graine pour le générateur pseudo-aléatoire (optionnel)
        seed_offset : int, nombre de pas à avancer dans la séquence pseudo-aléatoire (optionnel)
        save_all : bool, si True, sauvegarde tous les états intermédiaires du réseau
        verbose : bool, si True, affiche les informations de progression
    """
    
    size = lattice.shape[0]
    energy = -h * lattice.sum()
    # On commence par calculer l'énergie de la grille. On doit utiliser des boucles fort puisque Numba ne supporte pas les fonctions de convolution de Scipy
    for row in range(size):
        for col in range(size):
            energy += -lattice[row, col] * (
                lattice[(row+1)%size, col] +
                lattice[row, (col+1)%size]
            )
    # On initialise les listes de sauvegarde
    spin_mean_list = [np.mean(lattice)] 
    energy_list = [energy]                
    list_lattices = [lattice.copy()]       


    # On produit d'avance la séquence de nombre pseudo-aléatoires.
    random_numbers = pseudo_random_generator(seed, int(3*n_iter), seed_offset)

    for iter in range(n_iter):
        # Permet d'afficher le progrès de l'algorithme tous les 1000 itérations
        if iter % 1000 == 0 and verbose:
            print("h : ", h, "Itération:", iter, "Énergie:", energy)

        use_precomputed_random = seed is not None

        # On choisit un spin aléatoire à retourner
        if use_precomputed_random:
            row = int(random_numbers[3*iter] * size)
            col = int(random_numbers[3*iter+1] * size)
            r = random_numbers[3*iter+2]
        else:
            row = np.random.randint(0, size)
            col = np.random.randint(0, size)
            r = np.random.rand()

        s = lattice[row, col] # Le spin qu'on va potentiellement changer de signe

        # Calcul de la somme des voisins pour le spin choisi (terme de corrélations)
        neighbors_sum = (
            lattice[(row+1)%size, col]
            + lattice[(row-1)%size, col]
            + lattice[row, (col+1)%size]
            + lattice[row, (col-1)%size]
        )

        DeltaE = 2 * s * (h + neighbors_sum) # Raccourci pour calculer l'énergie du spin concerné. Puisqu'un seul spin change de spin, cela revient à multiplier par 2 l'énergie du spin concerné (voir le rapport).

        # On applique la condition de Metropolis. Si l'énergie est plus faible, on flip le spin. Sinon, on flip avec une probabilité donnée par la distribution de Boltzmann.
        r = random_numbers[3*iter+2] if seed is not None else np.random.rand()
        if DeltaE <= 0 or r < np.exp(-betaJ * DeltaE):
            lattice[row, col] *= -1
            energy += DeltaE

        spin_mean_list.append(np.mean(lattice))
        energy_list.append(energy)

        # On sauvegarde l'état du réseau tous les 2000 itérations si save_all est False. Sinon, on sauvegarde à chaque itération
        #if iter % 2000 == 0 and not save_all:
        if save_all or iter % 2000 == 0:
            list_lattices.append(lattice.copy())
        else:
            list_lattices.append(lattice.copy())

    return lattice, energy, spin_mean_list, energy_list, list_lattices, int(3*n_iter)


def standardize(matrice):  # standardisation des entrées de la matrice
    factor = (np.max(matrice)-np.min(matrice))**(-1)
    transl = np.min(matrice)
    return factor*(matrice - transl)

def get_control_points(p0, p1, p2, t=0.5):
    d01 = np.linalg.norm(p1 - p0)
    d12 = np.linalg.norm(p2 - p1)
    fa = t * d01 / (d01 + d12) 
    fb = t * d12 / (d01 + d12) 
    p1a = p1 - fa * (p2 - p0)  
    p1b = p1 + fb * (p2 - p0)  
    return p1a, p1b

def bezier_curve(p0, p1, p2, p3, n_points=100):
    t = np.linspace(0, 1, n_points)[:, None]
    curve = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return curve

# Fonction pour calculer l'ordonnée du polynôme de meilleur ajustement
def polynomial_fit(x,args):
    ordonnee = 0
    for i,arg in enumerate(args):
        pow = len(args)-i-1
        #print(pow)
        #print(arg)
        ordonnee += arg*x**pow
    return ordonnee

def polyfit_maximum(x,y,deg, res_factor, show=False): # x, y, degré, facteur d'augmentation de densité de points
    x_fit = np.linspace( np.min(x), np.max(x), int(len(x)*res_factor) )
    polyfit_args = np.polyfit(x, y,  deg) # coefficients du polynôme de meilleur ajustement linéaire de degré N
    polyn = [ polynomial_fit(x_, polyfit_args) for x_ in x_fit]
    if show:
        plt.scatter(x,y, color="black")
        plt.plot(x_fit, polyn, linestyle="--", color="gray")
        plt.plot()
        plt.show()

    max_id = np.argmax(polyn)
    return (x_fit[max_id], np.max(polyn)) # On renvoie le minimum absolu en coordonnées (x,y)

def gaussienne(x, A, B, C): 
    return A*np.exp(-1*B*(x-C)**2) 

def sigmoid(x, x0, k, A):
     y = A / (1 + np.exp(-k*(x-x0)))
     return y

def convolve_with_exp(ts, tau):
    t = np.linspace(0, ts[-1], ts.shape[0])
    return sp.signal.convolve(ts, exponential(t, tau))[:len(t)]

@njit
def exponential(t, tau):
    return np.exp(-1 * (t / tau))

@njit
def compute_phase_coherence(signal1, signal2):
    complex_phase_difference = np.exp(1j * (signal1 - signal2))
    R = np.abs(np.mean(complex_phase_difference))
    return R

def compute_order(ts, series_out=False, last_only=False): # on prend en argument une liste de séries temporelles
    if series_out:
        return np.abs(np.mean(np.exp(1j*ts), axis=0))
    elif not series_out and not last_only:
        return np.abs(np.mean(np.mean(np.exp(1j*ts), axis=0)))
    elif last_only:
        return np.abs(np.mean(np.exp(1j*ts[:,-1]), axis=0))

@njit
def fact(x):
    val = 1 

    for i in range(int(x)-1):
        val *= (x-i)
    return val

@njit
def compute_phase_coherence_matrix(ts):
    nodes_n_rem = ts.shape[0]
    C_ij = np.zeros((nodes_n_rem, nodes_n_rem))
    for i in range(nodes_n_rem):
        for j in range(i + 1, nodes_n_rem):
            C_ij[i,j] = np.abs(np.mean(np.exp(1j * (ts[i,:] - ts[j,:]))))
    return (C_ij+C_ij.T)

@njit
def compute_phase_coherence(signal1, signal2):
    complex_phase_difference = np.exp(1j * (signal1 - signal2))
    R = np.abs(np.mean(complex_phase_difference))
    return R

def identify_files(path, keywords=None, exclude=None):
    items = os.listdir(path)
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            if any(excluded in item for excluded in exclude):
                pass
            else:
                files.append(item)
    files.sort()
    return files

def load_hdf5(path):
    data = {}
    file = h5py.File(path, 'r')
    for dataset in file.keys():
        data[dataset] = np.array(file[dataset])
    file.close()
    return data

def save_hdf5(path, dictionary):
    datasets = list(dictionary.keys())
    file = h5py.File(path, 'w')
    for dataset in datasets:
        file.create_dataset(dataset, data=dictionary[dataset])
    file.close()

def baseline_minfilter(signal, window=300, sigma1=5, sigma2=100, debug=False):
    signal_flatstart = np.copy(signal)
    signal_flatstart[0] = signal[1]
    smooth = sp.ndimage.gaussian_filter1d(signal_flatstart, sigma1)
    mins = sp.ndimage.minimum_filter1d(smooth, window)
    baseline = sp.ndimage.gaussian_filter1d(mins, sigma2)
    if debug:
        debug_out = np.asarray([smooth, mins, baseline])
        return debug_out
    else:
        return baseline

def compute_dff_using_minfilter(timeseries, window=200, sigma1=0.1, sigma2=50):
    dff = np.zeros(timeseries.shape)
    for i in range(timeseries.shape[0]):
        if np.any(timeseries[i]):
            baseline = baseline_minfilter(timeseries[i], window=window, sigma1=sigma1, sigma2=sigma2)
            dff[i] = (timeseries[i] - baseline) / (baseline+1)
    return dff

@njit
def compute_correlation_matrix(timeseries, set_nan_to_one=True):
    N = timeseries.shape[0]
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            try:
                matrix[i, j] = np.corrcoef(timeseries[i], timeseries[j])[0, 1]
            except:
                matrix[i, j] = 1
            matrix[j, i] = matrix[i, j]
    return matrix

def correlate_matrices(matrix1, matrix2, choice=False):
    triangle = np.triu_indices(matrix1.shape[0], 1)
    r1 = sp.stats.pearsonr(matrix1[triangle], matrix2[triangle])[0]
    r2 = sp.stats.spearmanr(matrix1[triangle], matrix2[triangle])[0]
    r = [r1, r2]
    if choice:
        return r[np.argmax(r)]
    else:
        return r1

def delete(array, deleted):
    truncated = np.copy(array)
    truncated = np.delete(truncated, deleted, axis=0)
    truncated = np.delete(truncated, deleted, axis=1)
    return truncated

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

def fill_index(id):
    if len(str(id)) == 1:
        return "0" + str(id)
    else:
        return str(id)

def do_nothing():
    return None

def colors(n):
    couleurs = ["aqua", 
                "aquamarine", 
                "azure", 
                "beige", 
                "black", 
                "blue",
                "brown",
                "chartreuse",
                "chocolate",
                "coral",
                "crimson",
                "cyan", 
                "darkblue",
                "darkgreen", 
                "fuchsia",
                "gold",
                "goldenrod",
                "green",
                "grey",
                "indigo",
                "ivory",
                "khaki",
                "lavender",
                "lightblue",
                "lightgreen",
                "lime",
                "magenta", 
                "maroon",
                "navy",
                "olive",
                "orange",
                "orangered",
                "orchid",
                "pink",
                "plum",
                "purple",
                "red",
                "salmon",
                "sienna",
                "silver",
                "tan",
                "teal",
                "tomato",
                "turquoise",
                "violet",
                "wheat",
                "white",
                "yellow",
                "yellowgreen"
               ]

    return couleurs[0:n]