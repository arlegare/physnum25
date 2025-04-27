import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sc
import time
from numba import njit
from functions import metropolis_fast
from tqdm import tqdm

class Metropolis():
    def __init__(self, lattice_size, betaJ, magnetic_field, energy=None, previous_lattice=None, pourcentage_up=0.60, seed=None, seed_offset=0, verbose=False):
        """
        Initialise les paramètres de la simulation de Metropolis.

        Entrée :
            n_iter (int): Nombre d'itérations pour la simulation.
            lattice_size (int): Taille de la grille de spins.
            magnetic_field (float): Champ magnétique externe normalisée avec J (h = H/J).
            betaJ (float): Ratio de la constante de couplage J sur k_BT (positif pour ferromagnétisme, négatif pour antiferromagnétisme).
            previous_lattice (np.ndarray, optional): Grille de spins initiale. Si None, une grille sera générée.
            pourcentage_up (float): Pourcentage de spins orienté up dans la grille initiale (entre 0 et 1).
            seed (int, optional): Seed pour le générateur de nombres aléatoires. Si None, une seed aléatoire sera utilisée. L'argument peut seulement être utilisé si fast=False dans la fonction find_equilibrium.
        """
    
        self.size = lattice_size  
        self.h = magnetic_field  
        self.betaJ = betaJ
        self.up_perc = pourcentage_up  
        self.seed = seed
        self.seed_offset = 0 # Pour décaler le seed du générateur congruentiel linéaire. On le fait repartir à zéro à chaque nouvelle instance de Metropolis.
        self.rng = np.random.default_rng(self.seed)  # Générateur de seed pseudo-aléatoire indépendant. 
        self.energy_list = []  # Liste pour stocker les énergies à chaque itération.
        self.spin_mean_list = []  # Liste pour stocker la moyenne des spins à chaque itération.
        self.list_lattices = []  # Liste pour stocker les grilles de spins à chaque itération.
        self.verbose = verbose

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


    def find_equilibrium(self, n_iter=30000, buffer = 5000, run_max=True, fluct_eq=0.002, fast=True, save_all=False):
        """
        Trouve l'équilibre du système en utilisant l'algorithme de Metropolis.
        
        Paramètres:
            buffer (int): Taille du buffer pour le calcul de la fluctuation d'énergie. Il s'agit de la fenêtre de points sur laquelle on calcule la fluctuation d'énergie. S'applique seulement si fast=False et run_max=False.
            n_iter (int): Nombre d'itérations maximal pour la simulation.
            run_max (bool): Si True, la simulation s'arrête lorsque la fluctuation d'énergie est suffisamment petite. Sinon, la simulation s'arrête après n_iter_max itérations. Ceci s'applique seulement si fast=False. Dans le cas où fast=True, la simulation s'arrête toujours après n_iter_max itérations.
            fluct_eq (float): Fluctuation d'énergie à atteindre pour considérer que le système est en équilibre. Utilisé seulement si run_max=True.
            fast (bool): Si True, utilise la méthode rapide de Metropolis à l'aide de la fonction metropolis_fast assistée de Numba. Sinon, on utilise la méthode classique sans Numba.
            save_all (bool): Si True, sauvegarde la grille de spins à chaque itération. Sinon, sauvegarde la grille de spins tous les 2000 itérations.
        
        Méthode rapide: Permet d'accélérer le calcul en utilisant la compilation JIT de Numba pour optimiser les boucles et les calculs sur les tableaux NumPy. Cependant, cette méthode ne permet pas d'utiliser la convolution de Scipy pour le calcul de l'énergie de même qu'une seed pour les nombres aléatoires.

        Méthode classique: Utilise la méthode de Metropolis sans Numba, ce qui peut être plus lent mais permet d'utiliser la convolution de Scipy pour le calcul de l'énergie et d'utiliser une seed en particulier pour générer les nombres aléatoires.

        Renvoie:
            list_lattices (list): Liste des grilles de spins à chaque itération ou aux 2000 itérations dépendamment de l'argument save_all.
            energy (float): Énergie finale du système.
            spin_mean_list (list): Liste des moyennes de spins à chaque itération.
            energy_list (list): Liste des énergies à chaque itération.
        """

        h = self.h
        size = self.size
        betaJ = self.betaJ
        lattice = self.lattice.copy()
        energy = self.energy
        spin_mean_list = [np.mean(lattice)]
        list_lattices = [lattice.copy()]
        energy_list = [energy]
        rng = self.rng
        energy_fluctuation = 1e6 # Initialisation de la fluctuation d'énergie pour les premiers pas de temps avant qu'on atteigne un nombre d'itérations suffisant pour calculer la fluctuation d'énergie.
        seed_offsetting = 0

        if fast:
            lattice, energy, spin_mean_list, energy_list, list_lattices, seed_offsetting = metropolis_fast(lattice, h, betaJ, n_iter, self.seed, self.seed_offset, save_all, verbose=self.verbose)
            # Actualisation des attributs de la classe avec les résultats de la simulation
            self.seed_offset += seed_offsetting # On avance dans la séquence du générateur. 
            self.lattice = lattice
            self.energy = energy
            self.spin_mean_list = spin_mean_list
            self.energy_list = energy_list
            return list_lattices, energy, spin_mean_list, energy_list

        else:
            for iter in range(n_iter):
                new_lattice = lattice.copy()
                if iter % 1000 == 0 and self.verbose:
                    print(f"h = {h:.2f}, iter = {iter}, E = {energy:.2f}, ΔE_fluct = {energy_fluctuation:.2e}") # État de la simulation tous les 1000 itérations

                # On flip un spin aléatoire
                row, col = rng.integers(0, size), rng.integers(0, size)
                new_lattice[row, col] *= -1 

                # On calcul l'énergie du spin concerné puisque les autres ne changent pas. Le modulo permet de prendre en compte les conditions frontières périodiques.
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

                if iter % 2000 == 0 and not save_all:
                    list_lattices.append(lattice.copy()) # On sauvegarde la grille de spins tous les 2000 itérations si save_all=False
                else:
                    list_lattices.append(lattice.copy()) # On sauvegarde la grille de spins à chaque itération si save_all=True

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
        """
        Renvoie un résumé des paramètres de la simulation.
        
        Renvoie:
            dict: Dictionnaire contenant les paramètres de la simulation.
        """
        
        return {
            "seed": self.seed,
            "lattice_shape": self.lattice.shape,
            "initial_magnetization": np.mean(self.lattice),
            "magnetic_field": self.h,
            "betaJ": self.betaJ,
            "energy": self.energy,
            "spin_mean": np.mean(self.lattice),
        }


    def plot_lattice(self, lattice=None, title="Grille de spins"):
        """
        Trace la grille de spins.
        
        Paramètres:
            lattice (np.ndarray, optional): Grille de spins à tracer. Si None, utilise la grille actuelle de l'objet Metropolis.
            title (str): Titre du graphique.
        """

        if lattice is None:
            lattice = self.lattice
        plt.figure(figsize=(10,6))
        plt.imshow(lattice, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(title)
        plt.colorbar(label="Valeur du spin")
        plt.xticks([])
        plt.yticks([])
        plt.show()


    def plot_energy(self, title="Energie au cours de la simulation"):
        """
        Trace l'énergie au cours de la simulation.

        Paramètres:
            title (str): Titre du graphique.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.energy_list)
        plt.xlabel("Itération")
        plt.ylabel("E/J")
        plt.title(title)
        plt.show()
    

    def plot_spin_mean(self, title="Moment magnétique moyen"):
        """
        Trace le moment magnétique moyen au cours de la simulation.

        Paramètres:
            title (str): Titre du graphique.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.spin_mean_list)
        plt.xlabel("Itération")
        plt.ylabel(r"$\langle M \rangle $")
        plt.title(title)
        plt.show()


    def plot_hysteresis(self, h_low=-1, h_high=1, resolution=0.05, n_iter=30000, fast=True, save_all=False, buffer=5000, run_max=True, fluct_eq=0.002):
        """
        Trace la courbe d'hystérèse en faisant varier le champ magnétique.
        
        Paramètres:
            h_low (float): Valeur minimale du champ magnétique qu'on veut balayer.
            h_high (float): Valeur maximale du champ magnétique qu'on veut balayer.
            resolution (float): Résolution du balayage du champ magnétique. Plus la valeur est petite, plus le balayage est fin.
            n_iter (int): Nombre d'itérations maximal pour la simulation.
            fast (bool): Si True, utilise la méthode rapide de Metropolis à l'aide de la fonction metropolis_fast assistée de Numba. Sinon, on utilise la méthode classique sans Numba.
            save_all (bool): Si True, sauvegarde la grille de spins à chaque itération. Sinon, sauvegarde la grille de spins tous les 2000 itérations.
            buffer (int): Taille du buffer pour le calcul de la fluctuation d'énergie dans le cas où fast=False et run_max=False.
            run_max (bool): Si True, la simulation s'arrête lorsque la fluctuation d'énergie est suffisamment petite dans le cas où fast=False et run_max=false.
            fluct_eq (float): Fluctuation d'énergie à atteindre pour considérer que le système est en équilibre. Utilisé seulement si run_max=True dans le cas où fast=False. 
        """

        h_list = np.concatenate((np.arange(h_low, h_high, resolution), np.arange(h_high, h_low, -resolution)))
        spin_step_list = []
        for i in tqdm(range(len(h_list)), desc="Variation du champ magnétique"):
            self.h = h_list[i]  # On change le champ magnétique pour la prochaine itération
            lattices, _, spin_means, _ = metro.find_equilibrium(n_iter, fast, save_all, run_max, fluct_eq, buffer)
            spin_step_list.append(spin_means[-1])
            metro.lattice = lattices[-1]  # On change le champ magnétique pour la prochaine itération
        plt.figure(figsize=(10,6))
        plt.plot(h_list, spin_step_list, color="darkBlue", linewidth=2.5, label=r"$\beta J = $" + f"{self.beta:.2f}")
        plt.scatter(h_list, spin_step_list, color="black")
        plt.xlabel(r"Champ magnétique normalisé $h/J$")
        plt.ylabel(r"Magnétisation moyenne $\langle M \rangle $")
        plt.title("Courbe d'hystérèse")
        plt.ylim(-1, 1)
        plt.show()

        return h_list, spin_step_list


#metro = Metropolis(lattice_size=64, betaJ=0.7, magnetic_field=0.0, pourcentage_up=0, verbose=False)
#metro.plot_hysteresis(h_low=-2, h_high=2, resolution=0.1, fast=True)