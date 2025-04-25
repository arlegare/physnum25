import numpy as np
import h5py
from simul_ising import Metropolis  #  On importe la classe.
from tqdm import tqdm # Pour afficher où on est rendu dans la série de simulations.
import matplotlib.pyplot as plt

def save_simulation_to_hdf5(filename, run_id, sim, lattices, spin_means, energy_list):
    with h5py.File(filename, 'a') as f:
        group_name = f"run_{run_id:04d}"
        if group_name in f:
            print(f"Warning: Group {group_name} already exists. Skipping save.")
            return  # On saute si le groupe d'enregistrement existe déjà. 

        grp = f.create_group(group_name)
        grp.create_dataset("initial_lattice", data=sim.lattice)
        grp.create_dataset("final_lattice", data=lattices[-1])
        grp.create_dataset("energy", data=energy_list)
        grp.create_dataset("magnetization", data=spin_means)

        grp.attrs["seed"] = sim.seed
        grp.attrs["betaJ"] = sim.betaJ
        grp.attrs["magnetic_field"] = sim.h
        grp.attrs["pourcentage_up"] = sim.up_perc
        grp.attrs["n_iter"] = sim.n_iter
        grp.attrs["convol"] = sim.convol

def run_batch_simulations(h_range, n_seeds, filename="data/results.h5"):
    betaJ = 1.0 # Couplage adimensionné (merci Sam).
    size = 64 # Un beau multiple de 2, comme on aime.
    pourcentage_up = 0.5 # On peut donner un avantage aux spins up/down, histoire d'aller chercher préférentiellement un point fixe stable!

    final_magnetizations = []
    std_magnetizations = []

    run_id = 0

    for h in tqdm(h_range, desc="Variation du champ magnétique"):
        mags = []
        for i in range(n_seeds):
            seed = np.random.randint(0, int(1e6))
            sim = Metropolis(
                n_iter=0,
                lattice_size=size,
                magnetic_field=h,
                betaJ=betaJ,
                pourcentage_up=pourcentage_up,
                convol="scipy",
                seed=seed
            )
            lattices, spin_means, energy_list = sim.run()
            save_simulation_to_hdf5(filename, run_id, sim, lattices, spin_means, energy_list)
            mags.append(spin_means[-1])
            run_id += 1

        final_magnetizations.append(np.mean(mags))
        std_magnetizations.append(np.std(mags))

    return h_range, final_magnetizations, std_magnetizations

if __name__ == "__main__": # Fancy pythonic attitude, même si mon coeur est au langage C.
    h_range = np.linspace(-0.2, 0.2, 2)
    print(h_range)
    h_vals, mag_means, mag_stds = run_batch_simulations(h_range, n_seeds=20)

    plt.figure()
    plt.plot(h_vals, mag_means, label="Magnétisation moyenne")
    plt.fill_between(h_vals,
                     np.array(mag_means) - np.array(mag_stds),
                     np.array(mag_means) + np.array(mag_stds),
                     alpha=0.3, label="Écart-type")
    
    plt.xlabel("Champ magnétique h")
    plt.ylabel("Magnétisation finale moyenne")
    plt.title("Effet du champ magnétique sur la magnétisation")
    plt.legend()
    plt.grid(True)
    plt.show()
