from Metropolis import *

instance = Metropolis(100000, 100, 0, 2, 1)

lattices = instance.find_equilibrium()

def visualize_snapshots(snapshots):   # J'ai steal ça de copilot pour visualiser rapidement mais marche pas for now à cause que numba est dans une classe
    plt.figure(figsize=(10, 5))
    for idx, lattice in enumerate(snapshots):
        plt.subplot(1, len(snapshots), idx + 1)
        plt.imshow(lattice, cmap='binary', interpolation='none')
        plt.title(f"Step {idx * 10}")  # Adjust interval as needed
    plt.tight_layout()
    plt.show()
