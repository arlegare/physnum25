def generate_random_spin():
    import random
    return random.choice([-1, 1])

def calculate_magnetization(spins):
    return sum(spins) / len(spins)

def plot_results(energies, magnetizations):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(energies, label='Energy')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Energy vs. Step')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(magnetizations, label='Magnetization', color='orange')
    plt.xlabel('Step')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs. Step')
    plt.legend()

    plt.tight_layout()
    plt.show()