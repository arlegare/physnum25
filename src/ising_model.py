class IsingModel:
    def __init__(self, size, temperature, interaction_strength):
        self.size = size
        self.temperature = temperature
        self.interaction_strength = interaction_strength
        self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        import numpy as np
        return np.random.choice([-1, 1], size=(self.size, self.size, self.size))

    def calculate_energy(self):
        energy = 0
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    s = self.lattice[x, y, z]
                    neighbors = self.get_neighbors(x, y, z)
                    for n in neighbors:
                        energy -= s * self.lattice[n]
        return energy

    def get_neighbors(self, x, y, z):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if (dx, dy, dz) != (0, 0, 0):
                        nx, ny, nz = (x + dx) % self.size, (y + dy) % self.size, (z + dz) % self.size
                        neighbors.append((nx, ny, nz))
        return neighbors

    def update_spin(self, x, y, z):
        import numpy as np
        delta_energy = -2 * self.lattice[x, y, z] * self.calculate_energy()
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / self.temperature):
            self.lattice[x, y, z] *= -1

    def run_simulation(self, steps):
        for step in range(steps):
            x, y, z = np.random.randint(0, self.size, size=3)
            self.update_spin(x, y, z)