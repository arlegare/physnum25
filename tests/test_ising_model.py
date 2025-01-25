import unittest
from src.ising_model import IsingModel

class TestIsingModel(unittest.TestCase):

    def setUp(self):
        self.model = IsingModel(size=10, temperature=1.0, interaction_strength=1.0)

    def test_initialize_lattice(self):
        self.model.initialize_lattice()
        self.assertEqual(self.model.lattice.shape, (10, 10, 10))
        self.assertTrue((self.model.lattice == 1).any() or (self.model.lattice == -1).any())

    def test_calculate_energy(self):
        self.model.initialize_lattice()
        energy = self.model.calculate_energy()
        self.assertIsInstance(energy, float)

    def test_update_spin(self):
        self.model.initialize_lattice()
        initial_spin = self.model.lattice[0, 0, 0]
        self.model.update_spin(0, 0, 0)
        self.assertNotEqual(initial_spin, self.model.lattice[0, 0, 0])

    def test_run_simulation(self):
        self.model.run_simulation(steps=100)
        self.assertIsInstance(self.model.magnetization, float)

if __name__ == '__main__':
    unittest.main()