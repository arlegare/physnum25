# Ising Spin Glass Model Simulation

This project implements a simulation of the Ising spin glass model in three dimensions. The Ising model is a mathematical model used in statistical mechanics to understand phase transitions and magnetic properties of materials.

## Overview

The Ising spin glass model consists of a lattice of spins that can be in one of two states: up (+1) or down (-1). The interactions between neighboring spins and the external temperature influence the overall behavior of the system. This project provides tools to simulate and analyze the properties of the spin glass model.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd ising-spin-glass
pip install -r requirements.txt
```

## Usage

To run the simulation, you can use the `IsingModel` class defined in `src/ising_model.py`. Here is a simple example:

```python
from src.ising_model import IsingModel

model = IsingModel(size=10, temperature=1.0)
model.initialize_lattice()
model.run_simulation(steps=1000)
```

## Running Tests

To ensure that the implementation is correct, you can run the unit tests provided in the `tests` directory. Use the following command:

```bash
pytest tests/test_ising_model.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.