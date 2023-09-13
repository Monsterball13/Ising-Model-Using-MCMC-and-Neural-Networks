
import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(size):
    return np.random.choice([-1, 1], size=(size, size))

def calculate_energy(lattice):
    energy = 0
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            neighbor_sum = lattice[(i+1)%len(lattice), j] + lattice[i, (j+1)%len(lattice)] +                             lattice[(i-1)%len(lattice), j] + lattice[i, (j-1)%len(lattice)]
            energy += -lattice[i, j] * neighbor_sum
    return energy

def metropolis(lattice, temperature):
    for _ in range(len(lattice)**2):
        i, j = np.random.randint(0, len(lattice), size=2)
        old_spin = lattice[i, j]
        new_spin = -old_spin
        delta_energy = 2 * old_spin * (lattice[(i+1)%len(lattice), j] + lattice[i, (j+1)%len(lattice)] +
                                       lattice[(i-1)%len(lattice), j] + lattice[i, (j-1)%len(lattice)])
        if delta_energy <= 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            lattice[i, j] = new_spin

def calculate_magnetization(lattice):
    return np.sum(lattice)

def main():
    lattice_size = 16
    num_spins = lattice_size**2
    num_points = 1000
    temperatures = np.linspace(1.0, 4.0, num_points)
    magnetizations = []
    energies = []

    for temp in temperatures:
        lattice = initialize_lattice(lattice_size)
        for _ in range(1000):
            metropolis(lattice, temp)
        
        magnetization_sum = 0
        energy_sum = 0
        for _ in range(num_spins):
            metropolis(lattice, temp)
            magnetization = calculate_magnetization(lattice)
            energy = calculate_energy(lattice)
            magnetization_sum += magnetization
            energy_sum += energy
        
        magnetizations.append(abs(magnetization_sum) / num_spins)
        energies.append(energy_sum / num_spins)

    specific_heat = np.diff(energies) / np.diff(temperatures)
    
    plt.figure(figsize=(20, 15))

    plt.subplot(3, 1, 1)
    plt.scatter(temperatures, magnetizations, marker='o', color='b')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs Temperature')

    plt.subplot(3, 1, 2)
    plt.scatter(temperatures, energies, marker='o', color='r')
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.title('Energy vs Temperature')

    plt.subplot(3, 1, 3)
    plt.scatter(temperatures[:-1], specific_heat, marker='o', color='g')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.title('Specific Heat vs Temperature')

    plt.show()

if __name__ == "__main__":
    main()

