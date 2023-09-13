#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([74, 9.4, 7.8, 2.4, 6.3, 11.1])
mylabels = ["White", "Native American", "Black or African American", "Asian",  "Two or more races", "Hispanic"]
myexplode = [0, 0.3, 0, 0,0, 0]

plt.pie(y, labels = mylabels, explode = myexplode, shadow = True, startangle=90)
plt.title("Racial and Ethnic Composition of Oklahoma")
plt.show() 


# ## Visual Python Upgrade
# NOTE: 
# - Refresh your web browser to start a new version.
# - Save VP Note before refreshing the page.

# In[1]:


# Visual Python
get_ipython().system('pip install visualpython --upgrade')


# In[2]:


# Visual Python
get_ipython().system('visualpy install')


# ## Visual Python Upgrade
# NOTE: 
# - Refresh your web browser to start a new version.
# - Save VP Note before refreshing the page.

# In[1]:


# Visual Python
get_ipython().system('pip install visualpython --upgrade')


# In[2]:


# Visual Python
get_ipython().system('visualpy install')


# In[11]:


x= np.array([75, 23])
labels= ["Roman Catholic", "Protestant"]
explode= [0, 0.3]
plt.pie(x, labels= labels, shadow=True, startangle=90, explode= explode)

plt.title("Demographic of Derry")
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt


def f(x, y): 
    return (9.81-0.003388*y**2)

def f2(x): 
    return 53.81*np.tanh(0.18230806*x)

def rk4_ode(func, step, x0, xn, y0):
    x=np.arange(x0, xn, step)
    y=np.zeros(len(x))
    y[0]=y0
    for i in range(1, len(x)):
        k1= step*func(x[i-1], y[i-1])
        k2= step*func(x[i-1]+step/2, y[i-1]+k1/2)
        k3= step*func(x[i-1]+step/2, y[i-1]+k2/2)
        k4= step*func(x[i-1]+step, y[i-1]+k3)
        y[i]=y[i-1]+(k1+2*k2+2*k3+k4)/6
    return x, y

xode, yode=rk4_ode(f, 0.5, 0, 50, 0)
plt.figure()
plt.title("RK4 with exact solution for step-size 0.025")
plt.plot(xode, yode, 'ro', label='RK4 Method')
plt.plot(xode, f2(xode), label='Exact Solution')
plt.grid()
plt.legend()


# In[5]:


p= input("enter password: ")
password= "yashas"
imax= 5
i=1

while p!= password:
    i=i+1
    p= input("Wrong password. Try again: ")
    
    if i==imax:
        break
        
if i==imax:
    print("Password entered wrong", imax, "times")
    
else:
    print("Good job!")


# In[5]:





# In[8]:


import numpy as np
import matplotlib.pyplot as plt

a= np.array([-10,-9,-8,-7,-6,-5,-4,-3,-1,0,1,2,3,4,5,6,7,8,9,10])

b= a**2
c= a**3
d= np.exp(a)

plt.scatter(a, d)
plt.show()


# In[4]:


import numpy as np

import matplotlib.pyplot as plt


def generate_spin_states(N):
    # Generate all possible spin states using numpy.meshgrid
    spins = np.array([0, 1, -1])
    spin_states = np.meshgrid(*([spins] * N))
    spin_states = np.stack(spin_states, axis=-1).reshape(-1, N)

    return spin_states


N = 10

spin_array = generate_spin_states(N)

T = np.linspace(0.01, 10, 100)

energies = np.sum(spin_array**2, axis=1)
magnet = np.sum(spin_array, axis=1)

def partition_function(t):
  return np.sum(np.exp(-energies/t))

def Magnetization(t):
  M = (partition_function(t) * magnet)/(partition_function(t) * N)
  
  return M


plt.plot(T, [Magnetization(t) for t in T])

plt.xlabel(r"$T$")
plt.ylabel(r"$\frac{M}{N}$")


# In[1]:


get_ipython().system('pip install scikit-learn')


# In[2]:


import numpy as np
from sklearn.neural_network import BernoulliRBM
from itertools import product

# Define the parameters of the Quantum Ising Model
N = 4  # Number of spins
h = np.random.uniform(-1, 1, N)  # Local magnetic fields
J = np.random.uniform(-1, 1, (N, N))  # Coupling strengths

# Generate all possible spin configurations
spin_configurations = np.array(list(product([0, 1], repeat=N)))

# Calculate the energy for each spin configuration
def calculate_energy(config):
    return -0.5 * np.sum(h * (2 * config - 1)) - 0.5 * np.sum(J * np.outer(2 * config - 1, 2 * config - 1))

energies = np.array([calculate_energy(config) for config in spin_configurations])

# Normalize the energies
energies -= np.min(energies)
energies /= np.max(energies)

# Initialize the RBM
rbm = BernoulliRBM(n_components=8, learning_rate=0.01, n_iter=1000)

# Train the RBM
rbm.fit(spin_configurations, energies)

# Generate samples using Gibbs sampling
def gibbs_sample(initial_state, rbm, num_samples):
    samples = np.tile(initial_state, (num_samples, 1))
    for _ in range(num_samples):
        hidden_probs = 1 / (1 + np.exp(-(samples @ rbm.components_.T + rbm.intercept_hidden_)))
        hidden_states = np.random.rand(*hidden_probs.shape) < hidden_probs
        visible_probs = 1 / (1 + np.exp(-(hidden_states @ rbm.components_ + rbm.intercept_visible_)))
        visible_states = np.random.rand(*visible_probs.shape) < visible_probs
        samples = visible_states
    return samples

num_samples = 1000
initial_state = np.random.choice([0, 1], size=N)
generated_samples = gibbs_sample(initial_state, rbm, num_samples=num_samples)

print("Generated Samples:")
print(generated_samples)


# In[1]:


import csv as csv
import numpy as np
import matplotlib.pyplot as plt

def ReadCSV(fileLocation):
    '''
    Function to read the CSV data files, input is the name of the CSV file, output nparray of vectors.
    
    (string) -> nparray of dimension # objects in dataset * length of each vector in dataset.
    '''
    with open(fileLocation,'r') as csvFile:
        array = []
        reader = csv.reader(csvFile)

        for row in reader:
            array.append(row)
            
        return np.array(array).astype(np.double)
    
def PlotLattice(array, size, save = False, fileName = ""):
    '''
    Function to plot the size x size vector of a dataset as a digit in a matplotlib heatplot.
    
    (list of length size x size, size) -> void.
    '''
    plt.imshow(array.reshape((size,size)), cmap='Greys', interpolation='nearest')
   
    plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM

import InputOutput as IO

'''
Import the generated lattices and labels and convert the lattices from {-1,-1} to {0,1} because the RBM expects binary values.
'''
latticesTotal = IO.ReadCSV('Lattices16.csv')
temperatureLabels = IO.ReadCSV('TemperatureLabels16.csv')
latticesTotal = (latticesTotal + 1) / 2

'''
Import the observables as generated using MCMC to plot to compare to the RBM observables.
'''
cMCMC = IO.ReadCSV('SpecificHeatPerSpin16.csv')
mMCMC = IO.ReadCSV('MagnetisationPerSpin16.csv')
eMCMC = IO.ReadCSV('EnergyPerSpin16.csv')
temperatureLabelsMCMC = IO.ReadCSV('TemperatureLabelsMCMC16.csv')

cMCMC = [x[0] for x in cMCMC]
mMCMC = [x[0] for x in mMCMC]
eMCMC = [x[0] for x in eMCMC]

plt.plot(temperatureLabelsMCMC, eMCMC, 'bo', markersize = 1)
plt.xlabel(r'$T$ [dimensionless units]',fontsize=18)
plt.ylabel(r'$\langle E \rangle$ [dimensionless units]',fontsize=18)

plt.show()

plt.plot(temperatureLabelsMCMC, np.abs(mMCMC), 'bo', markersize = 1)
plt.xlabel(r'$T$ [dimensionless units]',fontsize=18)
plt.ylabel(r'$\langle E \rangle$ [dimensionless units]',fontsize=18)

plt.show()

plt.plot(temperatureLabelsMCMC, cMCMC, 'bo', markersize = 1)
plt.xlabel(r'$T$ [dimensionless units]',fontsize=18)
plt.ylabel(r'$c$ [dimensionless units]',fontsize=18)

plt.show()

latticeSize = int(len(latticesTotal[0])**(1/2))

'''
Function to calculate the total energy of a lattice at a given size.
'''
def LatticeEnergy(lattice, size):
    Energy = 0
    
    lattice = (2*lattice) - 1
    lattice = np.reshape(lattice, (size,size))
       
    for i in range(size):
        for j in range(size):
            Energy = Energy + -1*lattice[i,j] *( lattice[(i + 1)%size,j] + lattice[(i - 1)%size,j] + lattice[i,(j+1)%size] + lattice[i ,(j-1)%size])

    return Energy

'''
Function to calculate the total magnetisation of a lattice at a given size.
'''
def LatticeMagnetisation(lattice,size):    
    lattice = (2*lattice) - 1
    
    return np.sum(lattice)

print(latticesTotal.shape)

'''
Plot the energy and the magnetisation of each lattice in the dataset to verify whether it is correct.
'''
maglist = []
elist = []

tempList = np.unique(temperatureLabels, return_counts = True)[0]
nItr = np.unique(temperatureLabels, return_counts = True)[1][0]

for lattice in latticesTotal:
    
    
    maglist.append(np.abs(LatticeMagnetisation(lattice, latticeSize))/(latticeSize**2))
    elist.append(LatticeEnergy(lattice, latticeSize)/(latticeSize**2))
    
plt.plot(maglist, 'bo', markersize = 1)
plt.xlabel('Data points')
plt.ylabel('Magnetization')

plt.show()

plt.plot(elist, 'bo', markersize = 1)
plt.xlabel('Data points')
plt.ylabel('Energy')

plt.show()

'''
Reshape the lattice and label arrays such that each temperature dataset is in a separate vector.
'''
latticesTotal = np.reshape(latticesTotal, (len(tempList), nItr, latticeSize**2))
temperatureLabels = np.reshape(temperatureLabels, (len(tempList),nItr))

magnetisationVariance = []
energyVariance = []

magnetisation = []
energy = []

'''
Define a RBM, train it and use it to generate new lattices. Use these lattices to calculate the observables.
This process is done at each temperature in the range and at each number of hidden nodes that we want to calculate.
'''
numberHiddenNodes = [8,16,256, 512]

for h in range(len(numberHiddenNodes)):
    magnetisationOneTemperature = []
    energyOneTemperature = []
    
    magnetisationVarianceOneTemperature = []
    energyVarianceOneTemperature = []
    
    temperatureCounter = 0
    
    for latticeOneTemperature in latticesTotal:
        temperature = temperatureLabels[temperatureCounter,0]
        print("Temperature: " + str(temperature))
      
        rbm = BernoulliRBM(n_components = numberHiddenNodes[h], learning_rate=0.001 , verbose=False, n_iter = 100, batch_size = 10)                
        rbm.fit(latticeOneTemperature)
           
        '''
        Plot and save the weightmatrix for each temperature.
        '''
        plt.imshow(rbm.components_)
        plt.ylabel("Hidden nodes")
        plt.xlabel("Visible nodes")
        plt.savefig("weightsTemp" + str(temperature) + "Nh" + str(numberHiddenNodes[h]) + ".pdf")
        plt.show()
        
        '''
        Generate a random lattice and use Gibbs sampling to generate new lattices at this temperature. After equilibration sample the system and use those samples to calculate the observables.
        '''
        magnetisationSampleList = []
        energySampleList = []
        
        randomlattice =  np.random.randint(2, size=latticeSize**2)
    
        equilibrationTime = 1000
        numberSamples = 1000
               
        for i in range(equilibrationTime * latticeSize**2 + numberSamples * latticeSize**2):
            lattice = rbm.gibbs(randomlattice)
            
            if i > equilibrationTime * latticeSize**2 and i % latticeSize**2 == 0:           
                magnetisationSampleList.append(LatticeMagnetisation(lattice, latticeSize)/latticeSize**2)
                energySampleList.append(LatticeEnergy(lattice, latticeSize))
                        
        magnetisationOneTemperature.append(np.average(magnetisationSampleList))
        energyOneTemperature.append(np.average(energySampleList)/latticeSize**2)
        
        magnetisationVarianceOneTemperature.append((latticeSize**2/temperature) * np.var(magnetisationSampleList))
        energyVarianceOneTemperature.append((1/(temperature**2 * latticeSize**2)) * np.var(energySampleList))
            
        temperatureCounter = temperatureCounter + 1
        
    magnetisation.append(magnetisationOneTemperature)
    energy.append(energyOneTemperature)
    
    magnetisationVariance.append(magnetisationVarianceOneTemperature)
    energyVariance.append(energyVarianceOneTemperature)

colours = ['ro','go','bo', 'mo']

for i in range(len(numberHiddenNodes)):
    plt.plot(tempList, energyVariance[i], colours[i])
plt.ylabel("c [dimensionless units]")
plt.xlabel(r'$T$ [dimensionless units]')
plt.legend([r'$N_h = 8$', r'$N_h = 16$', r'$N_h = 256$', r'$N_h = 512$'], loc='upper right')

plt.show()

for i in range(len(numberHiddenNodes)):
    plt.plot(tempList, magnetisation[i], colours[i])
plt.ylabel(r'$\langle M \rangle$ [dimensionless units]')
plt.xlabel(r'$T$ [dimensionless units]')
plt.legend([r'$N_h = 8$', r'$N_h = 16$', r'$N_h = 256$', r'$N_h = 512$'], loc='upper right')

plt.show()

for i in range(len(numberHiddenNodes)):
    plt.plot(tempList, energy[i], colours[i])
plt.ylabel(r'$\langle E \rangle$ [dimensionless units]')
plt.xlabel(r'$T$ [dimensionless units]')
plt.legend([r'$N_h = 8$', r'$N_h = 16$', r'$N_h = 256$', r'$N_h = 512$'], loc='lower right')

plt.show()


# In[3]:


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


# In[ ]:


y = int(input("What's the square? "))
left = 0
right = y

print(left)

g = (left + right)/2

while abs(y-(g*g)) > 0.0001:
    if (g*g) < y:
        left = g
    else:
        right = g
    

print("The square root of", y , "is", g)


# In[ ]:





# In[ ]:




