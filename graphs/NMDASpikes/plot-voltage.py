# import matplotlib
#matplotlib.use('pdf')
# matplotlib.use('gtk')
import matplotlib.pyplot as plt
import numpy as np

t, v1, v2 = np.loadtxt('out/somaV.dat2',unpack=True,skiprows=1)
t, v3, _ = np.loadtxt('out/synapseV0.dat2',unpack=True,skiprows=1)
_, v4 = np.loadtxt('out/somaV.dat4',unpack=True,skiprows=1)

plt.plot(t, v1, 'blue')
plt.plot(t, v2, 'red')
plt.plot(t, v3, 'green')
plt.plot(t, v4, 'orange')
plt.show() 

