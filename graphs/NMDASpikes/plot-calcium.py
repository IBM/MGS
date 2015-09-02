import matplotlib
#matplotlib.use('pdf')
matplotlib.use('gtk')
import matplotlib.pyplot as plt
import numpy as np

t, v1, v2 = np.loadtxt('out/somaCa.dat2',unpack=True,skiprows=1)
_, v3 = np.loadtxt('out/somaCa.dat4',unpack=True,skiprows=1)

plt.plot(t, v1, 'blue')
plt.plot(t, v2, 'red')
plt.plot(t, v3, 'green')
plt.show() 

