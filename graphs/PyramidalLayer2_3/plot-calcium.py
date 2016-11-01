import matplotlib
#matplotlib.use('pdf')
matplotlib.use('gtk')
import matplotlib.pyplot as plt
import numpy as np

case = 1
if (case == 0):
    folder='out'
    t, v1, v2, v3, v4 = np.loadtxt(folder+'/somaCa.dat2',unpack=True,skiprows=1)
    _, v5 = np.loadtxt(folder+'/somaCa.dat4',unpack=True,skiprows=1)

    plt.plot(t, v1, 'blue')
    plt.plot(t, v2, 'red')
    plt.plot(t, v3, 'green')
    plt.plot(t, v4, 'orange')
    plt.plot(t, v5, 'yellow')
    plt.show()
elif (case == 1):
    folder='out2'
    t, v1, v2, v3 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1)

    plt.plot(t, v1, 'blue', label='neuron')
    plt.plot(t, v2, 'red', label='bouton')
    plt.plot(t, v3, 'green', label='spinehead')
    plt.legend()
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()


