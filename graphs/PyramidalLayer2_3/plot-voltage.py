# import matplotlib
#matplotlib.use('pdf')
# matplotlib.use('gtk')
import matplotlib.pyplot as plt
import numpy as np

if (case == 0):
    folder = "out"
    t, v1 = np.loadtxt(folder + '/V_soma.dat0',unpack=True,skiprows=1)
    #_, v2 = np.loadtxt('out/V_bouton1.dat3',unpack=True,skiprows=1)
    #_, v3 = np.loadtxt('out/V_bouton2.dat3',unpack=True,skiprows=1)
    #_, v4 = np.loadtxt('out/V_bouton3.dat2',unpack=True,skiprows=1)
    #_, v5 = np.loadtxt('out/V_spine1.dat2',unpack=True,skiprows=1)
    #_, v6 = np.loadtxt('out/V_spine2.dat2',unpack=True,skiprows=1)
    #_, v7 = np.loadtxt('out/V_spine3.dat2',unpack=True,skiprows=1)
    #_, v8, _ = np.loadtxt('out/V_synapse1.dat2', unpack=True,skiprows=1)
    #_, v9 = np.loadtxt('out/V_synapse2.dat2', unpack=True,skiprows=1)
    #_, v10 = np.loadtxt('out/V_synapse3.dat2', unpack=True,skiprows=1)

    plt.plot(t, v1, 'red')
    #plt.plot(t, v2, 'red')
    #plt.plot(t, v3, 'green')
    #plt.plot(t, v4, 'blue')
    #plt.plot(t, v5, 'r--')
    #plt.plot(t, v6, 'g--')
    #plt.plot(t, v7, 'b--')
    #plt.plot(t, v8, 'pink')
    #plt.plot(t, v9, 'aqua')
    #plt.plot(t, v10, 'purple')

    plt.show()

elif (case == 1):
    folder = "out2"
    t, v1 = np.loadtxt(folder + '/V_soma.dat0',unpack=True,skiprows=1)
    #_, v2 = np.loadtxt('out/V_bouton1.dat3',unpack=True,skiprows=1)
    #_, v3 = np.loadtxt('out/V_bouton2.dat3',unpack=True,skiprows=1)
    #_, v4 = np.loadtxt('out/V_bouton3.dat2',unpack=True,skiprows=1)
    #_, v5 = np.loadtxt('out/V_spine1.dat2',unpack=True,skiprows=1)
    #_, v6 = np.loadtxt('out/V_spine2.dat2',unpack=True,skiprows=1)
    #_, v7 = np.loadtxt('out/V_spine3.dat2',unpack=True,skiprows=1)
    #_, v8, _ = np.loadtxt('out/V_synapse1.dat2', unpack=True,skiprows=1)
    #_, v9 = np.loadtxt('out/V_synapse2.dat2', unpack=True,skiprows=1)
    #_, v10 = np.loadtxt('out/V_synapse3.dat2', unpack=True,skiprows=1)

    plt.plot(t, v1, 'red')
    #plt.plot(t, v2, 'red')
    #plt.plot(t, v3, 'green')
    #plt.plot(t, v4, 'blue')
    #plt.plot(t, v5, 'r--')
    #plt.plot(t, v6, 'g--')
    #plt.plot(t, v7, 'b--')
    #plt.plot(t, v8, 'pink')
    #plt.plot(t, v9, 'aqua')
    #plt.plot(t, v10, 'purple')

    plt.show()


