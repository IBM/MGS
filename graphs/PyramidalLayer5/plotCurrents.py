__author__ = "Hoang Trong Minh Tuan"
__copyright__ = "Copyright 2016, IBM"
__credits__ = []  # list of people who file pugs
__version__ = "0.1"
__maintainer__ = "Hoang Trong Minh Tuan"
__email__ = "tmhoangt@us.ibm.com"
__status__ = "Prototype"  # (Prototype, Development, Production)
import matplotlib
import matplotlib as mpl
#matplotlib.use('pdf')
matplotlib.use('gtk')
import re

import matplotlib.pyplot as plt
import numpy as np
import os.path
import csv
import sys
import argparse
import fnmatch
from mpldatacursor import datacursor

def getFile(folder,fileNamePrefix):
    for file in os.listdir(folder):
        if fnmatch.fnmatch(file, fileNamePrefix+'*'):
            return file
    return ""

def plot_case0():
    # only neuron (no spine/bouton)
    #    folder='out'
    #    t, v1, v2, v3, v4 = np.loadtxt(folder+'/somaCa.dat2',unpack=True,skiprows=1)
    #    _, v5 = np.loadtxt(folder+'/somaCa.dat4',unpack=True,skiprows=1)
    #
    #    plt.plot(t, v1, 'blue')
    #    plt.plot(t, v2, 'red')
    #    plt.plot(t, v3, 'green')
    #    plt.plot(t, v4, 'orange')
    #    plt.plot(t, v5, 'yellow')
    #    plt.show()
    folder='out2'
    f, axarr = plt.subplots(2, sharex=True)
    t, v0 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                            usecols=(0,1))

    axarr[0].plot(t, v0, 'blue', label='soma')
    axarr[0].legend()
    t, v0= np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                            usecols=(0,1))
    axarr[1].plot(t, v0, 'red', label='soma')
    plt.show()

def plot_case1():
    folder='out2'
    t, v1, v2, v3 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1)

    plt.plot(t, v1, 'blue', label='neuron')
    plt.plot(t, v2, 'red', label='bouton')
    plt.plot(t, v3, 'green', label='spinehead')
    plt.legend()
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def plot_case2():
    numProcess = 1
    folder='out2'
    t, v1, v2, v3 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                            usecols=(0,1,2,3))

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(t, v1, 'blue', label='neuron')
    axarr[0].plot(t, v2, 'red', label='bouton')
    axarr[0].plot(t, v3, 'green', label='spinehead')
    axarr[0].legend()
    t, v1, v2, v3 = np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                            usecols=(0,2,1,3))
    axarr[1].plot(t, v1, 'blue', label='neuron')
    axarr[1].plot(t, v2, 'red', label='bouton')
    axarr[1].plot(t, v3, 'green', label='spinehead')
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def plot_case3():
    # no neuron, only bouton + spinehead
    numProcess = 1
    folder='out2'
    f, axarr = plt.subplots(3, sharex=True)
    if (0):# just 2 neurons
        t, v1, v2 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2))

        axarr[0].plot(t, v1, 'red', label='bouton')
        axarr[0].plot(t, v2, 'green', label='spinehead')
        axarr[0].legend()
        t, v1, v2 = np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2))
        axarr[1].plot(t, v1, 'red', label='bouton')
        axarr[1].plot(t, v2, 'green', label='spinehead')
    else: # 4 neurons (2 pair: bouton+spine)
        t, v1, v2, v3,v4 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2,3,4))

        # f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(t, v1, 'red', label='bouton')
        axarr[0].plot(t, v2, 'green', label='spinehead')
        axarr[0].legend()
        axarr[0].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[0].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
        #axarr[0].set_xlim(left=0);
        #axarr[0].set_xlim(right=60);
        t, v1, v2, v3,v4 = np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2,3,4))
        axarr[1].plot(t, v1, 'red', label='bouton')
        axarr[1].plot(t, v2, 'green', label='spinehead')
        axarr[1].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[1].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
        axarr[2].plot(t, v3, 'red', label='bouton')
        axarr[2].plot(t, v4, 'green', label='spinehead')
        f.gca().set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
        f.gca().set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def plot_case4():
    # no neuron, only bouton + spinehead
    # with NMDAR current recording
    numProcess = 1
    #folder='out2'
    folder='out2_May27'
    if (1):# just 2 neurons
        f, axarr = plt.subplots(3, sharex=True)
        t, v1, v2 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2))

        axarr[0].plot(t, v1, 'red', label='bouton')
        axarr[0].plot(t, v2, 'green', label='spinehead')
        axarr[0].legend()
        t, v1, v2 = np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2))
        axarr[1].plot(t, v1, 'red', label='bouton')
        axarr[1].plot(t, v2, 'green', label='spinehead')
    else: # 4 neurons (2 pair: bouton+spine)
        f, axarr = plt.subplots(3, 2)
        t, v1, v2, v3,v4 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2,3,4))

        # f, axarr = plt.subplots(3, sharex=True)
        axarr[0, 0].plot(t, v1, 'red', label='bouton')
        axarr[0, 0].plot(t, v2, 'green', label='spinehead')
        axarr[0, 0].legend()
        axarr[0, 0].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[0, 0].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
        axarr[0, 0].set_title("[Ca2+] (uM)");
        #axarr[0].set_xlim(left=0);
        #axarr[0].set_xlim(right=60);
        t, v1, v2, v3,v4 = np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2,3,4))
        axarr[1, 0].plot(t, v1, 'red', label='bouton')
        axarr[1, 0].plot(t, v2, 'green', label='spinehead')
        axarr[1, 0].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[1, 0].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
        axarr[1, 0].set_title("Vm (mV)");
        axarr[2, 0].plot(t, v3, 'red', label='bouton')
        axarr[2, 0].plot(t, v4, 'green', label='spinehead')
        axarr[2, 0].set_title("Vm (mV)");
#     axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
#     axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);
        if (os.path.isfile(folder+'/spineNMDAR.dat0')):
            t, v1, v2 = np.loadtxt(folder + '/spineNMDAR.dat0',unpack=True,skiprows=1,
                                    usecols=(0,1,2))
            axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
            axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
            axarr[0, 1].set_title("I_NMDAR (pA/uM^2)");

        if (os.path.isfile(folder+'/spineAMPAR.dat0')):
            t, v1, v2 = np.loadtxt(folder + '/spineAMPAR.dat0',unpack=True,skiprows=1,
                                    usecols=(0,1,2))
            axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
            axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
            axarr[2, 1].set_title("I_AMPAR (pA/uM^2)");
            axarr[2, 1].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
            axarr[2, 1].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def plot_case5():
    # 1 neuron, only bouton + spinehead
    # with NMDAR current recording
    numProcess = 1
    folder='out2'
    if (0):# just 2 neurons
        f, axarr = plt.subplots(3, sharex=True)
        t, v0, v1, v2 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2,3))

        axarr[0].plot(t, v0, 'blue', label='soma')
        axarr[0].plot(t, v1, 'red', label='bouton')
        axarr[0].plot(t, v2, 'green', label='spinehead')
        axarr[0].legend()
        t, v0, v1, v2 = np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                                usecols=(0,1,2, 3))
        axarr[1].plot(t, v1, 'red', label='bouton')
        axarr[1].plot(t, v1, 'red', label='bouton')
        axarr[1].plot(t, v2, 'green', label='spinehead')
    else: # 4 neurons (2 pair: bouton+spine)
        f, axarr = plt.subplots(3, 2)
        t, v0, v1, v2, v3, v4 = np.loadtxt(folder + '/somaCa.dat0',
                                        unpack=True, skiprows=1,
                                        usecols=(0, 1, 2, 3, 4, 5))

        # f, axarr = plt.subplots(3, sharex=True)
        axarr[0, 0].plot(t, v0, 'blue', label='soma')
        axarr[0, 0].plot(t, v1, 'red', label='bouton')
        axarr[0, 0].plot(t, v2, 'green', label='spinehead')
        if (os.path.isfile(folder+'/dendriticCa.dat0')):
            t, v1 = np.loadtxt(folder + '/dendriticCa.dat0',
                            unpack=True, skiprows=1,
                            usecols=(0, 1))
            axarr[0, 0].plot(t, v1, 'black', label='distal-den')
        axarr[0, 0].legend()
        axarr[0, 0].set_ylim(bottom=min(np.amin(v1), np.amin(v2)) - 0.5)
        axarr[0, 0].set_ylim(top=max(np.amax(v1), np.amax(v2)) + 0.5)
        axarr[0, 0].set_title("[Ca2+] (uM)")
        # axarr[0].set_xlim(left=0);
        # axarr[0].set_xlim(right=60);
        t, v0, v1, v2, v3, v4 = np.loadtxt(folder + '/somaV.dat0',
                                        unpack=True, skiprows=1,
                                        usecols=(0, 1, 2, 3, 4, 5))
        axarr[1, 0].plot(t, v0, 'blue', label='soma')
        axarr[1, 0].plot(t, v1, 'red', label='bouton')
        axarr[1, 0].plot(t, v2, 'green', label='spinehead')
        axarr[1, 0].set_ylim(bottom=min(np.amin(v1), np.amin(v2)) - 0.5)
        axarr[1, 0].set_ylim(top=max(np.amax(v1), np.amax(v2)) + 0.5)
        axarr[1, 0].set_title("Vm (mV)")
        axarr[2, 0].plot(t, v3, 'red', label='bouton')
        axarr[2, 0].plot(t, v4, 'green', label='spinehead')
        axarr[2, 0].set_title("Vm (mV)")
        # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
        # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);
        if (os.path.isfile(folder + '/dendriticV.dat0')):
            t, v1 = np.loadtxt(folder + '/dendriticV.dat0',
                            unpack=True, skiprows=1,
                            usecols=(0, 1))
            axarr[2, 0].plot(t, v1, 'black', label='distal-den')
        axarr[2, 0].legend()
        spinehead_area = 0.20 # um^2
        if (os.path.isfile(folder+'/spineNMDAR.dat0')):
            t, v1, v2 = np.loadtxt(folder + '/spineNMDAR.dat0',unpack=True,skiprows=1,
                                    usecols=(0,1,2))
            #axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
            #axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
            #axarr[0, 1].set_title("I_NMDAR (pA/uM^2)");
            v1 = v1 * spinehead_area
            v2 = v2 * spinehead_area
            axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
            axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
            axarr[0, 1].set_title("I_NMDAR (pA)");

        if (os.path.isfile(folder+'/spineAMPAR.dat0')):
            t, v1, v2 = np.loadtxt(folder + '/spineAMPAR.dat0',unpack=True,skiprows=1,
                                    usecols=(0,1,2))
            #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
            #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
            #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
            v1 = v1 * spinehead_area
            v2 = v2 * spinehead_area
            axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
            axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
            axarr[2, 1].set_title("I_AMPAR (pA)");
            axarr[2, 1].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
            axarr[2, 1].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def plot_case6():
    #more comprehensive deal with multiple processes I/O
    folder='./out2'
    MPIprocess=6
    f, axarr = plt.subplots(3, 2)
    t, v0 = np.loadtxt(folder + '/somaCa.dat'+str(MPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, 'blue', label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')
    MPIprocess=15
    if (os.path.isfile(folder+'/dendriticCa.dat'+str(MPIprocess))):
        t, v1 = np.loadtxt(folder + '/dendriticCa.dat'+str(MPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, 'black', label='distal-den')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)
    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    MPIprocess=6
    t, v0 = np.loadtxt(folder + '/somaV.dat'+str(MPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    axarr[1, 0].plot(t, v0, 'blue', label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0))
    maxVm = (np.amax(v0))
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);
    MPIprocess=15
    if (os.path.isfile(folder + '/dendriticV.dat'+str(MPIprocess))):
        t, v1 = np.loadtxt(folder + '/dendriticV.dat'+str(MPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, 'black', label='distal-den')
    axarr[2, 0].legend()
    spinehead_area = 0.20 # um^2


    MPIprocess=15
    if (os.path.isfile(folder+'/spineNMDAR.dat'+str(MPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineNMDAR.dat'+str(MPIprocess),
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
        #axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        #axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
        #axarr[0, 1].set_title("I_NMDAR (pA/uM^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
        axarr[0, 1].set_title("I_NMDAR (pA)");

    MPIprocess=15
    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(MPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineAMPAR.dat'+str(MPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1,2))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        axarr[2, 1].set_title("I_AMPAR (pA)");
        axarr[2, 1].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[2, 1].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def plot_case7():
    #more comprehensive deal with multiple processes I/O
    # with spines
    folder='./out2_spines'
    MPIprocess=4
    f, axarr = plt.subplots(3, 2)
    t, v0 = np.loadtxt(folder + '/somaCa.dat'+str(MPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, 'blue', label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')
    MPIprocess=15
    if (os.path.isfile(folder+'/dendriticCa.dat'+str(MPIprocess))):
        t, v1 = np.loadtxt(folder + '/dendriticCa.dat'+str(MPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, 'black', label='distal-den')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)
    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    MPIprocess=4
    t, v0 = np.loadtxt(folder + '/somaV.dat'+str(MPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    axarr[1, 0].plot(t, v0, 'blue', label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0))
    maxVm = (np.amax(v0))
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);
    MPIprocess=15
    if (os.path.isfile(folder + '/dendriticV.dat'+str(MPIprocess))):
        t, v1 = np.loadtxt(folder + '/dendriticV.dat'+str(MPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, 'black', label='distal-den')
    axarr[2, 0].legend()
    spinehead_area = 0.20 # um^2


    MPIprocess=15
    if (os.path.isfile(folder+'/spineNMDAR.dat'+str(MPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineNMDAR.dat'+str(MPIprocess),
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
        #axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        #axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
        #axarr[0, 1].set_title("I_NMDAR (pA/uM^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
        axarr[0, 1].set_title("I_NMDAR (pA)");

    MPIprocess=15
    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(MPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineAMPAR.dat'+str(MPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1,2))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        axarr[2, 1].set_title("I_AMPAR (pA)");
        axarr[2, 1].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[2, 1].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def plot_case8():
    #more comprehensive deal with multiple processes I/O
    # with spines
    #folder='./out2_May16'
    #somaMPIprocess=10
    #middledenMPIprocess=16
    #distaldenMPIprocess=23
    #thinSpineMPIprocess=15
    #folder='./out2_May16spines'
    #folder='./out2'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=0
    folder='./out2_May16triggersoma'
    somaMPIprocess=0
    middledenMPIprocess=1
    distaldenMPIprocess=1
    thinSpineMPIprocess=0
    somaColor = 'blue'
    middledenColor= 'red'
    distaldenColor= 'green'
    f, axarr = plt.subplots(3, 2)
    t, v0 = np.loadtxt(folder + '/somaCa.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, somaColor, label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')
    if (os.path.isfile(folder+'/distaldendriticCa.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticCa.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/middledendriticCa.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticCa.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, middledenColor, label='middle-den')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)

    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    t, v0 = np.loadtxt(folder + '/somaV.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    axarr[1, 0].plot(t, v0, somaColor, label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0))
    maxVm = (np.amax(v0))
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);
    if (os.path.isfile(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[2, 0].legend()
    if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, middledenColor, label='middle-den')
    axarr[2, 0].legend()

    spinehead_area = 0.20 # um^2
    if (os.path.isfile(folder+'/spineNMDAR.dat'+str(thinSpineMPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineNMDAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
        #axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        #axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
        #axarr[0, 1].set_title("I_NMDAR (pA/uM^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
        axarr[0, 1].set_title("I_NMDAR (pA)");

    thinSpineMPIprocess=15
    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1,2))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        axarr[2, 1].set_title("I_AMPAR (pA)");
        axarr[2, 1].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[2, 1].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()


def parse():
  global parser
  parser = argparse.ArgumentParser(prog = 'plotVmCa')
  parser.add_argument('--version', action='version', version='1.0.1')
  subparsers = parser.add_subparsers(dest="subparsers_name")
  parser_folder = subparsers.add_parser('folder')
  parser_folder.add_argument("folderName", help="full name of the folder with data to be plotted")
  parser_folder.set_defaults(which='folder')

  parser_protocol = subparsers.add_parser('protocol')
  parser_protocol.add_argument("number", default=-1, help="index of simulation protocol (0 = rest; 1 = inject soma; 2= inject shaft with dual-exp EPSP-like current; 3 = inject a particular presynaptic neuron; 4 = like 3, but at a distal region; 5 = trigger soma then at distal end (within a small window of time); 6 = trigger soma, then another spine (within a window of time))")
  parser_protocol.add_argument("date", default="May29", help="date of data, e.g. May30")
  #parser_protocol.set_defaults(func=plot_case9a)
  parser_protocol.add_argument("morphology", default="", help="(optional) name of morph, e.g. hay1")
  parser_protocol.set_defaults(which='protocol')

  #global args
  args = parser.parse_args()
  return args

def plot_case9a(args):
    #more comprehensive deal with multiple processes I/O
    # with spines
    #folder='./out2_May16'
    #somaMPIprocess=10
    #middledenMPIprocess=16
    #distaldenMPIprocess=23
    #thinSpineMPIprocess=15
    #folder='./out2_May16spines'
    #folder='./out2'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=0
    #folder='./out2_May17triggersoma'
    #folder='./out2_May18'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=0
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0
    #folder='./out2_May19'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=1
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0
    #folder='./out2_May20'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=1
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0
    #folder='./out2_May24'
    #folder='./out2_May24_triggerdistal'
    #folder='./out2_May25'
    #folder='./out2_May25_pairpulse'
    #timeStart = 145.0
    #timeEnd = 180.0
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=1
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0


    #folder='./out2_May26'
    #kfolder='./out2_May27_trigger_soma'
    folder='./out2_May27_fullspine'
    folder='./out2_May27'
    mapFolders = {}
    mapFolders[0] = 'out2_May29_rest'
    mapFolders[1] = 'out2_May29_triggersoma'
    mapFolders[2] = 'out2_May29_triggershaft'
    mapFolders[3] = 'out2_May29_triggerspine'
    mapFolders[4] = 'out2_May29_triggerdistalspines'
    mapFolders[5] = 'out2_May29_triggersoma_then_distalspines'
    mapFolders[6] = 'out2_May29_case06'
    tmpMap = {}
    if len(sys.argv) == 1:
      print("Help: python " + sys.argv[0] + " <number> <date>")
      print("  <number> = index of simulating protocol (use -1 if you only want to create folders)")
      print("  <date>   = the date of data, e.g. May28")
      sys.exit("")

    if len(sys.argv) > 2:#second arg should be the day (e.g. May29)
      for key,val in mapFolders.iteritems():
        print(val.replace('May29',sys.argv[2]))
        tmpMap[key] = val.replace('May29',sys.argv[2])
      mapFolders = tmpMap

    # create folder purpose (if not exist)
    for key,value in mapFolders.iteritems():
      if not os.path.exists(value):
        os.makedirs(value)

    if len(sys.argv) > 1 and int(sys.argv[1]) == -1:#first arg should be the protocol
      sys.exit("Folders created")

    if len(sys.argv) > 1:#first arg should be the protocol
      folder = mapFolders[int(sys.argv[1])]

    timeStart = 28.0
    #timeEnd = 60.0
    timeEnd = -1.0
    #timeStart = 270.0
    #timeEnd = 380.0
    somaMPIprocess=0
    middledenMPIprocess=1
    distaldenMPIprocess=1
    thinSpineMPIprocess=1
    perisomaticApicalDenMPIprocess=0
    perisomaticBasalDenMPIprocess=0
    axonAISMPIprocess = 0
    somaColor = 'blue'
    middledenColor= 'red'
    distaldenColor= 'green'
    perisomaticApicalDenColor= 'orange'
    perisomaticBasalDenColor= 'violet'
    axonAISColor = 'black'
    spineColor = 'magenta'
    preNeuronColor = 'brown'
    f, axarr = plt.subplots(3, 2)
    mpl.rcParams['lines.linewidth'] = 2
    t, v0 = np.loadtxt(folder + '/somaCa.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, somaColor, label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')

    if (os.path.isfile(folder+'/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/distaldendriticCa.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticCa.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, distaldenColor, label='distal-den')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/middledendriticCa.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticCa.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/axonAISCa.dat'+str(axonAISMPIprocess))):
        t, v1 = np.loadtxt(folder + '/axonAISCa.dat'+str(axonAISMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, axonAISColor, label='AIS')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    t, v0 = np.loadtxt(folder + '/somaV.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    axarr[1, 0].plot(t, v0, somaColor, label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0[idxStart:idxEnd]))
    maxVm = (np.amax(v0[idxStart:idxEnd]))
    #print(idxStart, idxEnd)
    #print(t[idxStart], t[idxEnd])
    #print(maxVm)
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);

    if (os.path.isfile(folder+'/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
        #axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    if (os.path.isfile(folder+'/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
        #axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    if (os.path.isfile(folder+'/axonAISV.dat'+str(axonAISMPIprocess))):
        t, v1 = np.loadtxt(folder + '/axonAISV.dat'+str(axonAISMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, axonAISColor, label='AIS')
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)
    if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)
    axarr[1, 0].legend()
    axarr[1, 0].set_ylim(bottom=minVm-0.5);
    axarr[1, 0].set_ylim(top=maxVm+0.5);
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)


    if (os.path.isfile(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[2, 0].legend()
    if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[2, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    if (os.path.isfile(folder + '/spineV.dat'+str(thinSpineMPIprocess))):
        #t, v1 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
        #                unpack=True, skiprows=1,
        #                usecols=(0, 1))
        t, v1,v2 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1,2))
        axarr[2, 0].plot(t, v1, spineColor, label='head-V')
        #jaxarr[2, 0].plot(t, v2, spineColor, linestyle='dashdot', label='neck-V')
        axarr[2, 0].plot(t, v2, 'black', linestyle='dashdot', label='neck-V')
    axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 0].legend()

    if (os.path.isfile(folder + '/proximalSpineV.dat'+str(thinSpineMPIprocess))):
        t, v1,v2 = np.loadtxt(folder + '/proximalSpineV.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1, 2))
        #axarr[2, 0].plot(t, v1, 'black', label='-50 um')
        axarr[2, 0].plot(t, v1, 'black', label='0 um')
        #axarr[2, 0].plot(t, v2, 'blue', label='-25 um')
        axarr[2, 0].plot(t, v2, 'blue', label='-10 um')

    if (os.path.isfile(folder + '/presynapticSomaVm.dat'+str(thinSpineMPIprocess))):
        t, v1 = np.loadtxt(folder + '/presynapticSomaVm.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, preNeuronColor, label='presyn-Soma-V')
    axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 0].legend()

    spinehead_area = 0.20 # um^2
    if (os.path.isfile(folder+'/spineNMDAR.dat'+str(thinSpineMPIprocess))):
        filename = folder + '/spineNMDAR.dat'+str(thinSpineMPIprocess)
        with open(filename) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols == 3):
          t, v1, v2 = np.loadtxt(filename,
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
          axarr[0, 1].plot(t, v2, 'green', label='I_NMDAR')
        elif (numcols ==2):
          t, v1 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR (pA)')
        axarr[0, 1].set_xlim(left=timeStart, right=timeEnd)
        axarr[0, 1].set_title("I_NMDAR (pA)");
    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        t, v1 = np.loadtxt(folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        axarr[0, 1].plot(t, v1, 'green', linestyle='--', label='I_AMPAR (pA)')
    axarr[0,1].legend()

    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        filename = folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess)
        with open(filename) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols >= 3):
          t, v1, v2 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                  usecols=(0,1,2))
          #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
          #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
          #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          axarr[1, 1].plot(t, v2, 'green', label='I_AMPAR')
          minCurrent = min(np.amin(v1[idxStart:idxEnd]),np.amin(v2[idxStart:idxEnd]))
          maxCurrent = max(np.amax(v1[idxStart:idxEnd]),np.amax(v2[idxStart:idxEnd]))
        elif (numcols == 2):
          t, v1 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                  usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          minCurrent = (np.amin(v1[idxStart:idxEnd]))
          maxCurrent = (np.amax(v1[idxStart:idxEnd]))
        axarr[1, 1].set_title("I_AMPAR (pA)");
        axarr[1, 1].set_ylim(bottom=minCurrent-0.5);
        axarr[1, 1].set_ylim(top=maxCurrent+0.5);
        axarr[1, 1].set_xlim(left=timeStart, right=timeEnd)
    # plt.legend(bbox_to_anchor=(1,1), loc=2)

    if (os.path.isfile(folder + '/spineCa.dat'+str(thinSpineMPIprocess))):
        #t, v1 = np.loadtxt(folder + '/spineCa.dat'+str(thinSpineMPIprocess),
        #                unpack=True, skiprows=1,
        #                usecols=(0, 1))
        t, v1,v2 = np.loadtxt(folder + '/spineCa.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1,2))
        axarr[2, 1].plot(t, v1, spineColor, label='spine-Ca')
        #axarr[2, 1].plot(t, v2, spineColor, linestyle='dashdot', label='neck-Ca')
        axarr[2, 1].plot(t, v2, 'black', linestyle='dashdot', label='neck-Ca')
    axarr[2, 1].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 1].legend()
    if (os.path.isfile(folder + '/proximalSpineCa.dat'+str(thinSpineMPIprocess))):
        t, v1,v2 = np.loadtxt(folder + '/proximalSpineCa.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1, 2))
        #axarr[2, 1].plot(t, v1, 'black', label='-50 um')
        #axarr[2, 1].plot(t, v2, 'blue', label='-25 um')
        axarr[2, 1].plot(t, v1, 'black', label='0 um')
        axarr[2, 1].plot(t, v2, 'blue', label='-10 um')
    plt.show()

def plot_case9():
    #more comprehensive deal with multiple processes I/O
    # with spines
    #folder='./out2_May16'
    #somaMPIprocess=10
    #middledenMPIprocess=16
    #distaldenMPIprocess=23
    #thinSpineMPIprocess=15
    #folder='./out2_May16spines'
    #folder='./out2'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=0
    #folder='./out2_May17triggersoma'
    #folder='./out2_May18'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=0
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0
    #folder='./out2_May19'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=1
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0
    #folder='./out2_May20'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=1
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0
    #folder='./out2_May24'
    #folder='./out2_May24_triggerdistal'
    #folder='./out2_May25'
    #folder='./out2_May25_pairpulse'
    #timeStart = 145.0
    #timeEnd = 180.0
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=1
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0

    args = parse()
    print (args.subparsers_name)
    print(args)
    #mainFolder="./"
    mainFolder="./data/"
    if (args.subparsers_name == 'folder'):
      folder = mainFolder + args.folderName
    elif (args.subparsers_name == 'protocol'):
      folder='./out2_May27_fullspine'
      folder='./out2_May27'
      mapFolders = {}
      mapFolders[0] = mainFolder + 'out2_May29_rest'
      mapFolders[1] = mainFolder + 'out2_May29_triggersoma'
      mapFolders[2] = mainFolder + 'out2_May29_triggershaft'
      mapFolders[3] = mainFolder + 'out2_May29_triggerspine'
      mapFolders[4] = mainFolder + 'out2_May29_triggerdistalspines'
      mapFolders[5] = mainFolder + 'out2_May29_triggersoma_then_distalspines'
      mapFolders[6] = mainFolder + 'out2_May29_case06'
      tmpMap = {}
      date = args.date
      protocol = args.number
      for key,val in mapFolders.iteritems():
        #print(val.replace('May29',date))
        tmpMap[key] = val.replace('May29',date)
      mapFolders = tmpMap
      # create folder purpose (if not exist)
      for key,value in mapFolders.iteritems():
        if not os.path.exists(value):
          os.makedirs(value)
      if int(protocol) == -1:#first arg should be the protocol
        sys.exit("Folders created")
      folder = mapFolders[int(protocol)]
      print("Plot folder " + folder)
    else:
      print("Unknown method")
      sys.exit("")
    #print(args)

    #folder='./out2_May26'
    #kfolder='./out2_May27_trigger_soma'
    #folder='./out2_May27_fullspine'
    #folder='./out2_May27'
    #mapFolders = {}
    #mapFolders[0] = 'out2_May29_rest'
    #mapFolders[1] = 'out2_May29_triggersoma'
    #mapFolders[2] = 'out2_May29_triggershaft'
    #mapFolders[3] = 'out2_May29_triggerspine'
    #mapFolders[4] = 'out2_May29_triggerdistalspines'
    #mapFolders[5] = 'out2_May29_triggersoma_then_distalspines'
    #mapFolders[6] = 'out2_May29_case06'
    #tmpMap = {}
    #if len(sys.argv) == 1:
    #  print("Help: python " + sys.argv[0] + " <number> <date>")
    #  print("  <number> = index of simulating protocol (use -1 if you only want to create folders)")
    #  print("  <date>   = the date of data, e.g. May28")
    #  sys.exit("")

    #if len(sys.argv) > 2:#second arg should be the day (e.g. May29)
    #  for key,val in mapFolders.iteritems():
    #    print(val.replace('May29',sys.argv[2]))
    #    tmpMap[key] = val.replace('May29',sys.argv[2])
    #  mapFolders = tmpMap

    ## create folder purpose (if not exist)
    #for key,value in mapFolders.iteritems():
    #  if not os.path.exists(value):
    #    os.makedirs(value)

    #if len(sys.argv) > 1 and int(sys.argv[1]) == -1:#first arg should be the protocol
    #  sys.exit("Folders created")

    #if len(sys.argv) > 1:#first arg should be the protocol
    #  folder = mapFolders[int(sys.argv[1])]

    #timeStart = 28.0
    timeStart = 0.0
    #timeEnd = 60.0
    timeEnd = -1.0
    #timeStart = 270.0
    #timeEnd = 380.0
    somaMPIprocess=2
    middledenMPIprocess=3
    distaldenMPIprocess=1
    thinSpineMPIprocess=1
    perisomaticApicalDenMPIprocess=3
    perisomaticBasalDenMPIprocess=0
    axonAISMPIprocess = 2
    somaColor = 'blue'
    middledenColor= 'red'
    distaldenColor= 'green'
    perisomaticApicalDenColor= 'orange'
    perisomaticBasalDenColor= 'violet'
    axonAISColor = 'black'
    spineColor = 'magenta'
    preNeuronColor = 'brown'
    f, axarr = plt.subplots(3, 2)
    mpl.rcParams['lines.linewidth'] = 2
    t, v0 = np.loadtxt(folder + '/somaCa.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, somaColor, label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')

    if (os.path.isfile(folder+'/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/distaldendriticCa.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticCa.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, distaldenColor, label='distal-den')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/middledendriticCa.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticCa.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/axonAISCa.dat'+str(axonAISMPIprocess))):
        t, v1 = np.loadtxt(folder + '/axonAISCa.dat'+str(axonAISMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, axonAISColor, label='AIS')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    t, v0 = np.loadtxt(folder + '/somaV.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    axarr[1, 0].plot(t, v0, somaColor, label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0[idxStart:idxEnd]))
    maxVm = (np.amax(v0[idxStart:idxEnd]))
    #print(idxStart, idxEnd)
    #print(t[idxStart], t[idxEnd])
    #print(maxVm)
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);

    if (os.path.isfile(folder+'/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
        #axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    if (os.path.isfile(folder+'/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
        #axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    if (os.path.isfile(folder+'/axonAISV.dat'+str(axonAISMPIprocess))):
        t, v1 = np.loadtxt(folder + '/axonAISV.dat'+str(axonAISMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, axonAISColor, label='AIS')
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)
    if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)
    axarr[1, 0].legend()
    axarr[1, 0].set_ylim(bottom=minVm-0.5);
    axarr[1, 0].set_ylim(top=maxVm+0.5);
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)


    if (os.path.isfile(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[2, 0].legend()
    if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[2, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    myFile = folder + '/spineV.dat'+str(thinSpineMPIprocess)
    if (os.path.isfile(myFile)):
        with open(myFile, 'rb') as f:
            lines = [f.readline()]
            lines = [f.readline()]
        numCols = len(np.loadtxt(lines, dtype='float'))#.shape[1]
        if (numCols == 2):
            t, v1 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
                            unpack=True, skiprows=1,
                            usecols=(0, 1))
        else:
            t, v1,v2 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
                            unpack=True, skiprows=1,
                            usecols=(0, 1,2))
        axarr[2, 0].plot(t, v1, spineColor, label='head-V')
        #jaxarr[2, 0].plot(t, v2, spineColor, linestyle='dashdot', label='neck-V')
        if (numCols > 2):
            axarr[2, 0].plot(t, v2, 'black', linestyle='dashdot', label='neck-V')
    axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 0].legend()

    if (os.path.isfile(folder + '/proximalSpineV.dat'+str(thinSpineMPIprocess))):
        t, v1,v2 = np.loadtxt(folder + '/proximalSpineV.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1, 2))
        #axarr[2, 0].plot(t, v1, 'black', label='-50 um')
        axarr[2, 0].plot(t, v1, 'black', label='0 um')
        #axarr[2, 0].plot(t, v2, 'blue', label='-25 um')
        axarr[2, 0].plot(t, v2, 'blue', label='-10 um')

    if (os.path.isfile(folder + '/presynapticSomaVm.dat'+str(thinSpineMPIprocess))):
        t, v1 = np.loadtxt(folder + '/presynapticSomaVm.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, preNeuronColor, label='presyn-Soma-V')
    axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 0].legend()

    spinehead_area = 0.20 # um^2
    if (os.path.isfile(folder+'/spineNMDAR.dat'+str(thinSpineMPIprocess))):
        filename = folder + '/spineNMDAR.dat'+str(thinSpineMPIprocess)
        with open(filename) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols == 3):
          t, v1, v2 = np.loadtxt(filename,
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
          axarr[0, 1].plot(t, v2, 'green', label='I_NMDAR')
        elif (numcols ==2):
          t, v1 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR (pA)')
        axarr[0, 1].set_xlim(left=timeStart, right=timeEnd)
        axarr[0, 1].set_title("I_NMDAR (pA)");
    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        t, v1 = np.loadtxt(folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        axarr[0, 1].plot(t, v1, 'green', linestyle='--', label='I_AMPAR (pA)')
    axarr[0,1].legend()

    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        filename = folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess)
        with open(filename) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols >= 3):
          t, v1, v2 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                  usecols=(0,1,2))
          #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
          #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
          #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          axarr[1, 1].plot(t, v2, 'green', label='I_AMPAR')
          minCurrent = min(np.amin(v1[idxStart:idxEnd]),np.amin(v2[idxStart:idxEnd]))
          maxCurrent = max(np.amax(v1[idxStart:idxEnd]),np.amax(v2[idxStart:idxEnd]))
        elif (numcols == 2):
          t, v1 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                  usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          minCurrent = (np.amin(v1[idxStart:idxEnd]))
          maxCurrent = (np.amax(v1[idxStart:idxEnd]))
        axarr[1, 1].set_title("I_AMPAR (pA)");
        axarr[1, 1].set_ylim(bottom=minCurrent-0.5);
        axarr[1, 1].set_ylim(top=maxCurrent+0.5);
        axarr[1, 1].set_xlim(left=timeStart, right=timeEnd)
    # plt.legend(bbox_to_anchor=(1,1), loc=2)

    myFile = folder + '/spineCa.dat'+str(thinSpineMPIprocess)
    if (os.path.isfile(myFile)):
        with open(myFile, 'rb') as f:
            lines = [f.readline()]
            lines = [f.readline()]
        numCols = len(np.loadtxt(lines, dtype='float'))#.shape[1]
        if (numCols == 2):
            t, v1 = np.loadtxt(folder + '/spineCa.dat'+str(thinSpineMPIprocess),
                            unpack=True, skiprows=1,
                            usecols=(0, 1))
        else:
            t, v1,v2 = np.loadtxt(folder + '/spineCa.dat'+str(thinSpineMPIprocess),
                            unpack=True, skiprows=1,
                            usecols=(0, 1,2))
        axarr[2, 1].plot(t, v1, spineColor, label='spine-Ca')
        #axarr[2, 1].plot(t, v2, spineColor, linestyle='dashdot', label='neck-Ca')
        if (numCols > 2):
            axarr[2, 1].plot(t, v2, 'black', linestyle='dashdot', label='neck-Ca')
    axarr[2, 1].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 1].legend()
    if (os.path.isfile(folder + '/proximalSpineCa.dat'+str(thinSpineMPIprocess))):
        t, v1,v2 = np.loadtxt(folder + '/proximalSpineCa.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1, 2))
        #axarr[2, 1].plot(t, v1, 'black', label='-50 um')
        #axarr[2, 1].plot(t, v2, 'blue', label='-25 um')
        axarr[2, 1].plot(t, v1, 'black', label='0 um')
        axarr[2, 1].plot(t, v2, 'blue', label='-10 um')
    plt.show()

def plot_case9_adv():
    #more comprehensive deal with multiple processes I/O
    # with spines

    args = parse()
    print (args.subparsers_name)
    print(args)
    #mainFolder="./"
    mainFolder="./data/"
    if (args.subparsers_name == 'folder'):
      folder = mainFolder + args.folderName
    elif (args.subparsers_name == 'protocol'):
      folder='./out2_May27_fullspine'
      folder='./out2_May27'
      morph = args.morphology
      if (morph != ""):
          morph += "_"
      mapFolders = {}
      mapFolders[0] = mainFolder + morph + 'May29_rest'
      mapFolders[1] = mainFolder + morph + 'May29_triggersoma'
      mapFolders[2] = mainFolder + morph + 'May29_triggershaft'
      mapFolders[3] = mainFolder + morph + 'May29_triggerspine'
      mapFolders[4] = mainFolder + morph + 'May29_triggerdistalspines'
      mapFolders[5] = mainFolder + morph + 'May29_triggersoma_then_distalspines'
      mapFolders[6] = mainFolder + morph + 'May29_case06'
      tmpMap = {}
      date = args.date
      protocol = args.number
      for key,val in mapFolders.iteritems():
        #print(val.replace('May29',date))
        tmpMap[key] = val.replace('May29',date)
      mapFolders = tmpMap
      # create folder purpose (if not exist)
      for key,value in mapFolders.iteritems():
        if not os.path.exists(value):
          os.makedirs(value)
      if int(protocol) == -1:#first arg should be the protocol
        sys.exit("Folders created")
      folder = mapFolders[int(protocol)]
      print("Plot folder " + folder)
    else:
      print("Unknown method")
      sys.exit("")
    #print(args)

    #timeStart = 0.0
    timeStart = 200.0
    timeEnd = -1.0
    #timeEnd = 470.0
    #timeStart = 270.0
    #timeEnd = 380.0

    #colors = ['blue', 'red', 'green', 'black', 'orange', 'violet', 'magenta', 'brown']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--' ]

    #linestyles = ['_', '-', '--', ':']
    #markers = []
    #for m in LIne2D.markers:
    #    try:
    #        if len(m) == 1 and m != ' ':
    #            markers.append(m)
    #    except TypeError:
    #        pass
    #styles = markers + [
    #    r'$\lambda$',
    #    r'$\bowtie$',
    #    r'$\circlearrowleft$',
    #    r'$\clubsuit$',
    #    r'$\checkmark$']
    # page size (3 rows, 2 columns)
    #fig_01 = plt.figure(0)
    f1, axarr1 = plt.subplots(3, 2)
    axarr = axarr1

    # common configuration
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['legend.fontsize'] = 13


    ############################
    ### row 1
    ## For 1 file
    myFile = folder+'/'+getFile(folder,'somaCurrents.dat')
    YLABEL = "soma"
    file=open(myFile)
    lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    channelNames = re.split(",",re.sub('[\s+]','',lines[1]))
    del channelNames[-1], channelNames[0]
    #print lines[1]
    print channelNames
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    hide = []   #decide what channel to hide
    #hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
    #hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 0; gc = 0 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        style = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        if (not channelNames[i] in hide):
            axarr[gr, gc].plot(t, ydata , linestyle=style, color=color, label=channelNames[i])
            axarr[gr, gc].legend()
            minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
            maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. currents (pA/um^2)")
    axarr[gr, gc].set_ylabel(YLABEL)


    ############################
    ## For 1 file
    ### first y-axis (Voltage)
    myFile = folder+'/'+getFile(folder,'somaV.dat')
    #file=open(myFile)
    #lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    #del channelNames[-1], channelNames[0]
    #print lines[1]
    #print channelNames
    channelNames=["Vm"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=1,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 0; gc = 1 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        axarr[gr, gc].legend(loc=4)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. Vm | Calcium")
    for tl in axarr[gr, gc].get_yticklabels():
        tl.set_color(color)
    ax1 = axarr[gr, gc]
    ### second y-axis
    ## Calcium
    myFile = folder+'/'+getFile(folder,'somaCa.dat')
    channelNames=["Ca"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    #gr = 0; gc = 1 # graph row, col
    ax2 = ax1.twinx()
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[(i+1)%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        #axarr[gr, gc].legend()
        ax2.plot(t, ydata , color, label=channelNames[i])
        ax2.legend(loc=0)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    #s2 = np.sin(2*np.pi*t)
    #ax2.plot(t, s2, 'r.')
    #ax2.set_ylabel('sin', color='r')
    ax2.set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    ax2.set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    ax2.set_xlim(left=timeStart, right=timeEnd)
    for tl in ax2.get_yticklabels():
        tl.set_color(color)
    print minVal, maxVal

    ############################
    ### row 2
    ## For 1 file
    myFile = folder+'/'+getFile(folder,'axonAISCurrents.dat')
    YLABEL = "axon-AIS"
    file=open(myFile)
    lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    channelNames = re.split(",",re.sub('[\s+]','',lines[1]))
    del channelNames[-1], channelNames[0]
    #print lines[1]
    print channelNames
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    hide = []   #decide what channel to hide
    #hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
    #hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 1; gc = 0 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        if (not channelNames[i] in hide):
            axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
            axarr[gr, gc].legend()
            minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
            maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. currents (pA/um^2)")
    axarr[gr, gc].set_ylabel(YLABEL)


    ############################
    ## For 1 file
    ### first y-axis (Voltage)
    myFile = folder+'/'+getFile(folder,'axonAISV.dat')
    #file=open(myFile)
    #lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    #del channelNames[-1], channelNames[0]
    #print lines[1]
    #print channelNames
    channelNames=["Vm"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 1; gc = 1 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        axarr[gr, gc].legend(loc=4)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. Vm | Calcium")
    for tl in axarr[gr, gc].get_yticklabels():
        tl.set_color(color)
    ax1 = axarr[gr, gc]
    ### second y-axis
    ## Calcium
    myFile = folder+'/'+getFile(folder,'axonAISCa.dat')
    if (os.path.isfile(myFile)):
      channelNames=["Ca"]
      nCols = len(channelNames)

      arrays = np.loadtxt(myFile,
                          skiprows=2,
                        # usecols=(0, 1)
                         )
      arraysT = np.transpose(arrays)
      t = arraysT[0]
      #############
      # Define time range
      idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
      if (timeEnd == -1.0):
        idxEnd = len(t)-1
        timeEnd = t[idxEnd]
      else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
      #END
      #################
      arraysT = np.delete(arraysT, 0, 0)
      #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
      #print arr
      #arr = np.delete(arr,0,1)
      #print arr
      #return
      #print t
      #print nCols
      minVal = 200000000.0
      maxVal = -minVal
      i =0
      #gr = 0; gc = 1 # graph row, col
      ax2 = ax1.twinx()
      for ydata in arraysT:
          #plot (t, col)
          # legend(time, channelNames[i])
          color = colors[(i+1)%len(colors)]
          marker = linestyles[((i/len(colors))%len(linestyles))]
          #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
          #axarr[gr, gc].legend()
          ax2.plot(t, ydata , color, label=channelNames[i])
          ax2.legend(loc=0)
          minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
          maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
          #print len(t), len(ydata)
          i += 1
      #s2 = np.sin(2*np.pi*t)
      #ax2.plot(t, s2, 'r.')
      #ax2.set_ylabel('sin', color='r')
      ax2.set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
      ax2.set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
      ax2.set_xlim(left=timeStart, right=timeEnd)
      for tl in ax2.get_yticklabels():
          tl.set_color(color)
      print minVal, maxVal


    ############################
    ### row 3
    ## For 1 file
    myFile = folder+'/'+getFile(folder,'proximalTrunkCurrents.dat')
    YLABEL = "proximalTrunk"
    file=open(myFile)
    lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    channelNames = re.split(",",re.sub('[\s+]','',lines[1]))
    del channelNames[-1], channelNames[0]
    #print lines[1]
    print channelNames
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    hide = []   #decide what channel to hide
    #hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
    #hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 2; gc = 0 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        if (not channelNames[i] in hide):
            axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
            axarr[gr, gc].legend()
            minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
            maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. currents (pA/um^2)")
    axarr[gr, gc].set_ylabel(YLABEL)
    datacursor(axarr[gr,gc])


    ############################
    ## For 1 file
    ### first y-axis (Voltage)
    myFile = folder+'/'+getFile(folder,'proximalTrunkV.dat')
    #file=open(myFile)
    #lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    #del channelNames[-1], channelNames[0]
    #print lines[1]
    #print channelNames
    channelNames=["Vm"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 2; gc = 1 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        axarr[gr, gc].legend(loc=4)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. Vm | Calcium")
    for tl in axarr[gr, gc].get_yticklabels():
        tl.set_color(color)
    ax1 = axarr[gr, gc]
    ### second y-axis
    ## Calcium
    myFile = folder+'/'+getFile(folder,'proximalTrunkCa.dat')
    if (os.path.isfile(myFile)):
      channelNames=["Ca"]
      nCols = len(channelNames)

      arrays = np.loadtxt(myFile,
                          skiprows=2,
                        # usecols=(0, 1)
                         )
      arraysT = np.transpose(arrays)
      t = arraysT[0]
      #############
      # Define time range
      idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
      if (timeEnd == -1.0):
        idxEnd = len(t)-1
        timeEnd = t[idxEnd]
      else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
      #END
      #################
      arraysT = np.delete(arraysT, 0, 0)
      #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
      #print arr
      #arr = np.delete(arr,0,1)
      #print arr
      #return
      #print t
      #print nCols
      minVal = 200000000.0
      maxVal = -minVal
      i =0
      #gr = 0; gc = 1 # graph row, col
      ax2 = ax1.twinx()
      for ydata in arraysT:
          #plot (t, col)
          # legend(time, channelNames[i])
          color = colors[(i+1)%len(colors)]
          marker = linestyles[((i/len(colors))%len(linestyles))]
          #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
          #axarr[gr, gc].legend()
          ax2.plot(t, ydata , color, label=channelNames[i])
          ax2.legend(loc=0)
          minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
          maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
          #print len(t), len(ydata)
          i += 1
      #s2 = np.sin(2*np.pi*t)
      #ax2.plot(t, s2, 'r.')
      #ax2.set_ylabel('sin', color='r')
      ax2.set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
      ax2.set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
      ax2.set_xlim(left=timeStart, right=timeEnd)
      for tl in ax2.get_yticklabels():
          tl.set_color(color)
      print minVal, maxVal
    datacursor(ax1)

    ##########################################################
    ### Page 2
    f2, axarr2 = plt.subplots(3, 2)
    axarr = axarr2
    ############################
    ### row 1
    ## For 1 file
    myFile = folder+'/'+getFile(folder,'distalTuftedCurrents.dat')
    YLABEL = "distalTufted"
    file=open(myFile)
    lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    channelNames = re.split(",",re.sub('[\s+]','',lines[1]))
    del channelNames[-1], channelNames[0]
    #print lines[1]
    print channelNames
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    hide = []   #decide what channel to hide
    #hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
    #hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 0; gc = 0 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        if (not channelNames[i] in hide):
            axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
            axarr[gr, gc].legend()
            minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
            maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. currents (pA/um^2)")
    axarr[gr, gc].set_ylabel(YLABEL)


    ############################
    ## For 1 file
    ### first y-axis (Voltage)
    myFile = folder+'/'+getFile(folder,'distalTuftedV.dat')
    #file=open(myFile)
    #lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    #del channelNames[-1], channelNames[0]
    #print lines[1]
    #print channelNames
    channelNames=["Vm"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=1,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 0; gc = 1 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        axarr[gr, gc].legend(loc=4)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. Vm | Calcium")
    for tl in axarr[gr, gc].get_yticklabels():
        tl.set_color(color)
    ax1 = axarr[gr, gc]
    ### second y-axis
    ## Calcium
    myFile = folder+'/'+getFile(folder,'distalTuftedCa.dat')
    channelNames=["Ca"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    #gr = 0; gc = 1 # graph row, col
    ax2 = ax1.twinx()
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[(i+1)%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        #axarr[gr, gc].legend()
        ax2.plot(t, ydata , color, label=channelNames[i])
        ax2.legend(loc=0)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    #s2 = np.sin(2*np.pi*t)
    #ax2.plot(t, s2, 'r.')
    #ax2.set_ylabel('sin', color='r')
    ax2.set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    ax2.set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    ax2.set_xlim(left=timeStart, right=timeEnd)
    for tl in ax2.get_yticklabels():
        tl.set_color(color)
    print minVal, maxVal

    ############################
    ### row 2
    ## For 1 file
    myFile = folder+'/'+getFile(folder,'proximalTuftedCurrents.dat')
    YLABEL = "proximalTufted"
    file=open(myFile)
    lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    channelNames = re.split(",",re.sub('[\s+]','',lines[1]))
    del channelNames[-1], channelNames[0]
    #print lines[1]
    print channelNames
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    hide = []   #decide what channel to hide
    #hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
    #hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 1; gc = 0 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        if (not channelNames[i] in hide):
            axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
            axarr[gr, gc].legend()
            minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
            maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. currents (pA/um^2)")
    axarr[gr, gc].set_ylabel(YLABEL)


    ############################
    ## For 1 file
    ### first y-axis (Voltage)
    myFile = folder+'/'+getFile(folder,'proximalTuftedV.dat')
    #file=open(myFile)
    #lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    #del channelNames[-1], channelNames[0]
    #print lines[1]
    #print channelNames
    channelNames=["Vm"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 1; gc = 1 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        axarr[gr, gc].legend(loc=4)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. Vm | Calcium")
    for tl in axarr[gr, gc].get_yticklabels():
        tl.set_color(color)
    ax1 = axarr[gr, gc]
    ### second y-axis
    ## Calcium
    myFile = folder+'/'+getFile(folder,'proximalTuftedCa.dat')
    if (os.path.isfile(myFile)):
      channelNames=["Ca"]
      nCols = len(channelNames)

      arrays = np.loadtxt(myFile,
                          skiprows=2,
                        # usecols=(0, 1)
                         )
      arraysT = np.transpose(arrays)
      t = arraysT[0]
      #############
      # Define time range
      idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
      if (timeEnd == -1.0):
        idxEnd = len(t)-1
        timeEnd = t[idxEnd]
      else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
      #END
      #################
      arraysT = np.delete(arraysT, 0, 0)
      #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
      #print arr
      #arr = np.delete(arr,0,1)
      #print arr
      #return
      #print t
      #print nCols
      minVal = 200000000.0
      maxVal = -minVal
      i =0
      #gr = 0; gc = 1 # graph row, col
      ax2 = ax1.twinx()
      for ydata in arraysT:
          #plot (t, col)
          # legend(time, channelNames[i])
          color = colors[(i+1)%len(colors)]
          marker = linestyles[((i/len(colors))%len(linestyles))]
          #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
          #axarr[gr, gc].legend()
          ax2.plot(t, ydata , color, label=channelNames[i])
          ax2.legend(loc=0)
          minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
          maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
          #print len(t), len(ydata)
          i += 1
      #s2 = np.sin(2*np.pi*t)
      #ax2.plot(t, s2, 'r.')
      #ax2.set_ylabel('sin', color='r')
      ax2.set_ylim(bottom=minVal - .05*max(0.1,abs(minVal)))
      ax2.set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
      ax2.set_xlim(left=timeStart, right=timeEnd)
      for tl in ax2.get_yticklabels():
          tl.set_color(color)
      print minVal, maxVal


    ############################
    ### row 3
    ## For 1 file
    myFile = folder+'/'+getFile(folder,'distalTrunkCurrents.dat')
    YLABEL = "distalTrunk"
    file=open(myFile)
    lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    channelNames = re.split(",",re.sub('[\s+]','',lines[1]))
    del channelNames[-1], channelNames[0]
    #print lines[1]
    print channelNames
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    hide = []   #decide what channel to hide
    #hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
    #hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 2; gc = 0 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        if (not channelNames[i] in hide):
            axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
            axarr[gr, gc].legend()
            minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
            maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. currents (pA/um^2)")
    axarr[gr, gc].set_ylabel(YLABEL)
    datacursor(axarr[gr,gc])


    ############################
    ## For 1 file
    ### first y-axis (Voltage)
    myFile = folder+'/'+getFile(folder,'distalTrunkV.dat')
    #file=open(myFile)
    #lines=file.readlines()
    #channelNames = re.split(",",lines[1])
    #del channelNames[-1], channelNames[0]
    #print lines[1]
    #print channelNames
    channelNames=["Vm"]
    nCols = len(channelNames)

    arrays = np.loadtxt(myFile,
                        skiprows=2,
                      # usecols=(0, 1)
                       )
    arraysT = np.transpose(arrays)
    t = arraysT[0]
    #############
    # Define time range
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
    #END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    #print arr
    #arr = np.delete(arr,0,1)
    #print arr
    #return
    #print t
    #print nCols
    minVal = 200000000.0
    maxVal = -minVal
    i =0
    gr = 2; gc = 1 # graph row, col
    for ydata in arraysT:
        #plot (t, col)
        # legend(time, channelNames[i])
        color = colors[i%len(colors)]
        marker = linestyles[((i/len(colors))%len(linestyles))]
        axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
        axarr[gr, gc].legend(loc=4)
        minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
        maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
        #print len(t), len(ydata)
        i += 1
    print i
    print minVal, maxVal
    axarr[gr, gc].set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
    axarr[gr, gc].set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
    axarr[gr, gc].set_xlim(left=timeStart, right=timeEnd)
    axarr[gr, gc].set_title("time (ms) vs. Vm | Calcium")
    for tl in axarr[gr, gc].get_yticklabels():
        tl.set_color(color)
    ax1 = axarr[gr, gc]
    ### second y-axis
    ## Calcium
    myFile = folder+'/'+getFile(folder,'distalTrunkCa.dat')
    if (os.path.isfile(myFile)):
      channelNames=["Ca"]
      nCols = len(channelNames)

      arrays = np.loadtxt(myFile,
                          skiprows=2,
                        # usecols=(0, 1)
                         )
      arraysT = np.transpose(arrays)
      t = arraysT[0]
      #############
      # Define time range
      idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
      if (timeEnd == -1.0):
        idxEnd = len(t)-1
        timeEnd = t[idxEnd]
      else:
        try:
            idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)
        except StopIteration as e:
            idxEnd = len(t)-1
      #END
      #################
      arraysT = np.delete(arraysT, 0, 0)
      #arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
      #print arr
      #arr = np.delete(arr,0,1)
      #print arr
      #return
      #print t
      #print nCols
      minVal = 200000000.0
      maxVal = -minVal
      i =0
      #gr = 0; gc = 1 # graph row, col
      ax2 = ax1.twinx()
      for ydata in arraysT:
          #plot (t, col)
          # legend(time, channelNames[i])
          color = colors[(i+1)%len(colors)]
          marker = linestyles[((i/len(colors))%len(linestyles))]
          #axarr[gr, gc].plot(t, ydata , color, label=channelNames[i])
          #axarr[gr, gc].legend()
          ax2.plot(t, ydata , color, label=channelNames[i])
          ax2.legend(loc=0)
          minVal = min(np.amin(ydata[idxStart:idxEnd]), minVal)
          maxVal = max(np.amax(ydata[idxStart:idxEnd]), maxVal)
          #print len(t), len(ydata)
          i += 1
      #s2 = np.sin(2*np.pi*t)
      #ax2.plot(t, s2, 'r.')
      #ax2.set_ylabel('sin', color='r')
      ax2.set_ylim(bottom=minVal - 0.05*max(0.1,abs(minVal)))
      ax2.set_ylim(top=maxVal + 0.05*max(0.1,abs(maxVal)))
      ax2.set_xlim(left=timeStart, right=timeEnd)
      for tl in ax2.get_yticklabels():
          tl.set_color(color)
      print minVal, maxVal
    datacursor()
    plt.show()

    #f, axarr = plt.subplots(3, 2)
    return

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, somaColor, label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')

    myFile = folder+'/'+getFile(folder,'proximalTrunkCa.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, proximalTrunkColor, label='perisomatic-ApicalDen')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    myFile = folder+'/'+getFile(folder,'proximalBasalDenCa.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, proximalBasalDenColor, label='perisomatic-BasalDen')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    myFile = folder+'/'+getFile(folder,'distalTuftedCa.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, distaldenColor, label='distal-den')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    myFile = folder+'/'+getFile(folder,'distalTrunkCa.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    myFile = folder+'/'+getFile(folder,'axonAISCa.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, axonAISColor, label='AIS')
        axarr[0, 0].legend()
        minCa = min(np.amin(v1), minCa)
        maxCa = max(np.amax(v1), maxCa)

    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    myFile = folder+'/'+getFile(folder,'somaV.dat')
    t, v0 = np.loadtxt(myFile,
                       unpack=True, skiprows=1,
                       usecols=(0, 1))
    axarr[1, 0].plot(t, v0, somaColor, label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0[idxStart:idxEnd]))
    maxVm = (np.amax(v0[idxStart:idxEnd]))
    #print(idxStart, idxEnd)
    #print(t[idxStart], t[idxEnd])
    #print(maxVm)
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);

    myFile = folder+'/'+getFile(folder,'proximalTrunkV.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, proximalTrunkColor, label='perisomatic-ApicalDen')
        #axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    myFile = folder+'/'+getFile(folder,'proximalBasalDenV.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, proximalBasalDenColor, label='perisomatic-BasalDen')
        #axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    myFile = folder+'/'+getFile(folder,'axonAISV.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, axonAISColor, label='AIS')
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    myFile = folder+'/'+getFile(folder,'distalTrunkV.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[1, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)
    axarr[1, 0].legend()
    axarr[1, 0].set_ylim(bottom=minVm-0.5);
    axarr[1, 0].set_ylim(top=maxVm+0.5);
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)

    myFile = folder+'/'+getFile(folder,'distalTuftedV.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[2, 0].legend()

    myFile = folder+'/'+getFile(folder,'proximalTuftedV.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, tufteddenColor, label='tufted-den')
    axarr[2, 0].legend()

    myFile = folder+'/'+getFile(folder,'distalTrunkV.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[2, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    myFile = folder+'/'+getFile(folder,'spineV.dat')
    if (os.path.isfile(myFile)):
        with open(myFile, 'rb') as f:
            lines = [f.readline()]
            lines = [f.readline()]
        numCols = len(np.loadtxt(lines, dtype='float'))#.shape[1]
        if (numCols == 2):
            t, v1 = np.loadtxt(myFile,
                            unpack=True, skiprows=1,
                            usecols=(0, 1))
        else:
            t, v1,v2 = np.loadtxt(myFile,
                            unpack=True, skiprows=1,
                            usecols=(0, 1,2))
        axarr[2, 0].plot(t, v1, spineColor, label='head-V')
        #jaxarr[2, 0].plot(t, v2, spineColor, linestyle='dashdot', label='neck-V')
        if (numCols > 2):
            axarr[2, 0].plot(t, v2, 'black', linestyle='dashdot', label='neck-V')
    axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 0].legend()

    myFile = folder+'/'+getFile(folder,'proximalSpineV.dat')
    if (os.path.isfile(myFile)):
        t, v1,v2 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1, 2))
        #axarr[2, 0].plot(t, v1, 'black', label='-50 um')
        axarr[2, 0].plot(t, v1, 'black', label='0 um')
        #axarr[2, 0].plot(t, v2, 'blue', label='-25 um')
        axarr[2, 0].plot(t, v2, 'blue', label='-10 um')

    myFile = folder+'/'+getFile(folder,'presynapticSomaVm.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, preNeuronColor, label='presyn-Soma-V')
    axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 0].legend()

    spinehead_area = 0.20 # um^2
    myFile = folder+'/'+getFile(folder,'spineNMDAR.dat')
    if (os.path.isfile(myFile)):
        with open(myFile) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols == 3):
          t, v1, v2 = np.loadtxt(myFile,
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
          axarr[0, 1].plot(t, v2, 'green', label='I_NMDAR')
        elif (numcols ==2):
          t, v1 = np.loadtxt(myFile,
                                unpack=True,skiprows=1,
                                usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR (pA)')
        axarr[0, 1].set_xlim(left=timeStart, right=timeEnd)
        axarr[0, 1].set_title("I_NMDAR (pA)");

    myFile = folder+'/'+getFile(folder,'spineAMPAR.dat')
    if (os.path.isfile(myFile)):
        t, v1 = np.loadtxt(myFile,
                               unpack=True,skiprows=1,
                                usecols=(0,1))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        axarr[0, 1].plot(t, v1, 'green', linestyle='--', label='I_AMPAR (pA)')
    axarr[0,1].legend()

    myFile = folder+'/'+getFile(folder,'spineAMPAR.dat')
    if (os.path.isfile(myFile)):
        with open(myFile) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols >= 3):
          t, v1, v2 = np.loadtxt(myFile,
                                unpack=True,skiprows=1,
                                  usecols=(0,1,2))
          #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
          #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
          #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          axarr[1, 1].plot(t, v2, 'green', label='I_AMPAR')
          minCurrent = min(np.amin(v1[idxStart:idxEnd]),np.amin(v2[idxStart:idxEnd]))
          maxCurrent = max(np.amax(v1[idxStart:idxEnd]),np.amax(v2[idxStart:idxEnd]))
        elif (numcols == 2):
          t, v1 = np.loadtxt(myFile,
                                unpack=True,skiprows=1,
                                  usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          minCurrent = (np.amin(v1[idxStart:idxEnd]))
          maxCurrent = (np.amax(v1[idxStart:idxEnd]))
        axarr[1, 1].set_title("I_AMPAR (pA)");
        axarr[1, 1].set_ylim(bottom=minCurrent-0.5);
        axarr[1, 1].set_ylim(top=maxCurrent+0.5);
        axarr[1, 1].set_xlim(left=timeStart, right=timeEnd)
    # plt.legend(bbox_to_anchor=(1,1), loc=2)

    myFile = folder+'/'+getFile(folder,'spineCa.dat')
    if (os.path.isfile(myFile)):
        with open(myFile, 'rb') as f:
            lines = [f.readline()]
            lines = [f.readline()]
        numCols = len(np.loadtxt(lines, dtype='float'))#.shape[1]
        if (numCols == 2):
            t, v1 = np.loadtxt(folder + '/spineCa.dat'+str(thinSpineMPIprocess),
                            unpack=True, skiprows=1,
                            usecols=(0, 1))
        else:
            t, v1,v2 = np.loadtxt(folder + '/spineCa.dat'+str(thinSpineMPIprocess),
                            unpack=True, skiprows=1,
                            usecols=(0, 1,2))
        axarr[2, 1].plot(t, v1, spineColor, label='spine-Ca')
        #axarr[2, 1].plot(t, v2, spineColor, linestyle='dashdot', label='neck-Ca')
        if (numCols > 2):
            axarr[2, 1].plot(t, v2, 'black', linestyle='dashdot', label='neck-Ca')
    axarr[2, 1].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 1].legend()
    myFile = folder+'/'+getFile(folder,'proximalSpineCa.dat')
    if (os.path.isfile(myFile)):
        t, v1,v2 = np.loadtxt(myFile,
                        unpack=True, skiprows=1,
                        usecols=(0, 1, 2))
        #axarr[2, 1].plot(t, v1, 'black', label='-50 um')
        #axarr[2, 1].plot(t, v2, 'blue', label='-25 um')
        axarr[2, 1].plot(t, v1, 'black', label='0 um')
        axarr[2, 1].plot(t, v2, 'blue', label='-10 um')
    plt.show()

def plot_current_1():
    #more comprehensive deal with multiple processes I/O
    # with spines
    #folder='./out2_May16'
    #somaMPIprocess=10
    #middledenMPIprocess=16
    #distaldenMPIprocess=23
    #thinSpineMPIprocess=15
    #folder='./out2_May16spines'
    #folder='./out2'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=0
    #folder='./out2_May17triggersoma'
    #folder='./out2_May18'
    #somaMPIprocess=0
    #middledenMPIprocess=1
    #distaldenMPIprocess=1
    #thinSpineMPIprocess=0
    #perisomaticApicalDenMPIprocess=0
    #perisomaticBasalDenMPIprocess=0
    #axonAISMPIprocess = 0
    folder='./out2_May19'
    somaMPIprocess=0
    middledenMPIprocess=1
    distaldenMPIprocess=1
    thinSpineMPIprocess=1
    perisomaticApicalDenMPIprocess=0
    perisomaticBasalDenMPIprocess=0
    axonAISMPIprocess = 0
    somaColor = 'blue'
    middledenColor= 'red'
    distaldenColor= 'green'
    perisomaticApicalDenColor= 'orange'
    perisomaticBasalDenColor= 'violet'
    axonAISColor = 'black'
    spineColor = 'magenta'
    preNeuronColor = 'brown'
    timeStart = 0
    timeEnd=1000
    f, axarr = plt.subplots(3, 2)
    t, v0 = np.loadtxt(folder + '/somaCa.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, somaColor, label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')

    if (os.path.isfile(folder+'/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/distaldendriticCa.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticCa.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/middledendriticCa.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticCa.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, middledenColor, label='middle-den')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)

    if (os.path.isfile(folder+'/axonAISCa.dat'+str(axonAISMPIprocess))):
        t, v1 = np.loadtxt(folder + '/axonAISCa.dat'+str(axonAISMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[0, 0].plot(t, v1, axonAISColor, label='AIS')
    axarr[0, 0].legend()
    minCa = min(np.amin(v1), minCa)
    maxCa = max(np.amax(v1), maxCa)

    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    t, v0 = np.loadtxt(folder + '/somaV.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    axarr[1, 0].plot(t, v0, somaColor, label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0))
    maxVm = (np.amax(v0))
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);

    if (os.path.isfile(folder+'/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
    #axarr[1, 0].legend()
    minVm = min(np.amin(v1), minVm)
    maxVm = max(np.amax(v1), maxVm)

    if (os.path.isfile(folder+'/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
    #axarr[1, 0].legend()
    minVm = min(np.amin(v1), minVm)
    maxVm = max(np.amax(v1), maxVm)

    if (os.path.isfile(folder+'/axonAISV.dat'+str(axonAISMPIprocess))):
        t, v1 = np.loadtxt(folder + '/axonAISV.dat'+str(axonAISMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[1, 0].plot(t, v1, axonAISColor, label='AIS')
    axarr[1, 0].legend()
    minVm = min(np.amin(v1), minVm)
    maxVm = max(np.amax(v1), maxVm)
    axarr[1, 0].set_ylim(bottom=minVm-0.5);
    axarr[1, 0].set_ylim(top=maxVm+0.5);
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)


    if (os.path.isfile(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[2, 0].legend()
    if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, middledenColor, label='middle-den')
        minVm = min(np.amin(v1), minVm)
        maxVm = max(np.amax(v1), maxVm)
    axarr[2, 0].legend()

    if (os.path.isfile(folder + '/spineV.dat'+str(thinSpineMPIprocess))):
        #t, v1 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
        #                unpack=True, skiprows=1,
        #                usecols=(0, 1))
        t, v1,v2 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1,2))
        axarr[2, 0].plot(t, v1, spineColor, label='spine')
        axarr[2, 0].plot(t, v2, spineColor, linestyle='dashdot', label='spine')
    axarr[2, 0].legend()

    spinehead_area = 0.20 # um^2
    if (os.path.isfile(folder+'/spineNMDAR.dat'+str(thinSpineMPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineNMDAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
        #axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        #axarr[1, 1].plot(t, v2, 'green', label='I_NMDAR')
        #axarr[0, 1].set_title("I_NMDAR (pA/uM^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
        axarr[0, 1].plot(t, v2, 'green', label='I_NMDAR')
        axarr[0, 1].set_title("I_NMDAR (pA)");
    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        print("AAA")
        t, v1, v2 = np.loadtxt(folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1,2))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[0, 1].plot(t, v1, 'red', linestyle='--', label='I_AMPAR')
        axarr[0, 1].plot(t, v2, 'green',linestyle='dashdot', label='I_AMPAR')
    axarr[0,1].legend()

    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        t, v1, v2 = np.loadtxt(folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1,2))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        v2 = v2 * spinehead_area
        axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
        axarr[1, 1].plot(t, v2, 'green', label='I_AMPAR')
        axarr[1, 1].set_title("I_AMPAR (pA)");
        axarr[1, 1].set_ylim(bottom=min(np.amin(v1),np.amin(v2))-0.5);
        axarr[1, 1].set_ylim(top=max(np.amax(v1),np.amax(v2))+0.5);
        axarr[1, 1].set_xlim(left=timeStart, right=timeEnd)
    # plt.legend(bbox_to_anchor=(1,1), loc=2)
    plt.show()

def simpleSpine():
    folder='out2_May27_rest'
    if (1):# just 2 neurons
        f, axarr = plt.subplots(3, sharex=True)
        mpl.rcParams['lines.linewidth'] = 2
        t, v1 = np.loadtxt(folder+'/somaCa.dat0',unpack=True,skiprows=1,
                                usecols=(0,1))

        axarr[0].plot(t, v1, 'green', label='spinehead')
        axarr[0].legend()
        t, v1 = np.loadtxt(folder + '/somaV.dat0',unpack=True,skiprows=1,
                                usecols=(0,1))
        axarr[1].plot(t, v1, 'red', label='spinehead')
        minVm = (np.amin(v1))
        maxVm = (np.amax(v1))
        print(minVm, maxVm)
        axarr[1].set_ylim(bottom=minVm - 0.5)
        axarr[1].set_ylim(top=maxVm + 0.5)
    plt.show()

def plot_soma():
    #more comprehensive deal with multiple processes I/O
    # with spines
    #folder='./out2_May26'
    #kfolder='./out2_May27_trigger_soma'
    #folder='./out2_May27_fullspine'
    folder='./out2_May27_rest'
    timeStart = .0
    timeEnd = -1.0
    #timeEnd = 50.0
    #timeStart = 270.0
    #timeEnd = 380.0
    somaMPIprocess=0
    middledenMPIprocess=1
    distaldenMPIprocess=1
    thinSpineMPIprocess=1
    perisomaticApicalDenMPIprocess=0
    perisomaticBasalDenMPIprocess=0
    axonAISMPIprocess = 0
    somaColor = 'blue'
    middledenColor= 'red'
    distaldenColor= 'green'
    perisomaticApicalDenColor= 'orange'
    perisomaticBasalDenColor= 'violet'
    axonAISColor = 'black'
    spineColor = 'magenta'
    preNeuronColor = 'brown'
    f, axarr = plt.subplots(3, 2)
    mpl.rcParams['lines.linewidth'] = 2
    t, v0 = np.loadtxt(folder + '/somaCa.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    idxStart = next(x[0] for x in enumerate(t) if x[1] >= timeStart)
    if (timeEnd == -1.0):
      idxEnd = len(t)-1
      timeEnd = t[idxEnd]
    else:
      idxEnd = next(x[0] for x in enumerate(t) if x[1] >= timeEnd)

    minCa = np.amin(v0)
    maxCa = np.amax(v0)
    # f, axarr = plt.subplots(3, sharex=True)
    axarr[0, 0].plot(t, v0, somaColor, label='soma')
    # axarr[0, 0].plot(t, v1, 'red', label='bouton')
    # axarr[0, 0].plot(t, v2, 'green', label='spinehead')

    #if (os.path.isfile(folder+'/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/perisomaticApicalDenCa.dat'+str(perisomaticApicalDenMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[0, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
    #    axarr[0, 0].legend()
    #    minCa = min(np.amin(v1), minCa)
    #    maxCa = max(np.amax(v1), maxCa)

    #if (os.path.isfile(folder+'/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/perisomaticBasalDenCa.dat'+str(perisomaticBasalDenMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[0, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
    #    axarr[0, 0].legend()
    #    minCa = min(np.amin(v1), minCa)
    #    maxCa = max(np.amax(v1), maxCa)

    #if (os.path.isfile(folder+'/distaldendriticCa.dat'+str(distaldenMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/distaldendriticCa.dat'+str(distaldenMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[0, 0].plot(t, v1, distaldenColor, label='distal-den')
    #    axarr[0, 0].legend()
    #    minCa = min(np.amin(v1), minCa)
    #    maxCa = max(np.amax(v1), maxCa)

    #if (os.path.isfile(folder+'/middledendriticCa.dat'+str(middledenMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/middledendriticCa.dat'+str(middledenMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[0, 0].plot(t, v1, middledenColor, label='middle-den')
    #    axarr[0, 0].legend()
    #    minCa = min(np.amin(v1), minCa)
    #    maxCa = max(np.amax(v1), maxCa)

    #if (os.path.isfile(folder+'/axonAISCa.dat'+str(axonAISMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/axonAISCa.dat'+str(axonAISMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[0, 0].plot(t, v1, axonAISColor, label='AIS')
    #    axarr[0, 0].legend()
    #    minCa = min(np.amin(v1), minCa)
    #    maxCa = max(np.amax(v1), maxCa)

    axarr[0, 0].set_ylim(bottom=minCa - 0.5)
    axarr[0, 0].set_ylim(top=maxCa + 0.5)
    axarr[0, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[0, 0].set_title("[Ca2+] (uM)")
    # axarr[0].set_xlim(left=0);
    # axarr[0].set_xlim(right=60);
    t, v0 = np.loadtxt(folder + '/somaV.dat'+str(somaMPIprocess),
                                    unpack=True, skiprows=1,
                                    usecols=(0, 1))
    axarr[1, 0].plot(t, v0, somaColor, label='soma')
    # axarr[1, 0].plot(t, v1, 'red', label='bouton')
    # axarr[1, 0].plot(t, v2, 'green', label='spinehead')
    minVm = (np.amin(v0[idxStart:idxEnd]))
    maxVm = (np.amax(v0[idxStart:idxEnd]))
    print(idxStart, idxEnd)
    print(t[idxStart], t[idxEnd])
    print(maxVm)
    axarr[1, 0].set_ylim(bottom=minVm - 0.5)
    axarr[1, 0].set_ylim(top=maxVm + 0.5)
    axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)
    axarr[1, 0].set_title("Vm (mV)")
    #axarr[2, 0].plot(t, v3, 'red', label='bouton')
    #axarr[2, 0].plot(t, v4, 'green', label='spinehead')
    #axarr[2, 0].set_title("Vm (mV)")
    # axarr[2, 0].set_ylim(bottom=min(np.amin(v3),np.amin(v4))-0.5);
    # axarr[2, 0].set_ylim(top=max(np.amax(v3),np.amax(v4))+0.5);

    #if (os.path.isfile(folder+'/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/perisomaticApicalDenV.dat'+str(perisomaticApicalDenMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[1, 0].plot(t, v1, perisomaticApicalDenColor, label='perisomatic-ApicalDen')
    #    #axarr[1, 0].legend()
    #    minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
    #    maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    #if (os.path.isfile(folder+'/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/perisomaticBasalDenV.dat'+str(perisomaticBasalDenMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[1, 0].plot(t, v1, perisomaticBasalDenColor, label='perisomatic-BasalDen')
    #    #axarr[1, 0].legend()
    #    minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
    #    maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    #if (os.path.isfile(folder+'/axonAISV.dat'+str(axonAISMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/axonAISV.dat'+str(axonAISMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[1, 0].plot(t, v1, axonAISColor, label='AIS')
    #    minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
    #    maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)
    #if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[1, 0].plot(t, v1, middledenColor, label='middle-den')
    #    axarr[1, 0].legend()
    #    minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
    #    maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)
    #axarr[1, 0].legend()
    #axarr[1, 0].set_ylim(bottom=minVm-0.5);
    #axarr[1, 0].set_ylim(top=maxVm+0.5);
    #axarr[1, 0].set_xlim(left=timeStart, right=timeEnd)


    if (os.path.isfile(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/distaldendriticV.dat'+str(distaldenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, distaldenColor, label='distal-den')
    axarr[2, 0].legend()
    if (os.path.isfile(folder + '/middledendriticV.dat'+str(middledenMPIprocess))):
        t, v1 = np.loadtxt(folder + '/middledendriticV.dat'+str(middledenMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 0].plot(t, v1, middledenColor, label='middle-den')
        axarr[2, 0].legend()
        minVm = min(np.amin(v1[idxStart:idxEnd]), minVm)
        maxVm = max(np.amax(v1[idxStart:idxEnd]), maxVm)

    #if (os.path.isfile(folder + '/spineV.dat'+str(thinSpineMPIprocess))):
    #    #t, v1 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
    #    #                unpack=True, skiprows=1,
    #    #                usecols=(0, 1))
    #    t, v1,v2 = np.loadtxt(folder + '/spineV.dat'+str(thinSpineMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1,2))
    #    axarr[2, 0].plot(t, v1, spineColor, label='head-V')
    #    #jaxarr[2, 0].plot(t, v2, spineColor, linestyle='dashdot', label='neck-V')
    #    axarr[2, 0].plot(t, v2, 'black', linestyle='dashdot', label='neck-V')
    #axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    #axarr[2, 0].legend()

    #if (os.path.isfile(folder + '/proximalSpineV.dat'+str(thinSpineMPIprocess))):
    #    t, v1,v2 = np.loadtxt(folder + '/proximalSpineV.dat'+str(thinSpineMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1, 2))
    #    #axarr[2, 0].plot(t, v1, 'black', label='-50 um')
    #    axarr[2, 0].plot(t, v1, 'black', label='0 um')
    #    #axarr[2, 0].plot(t, v2, 'blue', label='-25 um')
    #    axarr[2, 0].plot(t, v2, 'blue', label='-10 um')

    #if (os.path.isfile(folder + '/presynapticSomaVm.dat'+str(thinSpineMPIprocess))):
    #    t, v1 = np.loadtxt(folder + '/presynapticSomaVm.dat'+str(thinSpineMPIprocess),
    #                    unpack=True, skiprows=1,
    #                    usecols=(0, 1))
    #    axarr[2, 0].plot(t, v1, preNeuronColor, label='presyn-Soma-V')
    #axarr[2, 0].set_xlim(left=timeStart, right=timeEnd)
    #axarr[2, 0].legend()
    plt.show()

def plot_SpienCurrent():
    spinehead_area = 0.20 # um^2
    if (os.path.isfile(folder+'/spineNMDAR.dat'+str(thinSpineMPIprocess))):
        filename = folder + '/spineNMDAR.dat'+str(thinSpineMPIprocess)
        with open(filename) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols == 3):
          t, v1, v2 = np.loadtxt(filename,
                               unpack=True,skiprows=1,
                               usecols=(0,1,2))
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR')
          axarr[0, 1].plot(t, v2, 'green', label='I_NMDAR')
        elif (numcols ==2):
          t, v1 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[0, 1].plot(t, v1, 'red', label='I_NMDAR (pA)')
        axarr[0, 1].set_xlim(left=timeStart, right=timeEnd)
        axarr[0, 1].set_title("I_NMDAR (pA)");
    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        t, v1 = np.loadtxt(folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess),
                               unpack=True,skiprows=1,
                                usecols=(0,1))
        #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
        #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
        #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
        v1 = v1 * spinehead_area
        axarr[0, 1].plot(t, v1, 'green', linestyle='--', label='I_AMPAR (pA)')
    axarr[0,1].legend()

    if (os.path.isfile(folder+'/spineAMPAR.dat'+str(thinSpineMPIprocess))):
        filename = folder + '/spineAMPAR.dat'+str(thinSpineMPIprocess)
        with open(filename) as f:
          reader = csv.reader(f, delimiter='\t')
          reader.next()  # skip first line
          validrow = next(reader)
          numcols = len(validrow)

        if (numcols == 3):
          t, v1, v2 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                  usecols=(0,1,2))
          #axarr[2, 1].plot(t, v1, 'red', label='I_AMPAR')
          #axarr[2, 1].plot(t, v2, 'green', label='I_AMPAR')
          #axarr[2, 1].set_title("I_AMPAR (pA/um^2)");
          v1 = v1 * spinehead_area
          v2 = v2 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          axarr[1, 1].plot(t, v2, 'green', label='I_AMPAR')
          minCurrent = min(np.amin(v1[idxStart:idxEnd]),np.amin(v2[idxStart:idxEnd]))
          maxCurrent = max(np.amax(v1[idxStart:idxEnd]),np.amax(v2[idxStart:idxEnd]))
        elif (numcols == 2):
          t, v1 = np.loadtxt(filename,
                                unpack=True,skiprows=1,
                                  usecols=(0,1))
          v1 = v1 * spinehead_area
          axarr[1, 1].plot(t, v1, 'red', label='I_AMPAR')
          minCurrent = (np.amin(v1[idxStart:idxEnd]))
          maxCurrent = (np.amax(v1[idxStart:idxEnd]))
        axarr[1, 1].set_title("I_AMPAR (pA)");
        axarr[1, 1].set_ylim(bottom=minCurrent-0.5);
        axarr[1, 1].set_ylim(top=maxCurrent+0.5);
        axarr[1, 1].set_xlim(left=timeStart, right=timeEnd)
    # plt.legend(bbox_to_anchor=(1,1), loc=2)

    if (os.path.isfile(folder + '/spineCa.dat'+str(thinSpineMPIprocess))):
        t, v1 = np.loadtxt(folder + '/spineCa.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1))
        axarr[2, 1].plot(t, v1, spineColor, label='spine-Ca')
    axarr[2, 1].set_xlim(left=timeStart, right=timeEnd)
    axarr[2, 1].legend()
    if (os.path.isfile(folder + '/proximalSpineCa.dat'+str(thinSpineMPIprocess))):
        t, v1,v2 = np.loadtxt(folder + '/proximalSpineCa.dat'+str(thinSpineMPIprocess),
                        unpack=True, skiprows=1,
                        usecols=(0, 1, 2))
        #axarr[2, 1].plot(t, v1, 'black', label='-50 um')
        #axarr[2, 1].plot(t, v2, 'blue', label='-25 um')
        axarr[2, 1].plot(t, v1, 'black', label='0 um')
        axarr[2, 1].plot(t, v2, 'blue', label='-10 um')
    plt.show()

if (__name__ == "__main__"):
    case = 9
    if (case == 0):
        plot_soma()

    elif (case == 100):
        plot_case0()
    elif (case == 1):
        plot_case1()
    elif (case == 2):
        plot_case2()

    elif (case == 3):
        # no neuron, only bouton + spinehead
        # NOTE: use =4 is better
        plot_case3()
    elif (case == 4):
        # no neuron, only bouton + spinehead
        # with NMDAR current recording
        plot_case4()

    elif (case == 5):
        # 1 neuron, only bouton + spinehead
        # with NMDAR current recording
        plot_case5()
    elif (case == 6):
        #deal with MPIprocess
        plot_case6()
    elif (case == 7):
        #deal with MPIprocess (neuron + spines)
        plot_case7()
    elif (case == 8):
        #deal with MPIprocess (neuron + spines)
        # here we fix to 24 MPI processes
        plot_case8()
    elif (case == 9):
        #deal with MPIprocess (neuron + spines)
        # here we fix to 24 MPI processes
        #plot_case9()
        plot_case9_adv()
    elif (case == 10):
        simpleSpine()
