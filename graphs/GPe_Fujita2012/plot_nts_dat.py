import numpy as np
import matplotlib.pyplot as plt
import sys

def getopts(argv):
    opts = {}
    while argv:
        if argv[0][0] == '-':
            opts[argv[0]] = argv[1]
        argv = argv[1:]
    return opts

if __name__ == '__main__':
    print(sys.argv)
    from sys import argv
    opts = getopts(argv)
    if '-filename' in opts:
        filename = opts['-filename']
    else:
        print('-filename argument required')

    D = np.genfromtxt(filename,skip_header=1,names=True)
    for name in D.dtype.names[1:]:
        plt.plot(D[D.dtype.names[0]], D[name],label=name)

    plt.legend()
    plt.show()

