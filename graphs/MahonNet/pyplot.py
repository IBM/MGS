#!/usr/bin/env python
import sys, io, os, numpy as np, matplotlib.pyplot as plt

HELPSTRING='PYPLOT()\n\n\
NAME\n\tpyplot -- Graph input data.\n\n\
SYNOPSIS\n\tpyplot [-d] [path/to/inputfile] ([-r] [path/to/inputfile] [-x] index [-y] index [-o] [outputfile] [-c] "COMMAND ...")\n\n\
DESCRIPTION\n\tEvery column is a set of values correlated by row. Both columns and rows are zero-indexed. Graphs are created where command is run.\n\n\
\tThe following options are available:\n\n\
\t-c\tWhat to graph. The following options are available:\n\n\
\t\t\n\
\t-d\tData file containing domain (and range).\n\n\
\t-h\tDiplay the help message for the pyplot command.\n\n\
\t-l\tDelimeter of columns. Default: whitespace.\n\n\
\t-m\t(Optional) Specify the x-axis label.\n\n\
\t-n\t(Optional) Specify the y-axis label.\n\n\
\t-o\t(Optional) Specify the output filename to be used.\n\n\
\t-r\t(Optional) Data file containing range.\n\n\
\t-x\tIndex of x column in input file(s). Default: 0\n\n\
\t-y\tIndex of y column in input file(s). Default: 1\n\
'

#formats:
SIMSPLIT="\n\n"
DAT={'names':('simulation','data'), 'formats':(int, np.ndarray)}

def parse(DELIM=None,x=0,y=1):
    args = ' '.join(sys.argv[1:]).split()
    infile=np.empty( (0,), dtype=DAT)
    outfile="graph.png"
    simulations=0
    data=[]
    rang=[]
    xlabel='Domain'
    ylabel='Range'

    if '-h' in args:
        #print HELPSTRING
        print(HELPSTRING)
        sys.exit(0)

    if args:
        try:
            while args:
                arg=args.pop(0)
                if arg == '-d':
                    data=io.open(args.pop(0), encoding="utf-8").read().split(SIMSPLIT)
                elif arg == '-r':
                    rang=io.open(args.pop(0), encoding="utf-8").read().split(SIMSPLIT)
                elif arg == '-l':
                    DELIM=args.pop(0)
                elif arg == '-m':
                    xlabel=args.pop(0)
                    while args and (args[0][0:1] != '-'):
                        xlabel+=' '+args.pop(0)
                elif arg == '-n':
                    ylabel=args.pop(0)
                    while args and (args[0][0:1] != '-'):
                        ylabel+=' '+args.pop(0)
                elif arg == '-x':
                    x=int(args.pop(0))
                elif arg == '-y':
                    y=int(args.pop(0))
        except:
            #print "Missing flag or argument.\n\n"+HELPSTRING
            print("Missing flag or argument.\n\n"+HELPSTRING)
            sys.exit(1)
        d=np.empty( (0,), dtype=DAT)
        r=np.empty( (0,), dtype=DAT)
        for sim in range(len(data)):
            if data[sim]:
                file=np.genfromtxt(io.StringIO(data[sim]))
                d=np.append(d, np.array((sim,file), dtype=DAT))
                simulations+=1
        r=d
        for sim in range(len(rang)):
            file=np.genfromtxt(io.StringIO(rang[sim]))
            r=np.append(d, np.array((sim,file), dtype=DAT))
        try:
            for sim in range(simulations):
                coordinates=np.column_stack([d['data'][sim][:,x], r['data'][sim][:,y]])
                pair=np.array((sim, coordinates), dtype=DAT)
                infile=np.append(infile, pair)
        except Exception as e:
            #print "There was an issue with the domain and/or range: "+str(e)+"."
            print("There was an issue with the domain and/or range: "+str(e)+".")
            sys.exit(1)
    else:
        #print HELPSTRING
        print(HELPSTRING)
        sys.exit(1)
    return infile, simulations, outfile, xlabel, ylabel

def graph(infile, simulations, outfile, d, r):
    for sim in range(simulations):
        plt.figure()
        plt.plot(infile['data'][sim][:,0],infile['data'][sim][:,1], 'ko')
        plt.xlabel(d)
        plt.ylabel(r)
        plt.savefig("s"+str(sim)+outfile)
        plt.close()

data, n, filename, d, r=parse()
graph(data,n,filename, d, r)
f=open('graph'+filename, "w+")
#print "started"
print("started")
f.close()
#print "done"
print("done")
