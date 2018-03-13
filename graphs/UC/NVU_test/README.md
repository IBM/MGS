make sure you generate the tissue.txt file
e.g.: 2x2 grid; with size each grid element is 200 micrometer
python generateNeurons.py neurons/scptmsn.swc 200 2 2

then copy Topology.h and set it value _X_, _Y_, _Z_  so that the product is equal to the number of processes of the mpirun, e.g. 8

finally run

mpirun -n 8  $GSLPARSER NVUNTS.gsl


TROUBLESHOOT:
LD_LIBRARY_PATH has link to $SUITESPARSE/lib
