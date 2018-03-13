#!/bin/bash
cp Topology.h .Topology_backup.h

echo "#define _X_ 2" > Topology.h
echo "#define _Y_ 1" >> Topology.h
echo "#define _Z_ 1" >> Topology.h
if [ "$1" = "-n" ]; then
mpiexec -n 1  $GSLPARSER model_tissue.gsl
else
#mpiexec -n 1 xterm -e cgdb -x gdb.txt ../../gsl/bin/gslparser
mpiexec -n 2 xterm -geometry 180x80 -e cgdb -x gdb.txt  $GSLPARSER
fi
cp .Topology_backup.h   Topology.h 
