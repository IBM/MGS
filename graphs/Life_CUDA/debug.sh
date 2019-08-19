#!/bin/bash
#RESOLUTION=WIDTHxHEIGHT
RESOLUTION=180x180
#OTHER_ARGS="-fs 20"
#mpiexec -mca btl self -mca pml ucx -n 1 xterm -geometry 180x80 -e cgdb -x gdb.txt  ${GSLPARSER}_cuda
#mpiexec -mca btl self -mca pml ucx -n 1 xterm -geometry 180x80 -e cgdb -x gdb.txt  ../../gsl/bin/gslparser_cuda2
mpiexec -mca btl self -mca pml ucx -n 1 xterm -geometry 180x80 -e cgdb -x gdb.txt  ../../gsl/bin/gslparser_cuda2
#xterm -geometry $RESOLUTION $OTHER_ARGS -e cgdb -x gdb.txt  ${GSLPARSER}_cuda
#cgdb -x gdb.txt  ${GSLPARSER}_cuda
#cuda-gdb -x gdb.txt  ${GSLPARSER}_cuda
#mpiexec -n 1 xterm -geometry 180x80 -e cuda-gdb -x gdb.txt  ${GSLPARSER}_cuda
#mpiexec -n 1 cuda-gdb -x gdb.txt  ${GSLPARSER}_cuda
#mpiexec -n 1 xterm -geometry 180x80 -e cgdb -x gdb.txt  ${GSLPARSER}_cpu
