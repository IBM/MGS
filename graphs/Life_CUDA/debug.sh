#!/bin/bash
#mpiexec -n 1 xterm -geometry 180x80 -e cgdb -x gdb.txt  ${GSLPARSER}_cuda
#cgdb -x gdb.txt  ${GSLPARSER}_cuda
cuda-gdb -x gdb.txt  ${GSLPARSER}_cuda
#mpiexec -n 1 xterm -geometry 180x80 -e cuda-gdb -x gdb.txt  ${GSLPARSER}_cuda
#mpiexec -n 1 cuda-gdb -x gdb.txt  ${GSLPARSER}_cuda
#mpiexec -n 1 xterm -geometry 180x80 -e cgdb -x gdb.txt  ${GSLPARSER}_cpu
