#!/bin/bash
#NOTE:
# 
if [ "$#" == "0" ]; then
  echo "./doPlot.sh <location> [protocol-index] <extension>  <morph>"
  echo "DEFAULT:"
  echo "    protocol-index    = 7"
fi
protocolIndex=7
if [ "$#" -ge "2" ]; then
  protocolIndex=$2
fi
morph=$4

python plot_currents.py protocol $morph Tuan $protocolIndex -extension=$3 -locData=$1
#python plot_currents.py protocol msnNodomain Tuan $protocolIndex -extension=$3 -locData=$1
