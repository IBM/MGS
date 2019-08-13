#!/bin/bash

# Script to manage local parameter searches.

##########################
# ### Parameters ###
##########################
numMPI=1
numTHR=4
seed=85939
changeFile="MahonNet.gsl"

#variables=`echo $(seq 0.04 0.04 0.8)`
#variables=`echo $(seq 0.2 0.2 0.2)`
#variables=`echo $(seq 4.51 0.1 8.51)`
#variables=`echo $(seq 0.0005 0.001 1)`
#variables=`echo $(seq 0.005 0.005 0.005)`
#variables=`echo $(seq 0.03 0.03 1)`
#variables=`echo $(seq 0.0005 0.0005 1)`
#variables=`echo $(seq 0.0005 1 0.0005)`
#variables=`echo $(seq 0.0005 0.0005 0.05)`
#variables=`echo $(seq 0.005 0.0025 0.5)` # inhibition range 
variables=`echo $(seq 0 0.005 0.02)`
#variables=(0.041) #0.033) #,0.041)

cb=0.2; #$(expr "scale = 5; 0.15" | bc | sed 's/^\./0./' );
#var1=0.0045;
synb=0.095;
deltat=0.2;
size=5 #0; # size*size cell network 
tscale=1.0;
noiselev=0.0;

# Gex - random variable drawn uniformly from [inpbase, inpbase + inprange]
inpbase=0.04381;
#inprange=0.05;
var_ge=(0.03 0.05 0.08) # excitation inprange = 0.002*g_e 
#var_ge=(0.08) #0.1) #, 0.08)

tlen=100;
#tlen=3;

run_serial () {
    local FOLDERSUFFIX=$1
    local ge=$2
    for var1 in $variables; do
      #local FOLDERSUFFIX="ge"$ge"_gi"$var1

      local GSL_FILE=${changeFile}_$FOLDERSUFFIX.run
      local DATA_FOLDER="data_$FOLDERSUFFIX"
      echo "ge = $ge, gi = "$var1  $GSL_FILE

      cp $changeFile $GSL_FILE
      sed -i 's/DATA_FOLDER/'"$DATA_FOLDER"'/g' ./$GSL_FILE
      sed -i 's/DELTATXX/'"$deltat"'/g' ./$GSL_FILE
      sed -i 's/CONPROBXX/'"$cb"'/g' ./$GSL_FILE
      sed -i 's/SIZEXX/'"$size"'/g' ./$GSL_FILE
      sed -i 's/TSCALEXX/'"$tscale"'/g' ./$GSL_FILE
      sed -i 's/NLEVXX/'"$noiselev"'/g' ./$GSL_FILE
      sed -i 's/TLENXX/'"$tlen"'/g' ./$GSL_FILE
      #f2=$(expr "scale = 5; 0.2*$var1/($cb)" | bc | sed 's/^\./0./' );
      #f3=$(expr "scale = 5; 0.2*($var1+0.06)/($cb)" | bc | sed 's/^\./0./' );
      f2=$(expr "scale = 5; $var1" | bc | sed 's/^\./0./' ); # f2 = var1 = 0.001*g_i (inhibition g_i)
      f3=$(expr "scale = 5; $var1+0.001" | bc | sed 's/^\./0./' ); # f3 = f2 + 0.001 
      #f2=$(expr "scale = 5; 0" | bc | sed 's/^\./0./' );
      #f3=$(expr "scale = 5; 0" | bc | sed 's/^\./0./' );
      sl=$(expr "scale = 5; $synb" | bc | sed 's/^\./0./' );
      sh=$(expr "scale = 5; $synb+0.01" | bc | sed 's/^\./0./' );
      #f2=0.12;
      #f3=0.18;
      sed -i 's/SYNBLOXX/'"$sl"'/g' ./$GSL_FILE
      sed -i 's/SYNBHIXX/'"$sh"'/g' ./$GSL_FILE 
      sed -i 's/WEILOXX/'"$f2"'/g' ./$GSL_FILE
      sed -i 's/WEIHIXX/'"$f3"'/g' ./$GSL_FILE 
      i2=$(expr "scale = 5; $inpbase" | bc | sed 's/^\./0./' );
      i3=$(expr "scale = 5; $i2+$ge" | bc | sed 's/^\./0./' );
      #i2=2.43;
      #i3=3.00;
      sed -i 's/INPLOXX/'"$i2"'/g' ./$GSL_FILE
      sed -i 's/INPHIXX/'"$i3"'/g' ./$GSL_FILE
      mpiexec -n $numMPI ../../gsl/bin/gslparser -t $numTHR -f $GSL_FILE -s $seed > /dev/null
      #rm $GSL_FILE;
      seed=$((seed+1))
    done;
    #echo append to output file
    echo -e "\n" >> "$DATA_FOLDER/Spike_Output.dat"
}


#for ge 
for ge in ${var_ge[@]}; do 
  #echo "ge="$ge 
  FOLDERSUFFIX="ge"$ge
  run_serial $FOLDERSUFFIX $ge &
  sleep 1
done; # ge

