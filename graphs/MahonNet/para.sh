#!/bin/bash

# Script to manage local parameter searches.

##########################
# ### Parameters ###
##########################
numMPI=1
numTHR=1
seed=85939
changeFile="MahonNet.gsl"

#variables=`echo $(seq 0.04 0.04 0.8)`
#variables=`echo $(seq 0.2 0.2 0.2)`
#variables=`echo $(seq 4.51 0.1 8.51)`
#variables=`echo $(seq 0.0005 0.001 1)`
#variables=`echo $(seq 0.005 0.005 0.005)`
#variables=`echo $(seq 0.03 0.03 1)`
variables=`echo $(seq 0.0005 0.0005 1)`

cb=0.2; #$(expr "scale = 5; 0.15" | bc | sed 's/^\./0./' );
#var1=0.0045;
synb=0.095;
deltat=0.2;
size=4;
tscale=1.0;
noiselev=0.0;
inpbase=0.04381;
inprange=0.01;
tlen=10000;

for var1 in $variables; do
cp $changeFile $changeFile.run;
sed -i 's/DELTATXX/'"$deltat"'/g' ./$changeFile.run
sed -i 's/CONPROBXX/'"$cb"'/g' ./$changeFile.run
sed -i 's/SIZEXX/'"$size"'/g' ./$changeFile.run
sed -i 's/TSCALEXX/'"$tscale"'/g' ./$changeFile.run
sed -i 's/NLEVXX/'"$noiselev"'/g' ./$changeFile.run
sed -i 's/TLENXX/'"$tlen"'/g' ./$changeFile.run
#f2=$(expr "scale = 5; 0.2*$var1/($cb)" | bc | sed 's/^\./0./' );
#f3=$(expr "scale = 5; 0.2*($var1+0.06)/($cb)" | bc | sed 's/^\./0./' );
f2=$(expr "scale = 5; $var1" | bc | sed 's/^\./0./' );
f3=$(expr "scale = 5; $var1+0.001" | bc | sed 's/^\./0./' );
#f2=$(expr "scale = 5; 0" | bc | sed 's/^\./0./' );
#f3=$(expr "scale = 5; 0" | bc | sed 's/^\./0./' );
sl=$(expr "scale = 5; $synb" | bc | sed 's/^\./0./' );
sh=$(expr "scale = 5; $synb+0.01" | bc | sed 's/^\./0./' );
#f2=0.12;
#f3=0.18;
sed -i 's/SYNBLOXX/'"$sl"'/g' ./$changeFile.run
sed -i 's/SYNBHIXX/'"$sh"'/g' ./$changeFile.run 
sed -i 's/WEILOXX/'"$f2"'/g' ./$changeFile.run
sed -i 's/WEIHIXX/'"$f3"'/g' ./$changeFile.run 
i2=$(expr "scale = 5; $inpbase" | bc | sed 's/^\./0./' );
i3=$(expr "scale = 5; $i2+$inprange" | bc | sed 's/^\./0./' );
#i2=2.43;
#i3=3.00;
sed -i 's/INPLOXX/'"$i2"'/g' ./$changeFile.run
sed -i 's/INPHIXX/'"$i3"'/g' ./$changeFile.run
mpiexec -n $numMPI ../../gsl/bin/gslparser -t $numTHR -f $changeFile.run -s $seed
rm $changeFile.run;
seed=$((seed+1))
done
