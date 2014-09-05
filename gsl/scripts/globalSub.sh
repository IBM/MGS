#!/bin/bash
grep -l $1 $3 | grep -v $4 &> tmp.sh
awk '{print "sed '"'"'s/'$1'/'$2'/g'"'"' "  $1 " > tmp.mdl; mv tmp.mdl " $1;}' tmp.sh &> tmp2.sh
rm -f tmp.sh
source ./tmp2.sh
rm -f tmp2.sh