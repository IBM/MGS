cd nts
find . -exec grep -l "All rights reserved" {} \; > cr.out
awk '{print "sed -e '"'"'s/2005-2012/2005-2014/g'"'"' " "../nts_copyright/" $1 " > " $1}' cr.out > cr2.out
source cr2.out
rm -f cr.out cr2.out
cd ..
