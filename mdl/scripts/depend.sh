#!/bin/bash
#TODO: used?
ARG1=$1
shift
DFILE=obj/${ARG1}.d
CFILE=src/${ARG1}.C
OFILE=obj/${ARG1}.o
DEPEND=`g++ -MM -MG $* ${CFILE} | grep -v "#" | cut -d: -f2 | sed -e 's/[\] *//g' `
echo ${DFILE}: ${DEPEND}
echo "	scripts/depend.sh ${ARG1} $* > ${DFILE}"
echo
echo ${OFILE}: ${DEPEND}
echo '	$(CC) -c $(CFLAGS) $(OBJECTONLYFLAGS) $< -o $@'
echo
