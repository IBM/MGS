#!/bin/bash
ARG1=$1
shift
ARG2=$1
shift
DFILE=${ARG1}/obj/${ARG2}.d
CFILE=${ARG1}/src/${ARG2}.C
OFILE=${ARG1}/obj/${ARG2}.o
DEPEND=`g++ -MM -MG $* ${CFILE} | cut -d: -f2 | sed -e 's/[\] *//g' `
echo ${DFILE}: ${DEPEND}
echo "	scripts/depend.sh ${ARG1} ${ARG2} $* > ${DFILE}"
echo
echo ${OFILE}: ${DEPEND}
echo '	$(CC) -c $(CFLAGS) $(OBJECTONLYFLAGS) $< -o $@'
echo
