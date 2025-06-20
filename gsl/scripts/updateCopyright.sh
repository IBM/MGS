#!/bin/bash

echo 'Filetypes of files containing the phrase "Copyright":'
FT="$(ack Copyright ../../ | awk '{ print $1 }' | sed 's/.*\.//' | sed 's/:.*//' | sort | uniq)"
echo "${FT}"

FT2="$(echo "${FT}" | sed ':a;N;$!ba;s/\n/\\\|/g')"
#echo "${FT2}"

find ../../ -regex ".*\.\($FT2\)$" ! -name "globalSub.sh" \
     -exec sed -i.bak 's/08\-23\-2011\-2/07\-18\-2025/g' {} \;
find ../../ -regex ".*\.\($FT2\)$" ! -name "globalSub.sh" \
     -exec sed -i.bak 's/11\-19\-2015/07\-18\-2025/g' {} \;
find ../../ -regex ".*\.\($FT2\)$" ! -name "globalSub.sh" \
     -exec sed -i.bak 's/2005\-2015/2005\-2025/g' {} \;
find ../../ -regex ".*\.\($FT2\)$" ! -name "globalSub.sh" \
     -exec sed -i.bak 's/2005\-2014/2005\-2025/g' {} \;
find ../../ -regex ".*\.\($FT2\)$" ! -name "globalSub.sh" \
     -exec sed -i.bak 's/Copyright 2016-2017/Copyright 2016\-2025/g' {} \;
find ../../ -regex ".*\.\($FT2\)$" ! -name "globalSub.sh" \
     -exec sed -i.bak 's/IBM, @2016-2017/IBM, @2016\-2025/g' {} \;
find ../../ -regex ".*\.\($FT2\)$" ! -name "globalSub.sh" \
     -exec sed -i.bak 's/IBM \- 2016/IBM \- 2016\-2025/g' {} \;
