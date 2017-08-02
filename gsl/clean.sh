#!/bin/bash
#
# Script to clean previous compile efforts

find . -name "*\.o" -exec /bin/rm {} \;
find . -name "*\.so" -exec /bin/rm {} \;
find . -name "*\.d" -exec /bin/rm {} \;
find . -name "*\.ld" -exec /bin/rm {} \;
find . -name "*\.yd" -exec /bin/rm {} \;
find . -name "*\.ad" -exec /bin/rm {} \;
find . -name "*\.def" -exec /bin/rm {} \;
find . -name "*\.undef" -exec /bin/rm {} \;
find . -type d -name "generated" -exec sh scripts/cleandir.sh {} \;
rm lib/liblens.a
rm framework/factories/include/LensRootConfig.h
rm dx/EdgeSetSubscriberSocket
rm dx/NodeSetSubscriberSocket
rm bin/gslparser
rm bin/createDF
rm so/Dependfile
