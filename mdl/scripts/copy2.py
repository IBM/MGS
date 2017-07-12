#!/usr/bin/python
#TODO: used?

import sys, os, string

if (len(sys.argv) < 2 ):
    print "Usage: codeMod.py <name>"
    sys.exit(-1)


try:
    f = open("scripts/C_.h", 'r')
    hLines = f.readlines()
    f.close()
    for i in xrange(len(hLines)):
        hLines[i] = hLines[i].replace("C_X1", sys.argv[1])
except IOError:
    print "C_.h is not present, find it."
    sys.exit(-1)
    pass

try:
    f = open("scripts/C_.C", 'r')
    cLines = f.readlines()
    f.close()
    for i in xrange(len(cLines)):
        cLines[i] = cLines[i].replace("C_X1", sys.argv[1])
except IOError:
    print "C_.C is not present, find it."
    sys.exit(-1)
    pass

hName = "include/" + sys.argv[1] + ".h"
f = open(hName, "w")
f.writelines(hLines)
f.close()

cName = "src/" + sys.argv[1] + ".C"
f = open(cName, "w")
f.writelines(cLines)
f.close()

