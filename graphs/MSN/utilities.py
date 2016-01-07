from __future__ import print_function
import sys

__author__ = "Hoang Trong Minh Tuan"
__copyright__ = "Copyright 2015, IBM"
__credits__ = []  # list of people who file pugs
__version__ = "0.1"
__maintainer__ = "Hoang Trong Minh Tuan"
__email__ = "tmhoangt@us.ibm.com"
__status__ = "Prototype"  # (Prototype, Development, Production)


def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)
