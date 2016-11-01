from __future__ import print_function
import os
import logging
from parser import Parser
from utilities import warning
import csv
# import pandas as pd;

__author__ = "Hoang Trong Minh Tuan"
__copyright__ = "Copyright 2015, IBM"
__credits__ = []  # list of people who file pugs
__version__ = "0.1"
__maintainer__ = "Hoang Trong Minh Tuan"
__email__ = "tmhoangt@us.ibm.com"
__status__ = "Prototype"  # (Prototype, Development, Production)

# http://stackoverflow.com/questions/1523427/what-is-the-common-header-format-of-python-files


if __name__ == "__main__":
    args = Parser.parse_command()

    swc_file = args.swc
    assert(swc_file is not None)
    if (not swc_file.endswith(".swc")):
        logging.debug("file extension unexpected")
        warning("Please use .swc file")
        Parser.print_help()
        exit()

    if (not os.path.isfile(swc_file)):
        warning(swc_file + " not found")
        exit()

    csv.register_dialect('swc', delimiter=" ", skipinitialspace=True)
    swc_fieldnames = ['row_index', 'branchtype',
                      'X', 'Y', 'Z', 'radius',
                      'parent_row_index']
    # The output file will be <oldname>_updated.swc
    swc_outputfile = os.path.splitext(swc_file)[0] + '_updated.swc'
    logging.debug(swc_outputfile)
    # writer = csv.DictWriter(swc_outputfile,
    #                        fieldnames=swc_fieldnames, dialect='swc')
    with open(swc_file, 'r') as fp:
        # dialect = csv.Sniffer().sniff(fp.read(2048), delimiters=" ")
        # fp.seek(0)
        reader = csv.DictReader(filter(lambda row: row[0] != '#', fp),
                                dialect='swc',
                                fieldnames=swc_fieldnames)
        i = 0
        coord_x = coord_y = coord_z = 0.0
        radius = 0.0
        for row in reader:
            if (row['branchtype'] == 1):
                coord_x = coord_x + row['X']
                coord_y = coord_y + row['Y']
                corrd_z = coord_z + row['Z']
                radius = max(radius, row['radius'])


            # print(row)
            i = i+1
            if (i == 2):
                # exit()
                pass
        print(coord_x, coord_y, coord_z, radius)
