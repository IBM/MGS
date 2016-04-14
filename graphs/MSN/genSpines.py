import itertools
import pandas as pd
import numpy as np
import numpy.random as rnd
import pystache
import sys
# import re
from math import sqrt
from copy import deepcopy
import math
import random
import os
import errno

rnd.seed(seed=1)
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
cur_version = sys.version_info

"""
NOTE: branchOrder being store starts from zero
"""
class SomeClass(object):
    branchType = {"soma": 1, "axon": 2, "basal": 3, "apical": 4,
                  "AIS": 5, "tufted": 6,"bouton":7 }
    spineType = {"generic": 0, "thin": 2, "mush": 3}
    #boutonType = {"excitatory":4, "inhibitory":5}
    boutonType = {"excitatory":4, "inhibitory":1}
    boutonTypeOnSpine = {"thin":4, "mush":5} #as a function of spine-type
    numFields = 7
    # def __new__(cls, *args, **kwargs):
    #   pass

    def __init__(self, x):
        """

        Pass in the path and file name of .swc file
        @param x = .swc filename (including path)

        """
        # NOTE: we use this to separate the spines not too closed to each other
        #       which make the touch detection harder to do right
        self.rotation_indices = [0,1,2,3,4]
        self.rotation_angles = [
           math.radians(0),
           math.radians(144),
           math.radians(288),
           math.radians(72),
           math.radians(216)
        ]
        self.swc_filename = x
        self.parse_file()
        print("""
              Please perform the operations in the following order
              1. one of this
              self.genSpine_MSN_distance_based()
              self.genSpine_PyramidalL5()
              self.genSpine_MSN_branchorder_based()
              self.genSpine_MSN_at_branchpoint()
              2. assign a rotation index the spines (make sure to adjacent spines not overlap)
              self.rotateSpines()
              3. save to output file (default: spines.txt)
              self.saveSpines(filename)
              4. generate bouton/spine SWC files (default:./neurons/)
              self.genboutonspineSWCFiles_MSN(folder)


              """)

    def update_dist2branchPoint_and_branchOrder(self, ids_as_parent):
        """
        Update the dist2soma for all points
                (currently absorbed in the self.point_lookup so far)
                based on the  given index 'id' of a new branching point

        @param ids_as_parent
        @type list

        STRATEGY:
            check for all ids with parent_id in the list ids_as_parent
        """
        new_ids_as_parent = deepcopy(ids_as_parent)
        for idparent in ids_as_parent:
            #idparent = the id of the line that need to be update (due to
            #           it becomes a branchpoint)
            for id in self.line_ids:
                parent_id = self.point_lookup[id]['parent']
                if (parent_id == idparent):
                    # update current dist2branchPoint of  'id'
                    parent_dist2branchPoint = \
                        self.point_lookup[idparent]['dist2branchPoint']
                    self.point_lookup[id]['dist2branchPoint'] = \
                        self.find_distance(id, parent_id) + \
                        parent_dist2branchPoint
                    self.point_lookup[id]['branchOrder'] = \
                        self.point_lookup[parent_id]['branchOrder']

                    new_ids_as_parent.append(id)
            new_ids_as_parent.remove(idparent)
        if (new_ids_as_parent):
            self.update_dist2branchPoint_and_branchOrder(new_ids_as_parent)

    def find_distance(self, id1, id2):
        """
        find the Eucledian distance between two points
        based upon the indices in self.point_lookup

        @param id1
        @param id2

        """
        parent_info = self.point_lookup[id2]
        parent_x = float(parent_info['siteX'])
        parent_y = float(parent_info['siteY'])
        parent_z = float(parent_info['siteZ'])

        point_info = self.point_lookup[id1]
        x = float(point_info['siteX'])
        y = float(point_info['siteY'])
        z = float(point_info['siteZ'])

        distance = sqrt((x - parent_x)**2 +
                        (y - parent_y)**2 + (z - parent_z)**2)
        return distance

    def parse_file(self):
        """

        INPUT: pass in the swc file of neuron
        organize the file in the form
        a list of 'line'
        each 'line' is organized as a map with
        - key = line index,
        - value = a map with asssociated key for each column's value
           KEYS are
        'type' = swc branchType
        'siteX'
        'siteY'
        'siteZ'
        'siteR'
        'dist2soma'
        'dist2branchPoint'
        'branchOrder'
        'numChildren'

        """
        # lines = open(self.swc_filename).read().splitlines()
        try:
            # file object
            myfile = open(self.swc_filename, "r+")
            # or "a+", whatever you need
        except IOError:
            print "Could not open file! Please check !" + self.swc_filename

        # a list of strings, each string represent a line
        lines = myfile.read().splitlines()

        # a list of list of field-values
        # each line (as a list) now is broken down into fields
        split_lines = map(lambda x: x.strip().split(' '), lines)


        # a vector containing each line in the form of point_lookup
        self.line_ids = []
        # take the line index as the key and value is
        # another map to enable
        # access to individual field (column)
        self.point_lookup = {}

        maxDist2BranchPoint = 0.0
        maxDist2Soma = 0.0
        start = 0
        while True:
            # remove potential comment-line in .swc file
            tmp = split_lines[start]
            field1 = tmp[0].lstrip()
            if (field1[0] == '#'):
                start += 1
            else:
                break
            # first patch = check error
        for line in split_lines[start:]:
            if (len(line) != self.__class__.numFields):
                sys.exit("There is line " +
                         str(line) +
                         " not conform to the format with 7 fields")

            # second patch = add data first
        #for line in split_lines[start:]:
        #    if (len(line) < self.__class__.numFields):
        #        continue
        #    id = line[0]
        #    # TODO: add 1. distance to soma;
        #    #           2. distance to nearest proximal branching point;
        #    #           3. branchOrder
        #    #           4. numChildren
        #    dist2soma = 0.0
        #    dist2branchPoint = 0.0
        #    branchOrder = 0  # branch just stemming from soma
        #    numChildren = 0
        #    self.point_lookup[id] = {'type': line[1],
        #                             'siteX': line[2], 'siteY': line[3],
        #                             'siteZ': line[4],
        #                             'siteR': line[5], 'parent': line[6],
        #                             'dist2soma': dist2soma,
        #                             'dist2branchPoint': dist2branchPoint,
        #                             'branchOrder': branchOrder,
        #                             'numChildren': numChildren}
        #    self.line_ids.append(id)

        for line in split_lines[start:]:
            if (len(line) != self.__class__.numFields):
                continue
            id = line[0]
            dist2soma = 0.0
            dist2branchPoint = 0.0
            branchOrder = 0  # branch just stemming from soma
            numChildren = 0

            do_update = False
            ids_as_parent = []
            if (int(line[6]) != -1):
                parent_id = line[6]
                parent_info = self.point_lookup[parent_id]
                parent_x = float(parent_info['siteX'])
                parent_y = float(parent_info['siteY'])
                parent_z = float(parent_info['siteZ'])
                parent_dist2soma = float(parent_info['dist2soma'])
                x = float(line[2])
                y = float(line[3])
                z = float(line[4])
                distance = sqrt((x - parent_x)**2 + (y - parent_y)**2 +
                                (z - parent_z)**2)
                dist2soma = distance + parent_dist2soma
                assert dist2soma >= 0.0
                self.point_lookup[parent_id]["numChildren"] += 1
                if (self.point_lookup[parent_id]["numChildren"] == 2) and \
                        (int(parent_id) != 1):
                    # NOTE: only update if a point become a branching
                    #  if it already a branching point, ignore the update
                    self.point_lookup[parent_id]["dist2branchPoint"] = 0.0
                    self.point_lookup[parent_id]["branchOrder"] += 1
                    # update dist2branchPoint and branchOrder of all its children
                    ids_as_parent = [parent_id]
                    do_update = True

                dist2branchPoint = distance + \
                    self.point_lookup[parent_id]["dist2branchPoint"]
                branchOrder = self.point_lookup[parent_id]["branchOrder"]

            # TODO: add 1. distance to soma;
            #           2. distance to nearest proximal branching point;
            #           3. branchOrder
            #           4. numChildren
            assert dist2soma >= 0.0
            self.point_lookup[id] = {'type': line[1],
                                     'siteX': line[2], 'siteY': line[3],
                                     'siteZ': line[4],
                                     'siteR': line[5], 'parent': line[6],
                                     'dist2soma': dist2soma,
                                     'dist2branchPoint': dist2branchPoint,
                                     'branchOrder': branchOrder,
                                     'numChildren': numChildren}
            self.line_ids.append(id)
            if (dist2soma > maxDist2Soma):
                maxDist2Soma = dist2soma
                idOfMaxDist2Soma = id
            if (do_update):
                    self.update_dist2branchPoint_and_branchOrder(ids_as_parent)

        print("point with max-Dist2Soma is ", maxDist2Soma, " and id =", idOfMaxDist2Soma)
        # NOTE: keep this code for debug purpose
        """
        for x in range(len(self.point_lookup)):
            #if (self.point_lookup[str(x+1)]['dist2branchPoint'] == 0.0):
            print self.point_lookup[str(x+1)]['dist2branchPoint']
        """

    def __str__(self):
        print("id | parent_id | branchOrder | branchType | numChildren | dist2branchPoint")
        str = ""
        for id in self.line_ids:
            point_info = self.point_lookup[id]
            branchType = int(point_info['type'])
            parent_id  = int(point_info['parent'])
            dist2soma  = float(point_info['dist2soma'])
            dist2branchPoint = float(point_info['dist2branchPoint'])
            branchOrder      = int(point_info['branchOrder'])
            numChildren      = int(point_info['numChildren'])
            id = int(id)
            # print("%s, %d, %s, %d, %d", branchType, branchOrder, numChildren)
            #print(id, parent_id, branchType, branchOrder, numChildren)
            str += '%3d, %3d, %d, %2d, %d, %5.2f\n' % (id, parent_id,
                                                       branchOrder, branchType,
                                                       numChildren,
                                                       dist2branchPoint)
        return str

    def test(self):
        distance_slot = range(0,220,10) # NOTE: using 220: 22 elements --> 21 ranges
        # d1_slot = [0  ,  10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110, 120,
        #           130,140,150,160,170,180,190,200,210 ]
        incr_slot = [10] * (len(distance_slot)-1) # 21 increments (each with 10 micrometer)
        mean_spineoccurence_slot = [0.5, 0.5, 10, 15, 30, 40, 35, 30, 27, 25,  25, 22, 20,
                     18, 15, 13, 13, 12, 10, 9, 7 ] # every incr_slot[..] (micrometer)
        std_spineoccurence_slot  = [ 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]
        # 1spine occur every '...' )um
        freq_spineoccurence_slot = np.divide(incr_slot, mean_spineoccurence_slot)
        lambda_slot = []  # rate parameter for Poisson process (govern spine occurence)
        for x in freq_spineoccurence_slot:
            lambda_slot.append(1.0/x)

        # Strategy:
        # for each tree (dendritic)
        # traverse from soma to end
        #  at each segment - find the dist2soma and length
        #  check the index 'idx' it belong to in distance_slot
        #  get the rateParameter = lambda_slot[idx]
        #  generate the location of next-spine
        #     loc = self.nextTime(rateParameter)
        #  find the segment it belong to, put it there
        # Strategy:
        #

        print distance_slot

    def nextTime(self, rateParameter):
        """
        Generate the next time of occurence based on Poisson distribution
        """
        # which returns the time/location (offset from current time/position)
        # for the event to occur
        return -math.log(1.0 - random.random())/ rateParameter

    def nextLocation(self, rateParameter):
        """
        Generate the next location of occurence based on Poisson distribution
        """
        # which returns the time/location (offset from current time/position)
        # for the event to occur
        return -math.log(1.0 - random.random())/ rateParameter

    def genSpine_at_branchpoint(self):
        """
        Generate the spines only at branchpoint

        """
        # NOW: traverse the neuron at different branch-order
        #  and generate the 'true' number of spines at that branch-order
        #  with location is based on the
        branchpoint_list = []
        # get all 1. distal ends, 2. branchpoints
        #    to examine toward soma
        for id in self.line_ids:
            numChildren = int(self.point_lookup[id]['numChildren'])
            #if numChildren == 0:
            if numChildren == 0 or numChildren >= 2:
                branchpoint_list.append(id)
                branchOrder = int(self.point_lookup[id]['branchOrder'])
                #if branchOrder >= len(branchorder_slot):
                #    print("""ERROR: swc file has branchOrder exceed the data available
                #          Please check to update statistics data
                #          """)

        branchpoint_list = list(set(branchpoint_list))
        ############
        # start the spine generation process
        self.spineArray = []
        self.inhibitoryArray = []
        basalSpineCount = 0
        while branchpoint_list:
            #tmplist = []
            for id in branchpoint_list:
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                #spine_filename = 'spine' +
                pid = parent_id
                if  int(pid) == -1:
                    continue
                pid = str(pid)
                while (float(self.point_lookup[pid]['dist2branchPoint']) > 0.0
                    ):
                    pid = str(self.point_lookup[pid]['parent'])
                #if (self.point_lookup[pid]['dist2branchPoint'] == 0.0):
                #    tmplist.append(pid)

                if int(branchType) == self.__class__.branchType["basal"]:
                    ## HERE INFO for NEW SPINE
                    distal_id = id
                    x1 = self.point_lookup[distal_id]['siteX']
                    y1 = self.point_lookup[distal_id]['siteY']
                    z1 = self.point_lookup[distal_id]['siteZ']
                    siteX = float(x1) #np.interp(distance-spine_distance2distalpoint, [0, distance], [parent_x1, x1])
                    siteY = float(y1) #np.interp(distance-spine_distance2distalpoint, [0, distance], [parent_y1, y1])
                    siteZ = float(z1) #np.interp(distance-spine_distance2distalpoint, [0, distance], [parent_z1, z1])
                    basalSpineCount  += 1

                    boutonType = self.__class__.boutonType["excitatory"]
                    spineType = self.__class__.spineType["generic"]
                    boutonFileName = "bouton_generic"
                    spineFileName = "spine_generic"
                    siteR = 5  # radius (micrometer) of stimulus taking effect
                    period = 0 # period of stimulus
                    bouton_include= 0
                    spine_include = 0
                    synapse_include= 0

                    self.spineArray.append([
                                        str(branchType),
                                        str(boutonType),
                                        str(spineType),
                                        str(round(siteX,3)),
                                        str(round(siteY,3)),
                                        str(round(siteZ,3)),
                                        str(siteR),
                                        str(period),
                                        str(boutonFileName),
                                        str(spineFileName),
                                        str(bouton_include),
                                        str(spine_include),
                                        str(synapse_include)
                                        ])
                    ## END
                elif (int(branchType) == self.__class__.branchType["apical"]):
                    print("WARNING: not supporting generating apical on MSN")
                    pass
                #break
            #branchpoint_list = list(set(tmplist))
            branchpoint_list =[]

        print("basal spines count: ", basalSpineCount)
        pass


    def genSpine_MSN_branchorder_based(self, use_mean=True):
        """

        Call this function to generate spines with statistics for MSN neuron
            collected from Klapstein (2001)

        NOTE: Currently generate spines on 'basal' dendrite
                        ([(branchType),
                        (boutonType),
                        (spineType),
                        (siteX),
                        (siteY),
                        (siteZ),
                        (siteR),
                        (period),
                        (boutonFileName),
                        (spineFileName),
                        (bouton_include),  # if =1 then recording bouton
                        (spine_include),
                        (synapse_include)
                        ])
        self.spineArray with

        self.inhibitoryArray with
             spineType="N/A"  (i.e. no spine to use)
             There are two options: GABA bouton come proximal or distal side of spines
             But we haven't considered here yet
        """

        branchorder_slot= range(0,5,1) # branchorder 0..4 maximum
        if __debug__:
            print("branch_order")
            print(branchorder_slot)
        incr_slot = [10] * (len(branchorder_slot)) # 21 increments (each with 10 micrometer)
        if __debug__:
            print("range ??? (micrometer) of den-length to measure spine frequency")
            # print ["{0:1.2f}".format(i) for i in incr_slot]
            print(incr_slot)
        # Klapstein (2001) data
        ###START
        mean_spineoccurence_slot = [0.5, 1.5, 7, 7, 7.7] # every incr_slot[..] (micrometer)
        std_spineoccurence_slot  = [0.2, 0.4, 1.4, 1.3, 1.0]
        ###END
        if __debug__:
            print("spines_freq: at branch-order")
            print("mean=", mean_spineoccurence_slot)
            print("std=", std_spineoccurence_slot)
            print("########################")
        """
        true_spineoccurence_slot = []
        if use_mean:
            tmp = []
            for x in mean_spineoccurence_slot:
                tmp.append(math.ceil(x))
            true_spineoccurence_slot = tmp
        else:# take a random value with mean 'mean_spineoccurence_slot'
            # and std 'std_spineoccurence_slot'
            tmp = []
            for idx, mean in enumerate(mean_spineoccurence_slot):
                val  = np.random.normal(mean, std_spineoccurence_slot[idx])
                tmp.append(math.ceil(val))
            true_spineoccurence_slot = tmp
        """
        # NOW: traverse the neuron at different branch-order
        #  and generate the 'true' number of spines at that branch-order
        #  with location is based on the
        branchpoint_list = []
        # start with all distal ends to examine toward soma
        for id in self.line_ids:
            numChildren = int(self.point_lookup[id]['numChildren'])
            if numChildren == 0:
                branchpoint_list.append(id)
                branchOrder = int(self.point_lookup[id]['branchOrder'])
                if branchOrder >= len(branchorder_slot):
                    print("""ERROR: swc file has branchOrder exceed the data available
                          Please check to update statistics data
                          """)

        branchpoint_list = list(set(branchpoint_list))

        ############
        # start the spine generation process
        self.spineArray = []
        self.inhibitoryArray = []
        basalSpineCount = 0
        while branchpoint_list:
            tmplist = []
            for id in branchpoint_list:
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                #spine_filename = 'spine' +
                if int(branchType) == self.__class__.branchType["basal"]:
                    # Find the number of spines to be generated on this branch
                    val = 0
                    if (use_mean):
                        val = mean_spineoccurence_slot[branchOrder]
                    else:
                        mean = mean_spineoccurence_slot[branchOrder]
                        std = std_spineoccurence_slot[branchOrder]
                        # number of spines to be generated on this branch
                        val = int(np.random.normal(mean, std ))
                    numSpines=int(math.ceil(val*dist2branchPoint/incr_slot[branchOrder]))

                    # Assume
                    # each spine is put into its slot
                    # of equal length
                    if numSpines>0:
                        slot_len = dist2branchPoint/numSpines
                    else:
                        slot_len = dist2branchPoint
                    """
                    x1 = self.point_lookup[id]['x']
                    y1 = self.point_lookup[id]['y']
                    z1 = self.point_lookup[id]['z']
                    pid = parent_id
                    px1 = self.point_lookup[parent_id]['x']
                    py1 = self.point_lookup[parent_id]['y']
                    pz1 = self.point_lookup[parent_id]['z']
                    """
                    spine_distance2distalpoint = 0
                    distal_id = id
                    pid = parent_id
                    for i in range(numSpines):
                        # generate spine location (as distance from
                        # distal end of the slot)
                        # Assume:
                        #  spine appearance on each slot has no bias
                        # ... using uniform distribution
                        spineLocationAsDistancetoDistalEndCurrentSlot = np.random.uniform(0, slot_len)
                        spine_distance2distalpoint = i * slot_len + spineLocationAsDistancetoDistalEndCurrentSlot
                        #gap = self.find_distance(distal_id, pid )
                        gap = self.point_lookup[id]['dist2soma']  \
                            - self.point_lookup[pid]['dist2soma']
                        while (spine_distance2distalpoint > gap and
                                self.point_lookup[pid]['parent'] != '-1'):
                            distal_id= pid
                            pid = self.point_lookup[pid]['parent']
                            spine_distance2distalpoint  -= gap
                            #gap = self.find_distance(distal_id, pid )
                            gap = self.point_lookup[id]['dist2soma']  \
                                - self.point_lookup[pid]['dist2soma']
                        x1 = self.point_lookup[distal_id]['siteX']
                        y1 = self.point_lookup[distal_id]['siteY']
                        z1 = self.point_lookup[distal_id]['siteZ']
                        parent_x1 = self.point_lookup[pid]['siteX']
                        parent_y1 = self.point_lookup[pid]['siteY']
                        parent_z1 = self.point_lookup[pid]['siteZ']
                        parent_dist2branchPoint = \
                            float(self.point_lookup[pid]['dist2branchPoint'])
                        ## HERE INFO for NEW SPINE
                        # NOTE: dis2points = distance between 2 adjacent points
                        dis2points = self.find_distance(distal_id, pid)
                        siteX = np.interp(gap-spine_distance2distalpoint, [0, dis2points], [parent_x1, x1])
                        siteY = np.interp(gap-spine_distance2distalpoint, [0, dis2points], [parent_y1, y1])
                        siteZ = np.interp(gap-spine_distance2distalpoint, [0, dis2points], [parent_z1, z1])
                        basalSpineCount  += 1

                        boutonType = self.__class__.boutonType["excitatory"]
                        spineType = self.__class__.spineType["generic"]
                        boutonFileName = "bouton_generic"
                        spineFileName = "spine_generic"
                        siteR = 5  # radius (micrometer) of stimulus taking effect
                        period = 0 # period of stimulus
                        bouton_include= 0
                        spine_include = 0
                        synapse_include= 0

                        self.spineArray.append([
                                           str(branchType),
                                           str(boutonType),
                                           str(spineType),
                                           str(round(siteX,3)),
                                           str(round(siteY,3)),
                                           str(round(siteZ,3)),
                                           str(siteR),
                                           str(period),
                                           str(boutonFileName),
                                           str(spineFileName),
                                           str(bouton_include),
                                           str(spine_include),
                                           str(synapse_include)
                                           ])
                        """
                        self.spineArray.append([str(branchType),
                                                str(boutonMType),
                                                str(siteX),
                                                str(siteY),
                                                str(siteZ),
                                                str(siteR),
                                                str(period),
                                                str(boutonType),
                                                str(spineType),
                                                str(bouton_include),
                                                str(spine_include),
                                                str(synapse_include)])
                        """
                        ## END
                        if (self.point_lookup[pid]['dist2branchPoint'] == 0.0 and
                            pid != '-1'):
                            tmplist.append(pid)
                        # the we feed this to tree's points to generate the
                        # spine location

                        #self.spineArray.append([(branchType), (spineMType), (siteX), (siteY), (siteZ), (siteR), (period), (
                        #    boutonType), (spineType), (bouton_include), (spine_include), (synapse_include), (boutonMType)])
                elif (int(branchType) == self.__class__.branchType["apical"]):
                    print("WARNING: not supporting generating apical on MSN")
                    pass
                #break
            branchpoint_list = list(set(tmplist))

        print("basal spines count: ", basalSpineCount)
        """
        basalSpineCount = 0
        for id in self.line_ids:
            branchType = self.point_lookup[id]['type']
            point_info = self.point_lookup[id]
            parent_id = self.point_lookup[id]['parent']
            if (int(parent_id) is not -1):
                # parent_info = self.point_lookup[parent_id]
                # parent_x = float(parent_info['siteX'])
                # parent_y = float(parent_info['siteY'])
                # parent_z = float(parent_info['siteZ'])

                # x = float(point_info['siteX'])
                # y = float(point_info['siteY'])
                # z = float(point_info['siteZ'])

                #distance = sqrt((x - parent_x)**2 + (y - parent_y)**2 +
                #                (z - parent_z)**2)

                if int(branchType) == self.__class__.branchType["basal"]:
                    # self.genSpines(mean_basal_thin_interval,
                    #           self.__class__.spineType["thin"],
                    #           distance, basal_period,
                    #           boutonTypeThinBasal,
                    #           spineTypeThinBasal, basal)
                    # self.genSpines(mean_basal_mush_interval, self.__class__.spineType[
                    #           "mush"], distance, basal_period, boutonTypeMushBasal, spineTypeMushBasal, basal)
                    # genInhibitory(basal_inhibitory_period, distance)
                    # basalDistance += distance
                    pass
                elif int(branchType) == self.branchType["apical"]:
                    pass

        """

    def genSpine_MSN_distance_based(self, use_mean=True):
        """

        Generate the spines based on the histogram of spine occurences
            at different distance to the soma
        """
        self.spineArray = []
        self.inhibitoryArray = []

        distance_slot = range(0,220,10) # NOTE: using 220: 22 elements --> 21 ranges
        # d1_slot =                [0  ,  10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110, 120,
        #           130,140,150,160,170,180,190,200,210 ]
        incr_slot = [10] * (len(distance_slot)-1) # 21 increments (each with 10 micrometer)
        ## Wilson et al., (1983) data
        #mean_spineoccurence_slot = [0.5, 1.5, 3, 15, 30, 40, 35, 30, 27, 25,  25, 22, 20,
        #             18, 15, 13, 13, 12, 10, 9, 7 ] # every incr_slot[..] (micrometer)
        #std_spineoccurence_slot  = [ 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]
        ## Suarez et al., (2014) data
        mean_spineoccurence_slot = [0.5, 1.5,  3,  5,  7, 8, 9, 10, 8, 9, 8 , 8, 9 ,
                     8, 8, 7, 8, 9, 9, 8, 8 ] # every incr_slot[..] (micrometer)
        std_spineoccurence_slot  = [ 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]
        ##
        # 1spine occur every '...' )um
        ### TEST CASE HERE
        range2use = len(distance_slot)  # default mode
        #range2use = 3
        distance_slot = distance_slot[0:range2use]
        incr_slot = incr_slot[0:range2use-1]
        mean_spineoccurence_slot = mean_spineoccurence_slot[0:range2use-1]
        ### END TEST CASE

        print("Distance segments:", len(distance_slot))
        print(distance_slot)
        print("Bins: ", len(incr_slot))
        print("Bin-length ??? (micrometer) of dendritic tree to measure spine frequency")
        # print ["{0:1.2f}".format(i) for i in incr_slot]
        print(incr_slot)
        print("########################")
        true_spineoccurence_slot = []
        if use_mean:
            tmp = []
            for x in mean_spineoccurence_slot:
                tmp.append(math.ceil(x))
            true_spineoccurence_slot = tmp
        else:# take a random value with mean 'mean_spineoccurence_slot'
            # and std 'std_spineoccurence_slot'
            tmp = []
            for idx, mean in enumerate(mean_spineoccurence_slot):
                val  = np.random.normal(mean, std_spineoccurence_slot[idx])
                tmp.append(val)
            true_spineoccurence_slot = tmp

        print("Mean size: ", len(true_spineoccurence_slot))
        print("mean ??? #spines/bin. Bin length is given above, e.g. every 10 micrometer")
        print ["{0:0.2f}".format(i) for i in true_spineoccurence_slot]
        print("########################")
        # Suppose we observe 'X' spines in the range [20-30] micrometer from
        # soma
        # These spines occur randomly
        # So on average, 1 spines occur every (30-20)/X micrometer
        # So on average, 1 micrometer has X/(30-20) spines
        freq_spineoccurence_slot = np.divide(true_spineoccurence_slot, incr_slot)
        lambda_slot = []  # rate parameter for Poisson process (govern spine occurence)
        for x in freq_spineoccurence_slot:
            lambda_slot.append(1.0/x)
        print("Lambda size: ", len(lambda_slot))
        print("lambda: 1 spine every ??? micrometer")
        print ["{0:0.2f}".format(i) for i in lambda_slot]

        # NOW: traverse the neuron at different branch-order
        #  and generate the 'true' number of spines at that branch-order
        #  with location is based on the
        holdpoint_list = []
        # start with all distal ends to examine toward soma
        for id in self.line_ids:
            numChildren = int(self.point_lookup[id]['numChildren'])
            if numChildren == 0:
                holdpoint_list.append(id)
                branchOrder = int(self.point_lookup[id]['branchOrder'])
                #if branchOrder >= len(branchorder_slot):
                #    print("""ERROR: swc file has branchOrder exceed the data available
                #          Please check to update statistics data
                #          """)

        holdpoint_list = list(set(holdpoint_list))
        ############
        # start the spine generation process
        self.spineArray = []
        self.inhibitoryArray = []
        basalSpineCount = 0
        """
        Basically, what the algorithm does is
          1. start with all the distal points
          2. traverse back to the point that pass the distance-range limit
            2.a. if the next point is too coarse, create a new intermediate point
          3. generate the spines using the statistics within that range limit
        """
        while holdpoint_list:
            tmplist = []
            for id in holdpoint_list:
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                dist2soma = float(point_info['dist2soma'])
                #spine_filename = 'spine' +
                #NOTE: The last index that is smaller than dist2soma
                #   This should be the index of the starting end of the bin
                if dist2soma >= distance_slot[-1]:
                    binslot_index = len(distance_slot)-2
                else:
                    binslot_index = (i for i, x in enumerate(distance_slot)
                                     if x > dist2soma).next()-1
                    if dist2soma == distance_slot[binslot_index]:
                        binslot_index -= 1
                #if (dist2soma <= distance_slot[binslot_index]):
                #    print id,binslot_index
                assert binslot_index >= 0
                assert dist2soma > distance_slot[binslot_index]

                pid = parent_id
                cid = id
                # jump back to the point
                # 1.  point_lookup[pid] is the branchpoint not passing
                #           distance_slot[binslot_index]
                # or distance_slot[binslot_index] fall between point_lookup[pid]
                #           and point_lookup[cid]
                # jump back to the point that pass the binslot threshold of
                #   distance to soma and must not pass the branchpoint
                #
                while (float(self.point_lookup[pid]['dist2soma']) >
                       distance_slot[binslot_index] and
                       self.point_lookup[pid]['dist2branchPoint'] > 0.0):
                    cid = pid
                    pid = self.point_lookup[pid]['parent']
                    #print(cid, pid, self.point_lookup[pid]['dist2soma'],
                    #      self.point_lookup[pid]['dist2branchPoint'])

                distance=self.find_distance(cid, pid)
                #distance=self.point_lookup[id]['dist2branchPoint'] - \
                #    self.point_lookup[pid]['dist2branchPoint']
                x1 = self.point_lookup[cid]['siteX']
                y1 = self.point_lookup[cid]['siteY']
                z1 = self.point_lookup[cid]['siteZ']
                parent_x1 = self.point_lookup[pid]['siteX']
                parent_y1 = self.point_lookup[pid]['siteY']
                parent_z1 = self.point_lookup[pid]['siteZ']
                #GOAL: find a proximal-side point along the tree that
                #   1. not exceeding the binslot length
                #   2. not passing the branchpoint
                # if the point with dist2soma fall outside the range of
                #              the binslot length, then create an intermediate
                #              point
                if (distance_slot[binslot_index] >
                    self.point_lookup[pid]['dist2soma']): # case 1= the binslot point
                    #  falls into between the two points 'pid'  and 'cid'
                    #  So we need to
                    # create an intermediate point using this binslot point
                    binslot_distance2pid = - self.point_lookup[pid]['dist2soma'] \
                        + distance_slot[binslot_index]
                    assert binslot_distance2pid > 0.0
                    siteX = np.interp(binslot_distance2pid, [0, distance], [parent_x1, x1])
                    siteY = np.interp(binslot_distance2pid, [0, distance], [parent_y1, y1])
                    siteZ = np.interp(binslot_distance2pid, [0, distance], [parent_z1, z1])

                    newid = str(len(self.point_lookup)+1)
                    newid_dist2soma = distance_slot[binslot_index]
                    newid_dist2branchPoint = newid_dist2soma -\
                        self.point_lookup[pid]['dist2soma'] + \
                        self.point_lookup[pid]['dist2branchPoint']
                    assert newid_dist2soma >= 0.0
                    assert newid_dist2branchPoint >= 0.0
                    self.point_lookup[newid] = {'type': self.point_lookup[cid]['type'],
                                            'siteX': str(siteX), 'siteY': str(siteY),
                                            'siteZ': str(siteZ),
                                            'siteR': self.point_lookup[cid]['siteR'],
                                            'parent': pid,
                                            'dist2soma': newid_dist2soma,
                                            'dist2branchPoint': newid_dist2branchPoint,
                                            'branchOrder': branchOrder,
                                            'numChildren': 1}
                    self.point_lookup[cid]['parent'] = str(newid)
                    #if cid == id:
                    #    parent_id = self.point_lookup[id]['parent']
                    #parent_id = self.point_lookup[cid]['parent']

                    tmplist.append(newid)
                    length_region2consider = self.point_lookup[id]['dist2soma']- \
                        self.point_lookup[newid]['dist2soma']
                else: # case 2 = face the branchpoint
                    length_region2consider = self.point_lookup[id]['dist2soma']\
                        - self.point_lookup[pid]['dist2soma']
                    if (self.point_lookup[pid]['dist2soma'] > 0):
                        tmplist.append(pid)


                if int(branchType) == self.__class__.branchType["basal"]:
                    slot_len = lambda_slot[binslot_index] # length for 1 spine to occur

                    spine_distance2distalpoint = 0
                    distal_id = id
                    pid = parent_id
                    numSpines = int(math.floor(length_region2consider / slot_len))
                    for i in range(numSpines):
                        # generate spine location (as distance from
                        # distal end of the slot)
                        # Assume:
                        #  spine appearance on each slot has no bias
                        # ... using uniform distribution
                        spineLocationAsDistancetoDistalEndCurrentSlot = np.random.uniform(0, slot_len)
                        spine_distance2distalpoint = i * slot_len + spineLocationAsDistancetoDistalEndCurrentSlot
                        #gap = self.find_distance(distal_id, pid)
                        gap = self.point_lookup[id]['dist2soma'] \
                            - self.point_lookup[pid]['dist2soma']
                        while (spine_distance2distalpoint > gap and
                                self.point_lookup[pid]['parent'] != '-1'):
                            distal_id= pid
                            pid = self.point_lookup[pid]['parent']
                            spine_distance2distalpoint  -= gap
                            #gap = self.find_distance(distal_id, pid )
                            gap = self.point_lookup[id]['dist2soma']  \
                                - self.point_lookup[pid]['dist2soma']
                        x1 = self.point_lookup[distal_id]['siteX']
                        y1 = self.point_lookup[distal_id]['siteY']
                        z1 = self.point_lookup[distal_id]['siteZ']
                        parent_x1 = self.point_lookup[pid]['siteX']
                        parent_y1 = self.point_lookup[pid]['siteY']
                        parent_z1 = self.point_lookup[pid]['siteZ']
                        distance = self.find_distance(distal_id, pid)
                        #parent_dist2branchPoint = float(self.point_lookup[pid]['dist2branchPoint'])
                        ## HERE INFO for NEW SPINE
                        # NOTE: dis2points = distance between 2 adjacent points
                        dis2points = self.find_distance(distal_id, pid)
                        siteX = np.interp(gap-spine_distance2distalpoint, [0, dis2points], [parent_x1, x1])
                        siteY = np.interp(gap-spine_distance2distalpoint, [0, dis2points], [parent_y1, y1])
                        siteZ = np.interp(gap-spine_distance2distalpoint, [0, dis2points], [parent_z1, z1])
                        basalSpineCount  += 1

                        boutonType = self.__class__.boutonType["excitatory"]
                        spineType = self.__class__.spineType["generic"]
                        boutonFileName = "bouton_generic"
                        spineFileName = "spine_generic"
                        siteR = 5  # radius (micrometer) of stimulus taking effect
                        period = 0 # period of stimulus
                        bouton_include= 0
                        spine_include = 0
                        synapse_include= 0

                        self.spineArray.append([
                                           str(branchType),
                                           str(boutonType),
                                           str(spineType),
                                           str(round(siteX,3)),
                                           str(round(siteY,3)),
                                           str(round(siteZ,3)),
                                           str(siteR),
                                           str(period),
                                           str(boutonFileName),
                                           str(spineFileName),
                                           str(bouton_include),
                                           str(spine_include),
                                           str(synapse_include)
                                           ])


                """

                    numspines_region2consider = length_region2consider / \
                        lambda_slot[binslot_index]
                    #Find the point
                    # Find the number of spines to be generated on this branch
                    pid = id
                    distance2soma = distance_slot[binslot_index]
                    #while (self.point_lookup[pid]["dist2branchPoint"] > 0.0 and
                    #       distance2soma > distance ):
                    #    pass

                    #do (pid = self.point_lookup[pid]["parent"])
                    val = 0
                    if (use_mean):
                        val = mean_spineoccurence_slot[branchOrder]
                    else:
                        mean = mean_spineoccurence_slot[branchOrder]
                        std = std_spineoccurence_slot[branchOrder]
                        # number of spines to be generated on this branch
                        val = int(np.random.normal(mean, std ))
                    numSpines=int(math.ceil(val*dist2branchPoint/incr_slot[branchOrder]))

                """

                #elif int(branchType) == self.branchType["apical"]:
                #    pass
            holdpoint_list = list(set(tmplist))
        print("basal spines count: ", basalSpineCount)

    def genSpine_MSN_distance_based_test(self, use_mean=True):
        """
        Call this function to generate spines with statistics for MSN neuron
            collected from Wilson (1993)

        if use_mean = True:
            then generate the spines exact 'mean' spines in the range
            the exact location of these 'mean' number of spines is determined by
                the Poisson distribution

        """
        self.spineArray = []
        self.inhibitoryArray = []

        distance_slot = range(0,220,10) # NOTE: using 220: 22 elements --> 21 ranges
        # d1_slot =                [0  ,  10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110, 120,
        #           130,140,150,160,170,180,190,200,210 ]
        incr_slot = [10] * (len(distance_slot)-1) # 21 increments (each with 10 micrometer)
        mean_spineoccurence_slot = [0.5, 0.5, 10, 15, 30, 40, 35, 30, 27, 25,  25, 22, 20,
                     18, 15, 13, 13, 12, 10, 9, 7 ] # every incr_slot[..] (micrometer)
        std_spineoccurence_slot  = [ 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]
        # 1spine occur every '...' )um

        print("range ??? (micrometer) of den-length to measure spine frequency")
        # print ["{0:1.2f}".format(i) for i in incr_slot]
        print(incr_slot)
        print("########################")
        true_spineoccurence_slot = []
        if use_mean:
            tmp = []
            for x in mean_spineoccurence_slot:
                tmp.append(math.ceil(x))
            true_spineoccurence_slot = tmp
        else:# take a random value with mean 'mean_spineoccurence_slot'
            # and std 'std_spineoccurence_slot'
            tmp = []
            for idx, mean in enumerate(mean_spineoccurence_slot):
                val  = np.random.normal(mean, std_spineoccurence_slot[idx])
                tmp.append(val)
            true_spineoccurence_slot = tmp

        print("mean ??? #spines/slot. Here a slot is 10 micrometer")
        print ["{0:0.2f}".format(i) for i in true_spineoccurence_slot]
        print("########################")
        # Suppose we observe 'X' spines in the range [20-30] micrometer from
        # soma
        # These spines occur randomly
        # So on average, 1 spines occur every (30-20)/X micrometer
        # So on average, 1 micrometer has X/(30-20) spines
        freq_spineoccurence_slot = np.divide(true_spineoccurence_slot, incr_slot)
        lambda_slot = []  # rate parameter for Poisson process (govern spine occurence)
        for x in freq_spineoccurence_slot:
            lambda_slot.append(1.0/x)
        print("lambda: 1 spine every ??? micrometer")
        print ["{0:0.2f}".format(i) for i in lambda_slot]
        #print(lambda_slot)
        print("########################")
        print("location at which spine occurs [offset from 0-10]")
        for idx, val in enumerate(true_spineoccurence_slot):
            #spinesInfo = xrange(int(incr_slot[idx]/lambda_slot[idx]))
            spinesInfo = [self.nextLocation(lambda_slot[idx])
                          for i in
                          xrange(int(true_spineoccurence_slot[idx]))]
                          #xrange(int(incr_slot[idx]/freq_spineoccurence_slot[idx]))]
            # spinesInfo = np.random.poisson(lambda_slot[idx], val)
            # print(idx, val)
            tot = np.cumsum(spinesInfo) # np.ndarray
            tot = list(tot) # convert to list
            if idx==4:
                print("freq: 1 spine per", lambda_slot[idx], " (um)")
                print ["{0:0.2f}".format(i) for i in spinesInfo]
                #print ["{0:0.2f}".format(i) for i in tot]
        # Strategy:
        # for each tree (dendritic)
        # traverse from soma to end
        #  at each segment - find the dist2soma and length
        #  check the index 'idx' it belong to in distance_slot
        #  get the rateParameter = lambda_slot[idx]
        #  generate the location of next-spine
        #     loc = self.nextTime(rateParameter)
        #  find the segment it belong to, put it there
        basalSpineCount = 0
        for id in self.line_ids:
            branchType = self.point_lookup[id]['type']
            point_info = self.point_lookup[id]
            parent_id = self.point_lookup[id]['parent']
            if (int(parent_id) is not -1):
                # parent_info = self.point_lookup[parent_id]
                # parent_x = float(parent_info['siteX'])
                # parent_y = float(parent_info['siteY'])
                # parent_z = float(parent_info['siteZ'])

                # x = float(point_info['siteX'])
                # y = float(point_info['siteY'])
                # z = float(point_info['siteZ'])

                #distance = sqrt((x - parent_x)**2 + (y - parent_y)**2 +
                #                (z - parent_z)**2)

                if int(branchType) == self.__class__.branchType["basal"]:
                    # self.genSpines(mean_basal_thin_interval,
                    #           self.__class__.spineType["thin"],
                    #           distance, basal_period,
                    #           boutonTypeThinBasal,
                    #           spineTypeThinBasal, basal)
                    # self.genSpines(mean_basal_mush_interval, self.__class__.spineType[
                    #           "mush"], distance, basal_period, boutonTypeMushBasal, spineTypeMushBasal, basal)
                    # genInhibitory(basal_inhibitory_period, distance)
                    # basalDistance += distance
                    pass
                elif int(branchType) == self.branchType["apical"]:
                    pass

    def genSpines_Poisson(self, mean_interval, spineMType, distance, period, boutonType, spineType, branchType):
        """

        Generate a spine

        @param mean_interval = mean distance of spines
        @type float
        @param spineMType    = the morphology type ()
        @type spineType data member
        @param distance      =
        """
        distance_slot = range(0,220,10) # NOTE: using 220: 22 elements --> 21 ranges
        # d1_slot =                [0  ,  10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110, 120,
        #           130,140,150,160,170,180,190,200,210 ]
        incr_slot = [10] * (len(distance_slot)-1) # 21 increments (each with 10 micrometer)
        mean_spineoccurence_slot = [0.5, 0.5, 10, 15, 30, 40, 35, 30, 27, 25,  25, 22, 20,
                     18, 15, 13, 13, 12, 10, 9, 7 ] # every incr_slot[..] (micrometer)
        std_spineoccurence_slot  = [ 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]
        # 1spine occur every '...' )um
        freq_spineoccurence_slot = np.divide(incr_slot, mean_spineoccurence_slot)
        lambda_slot = []  # rate parameter for Poisson process (govern spine occurence)
        for x in freq_spineoccurence_slot:
            lambda_slot.append(1.0/x)

        spine_location = 0
        siteR = 5
        bouton_include = 0
        spine_include = 0
        synapse_include = 0
        junctionRate = .03
        spineJunctionrate = .61
        mushroomJunctionRate = .21
        while True:
            # Assume spine appearance with adjacent-distance
            # follows exponential distribution
            interval = rnd.exponential(mean_interval)
            spine_location += interval
            if (spine_location < distance):
                siteX = np.interp(spine_location, [0, distance], [parent_x, x])
                siteY = np.interp(spine_location, [0, distance], [parent_y, y])
                siteZ = np.interp(spine_location, [0, distance], [parent_z, z])
                if spineMType == 2:
                    boutonMType = 4
                elif spineMType == 3:
                    boutonMType = 5
                self.spineArray.append([str(branchType), str(spineMType), str(siteX), str(siteY), str(siteZ), str(siteR), str(period), str(
                    boutonType), str(spineType), str(bouton_include), str(spine_include), str(synapse_include), str(boutonMType)])
                junctionRand = rnd.rand()
                if junctionRand <= junctionRate:
                    junctionTypeRand = rnd.rand()
                    junctionX = siteX
                    junctionY = siteY
                    junctionZ = siteZ
                    if junctionTypeRand <= mushroomJunctionRate:
                        junctionMType = 2
                        boutonMType = 4
                        junctionBoutonType = bouton + thin + agingType + branchType
                        junctionSpineType = spine + thin + agingType + branchType
                    elif junctionTypeRand > mushroomJunctionRate:
                        junctionMType = 3
                        boutonMType = 5
                        junctionBoutonType = bouton + mush + agingType + branchType
                        junctionSpineType = spine + mush + agingType + branchType
                        self.spineArray.append([str(branchType), str(spineMType), str(siteX), str(siteY), str(siteZ), str(siteR), str(period), str(
                            junctionBoutonType), str(junctionSpineType), str(bouton_include), str(spine_include), str(synapse_include), str(boutonMType)])
                    else:
                        break

    def genSpine_PyramidalL5(self):
        """
        Call this function to generate spines with statistics for Pyramidal LayerV neuron

        """
        agingType = 'young'
        bouton = 'bouton_'
        spine = 'spine_'
        thin = 'thin_'
        mush = 'mushroom_'
        apical = '_apical'
        basal = '_basal'

        boutonTypeThinApical = bouton + thin + agingType + apical
        spineTypeThinApical = spine + thin + agingType + apical
        boutonTypeMushApical = bouton + mush + agingType + apical
        spineTypeMushApical = spine + mush + agingType + apical
        boutonTypeThinBasal = bouton + thin + agingType + basal
        spineTypeThinBasal = spine + thin + agingType + basal
        boutonTypeMushBasal = bouton + mush + agingType + basal
        spineTypeMushBasal = spine + mush + agingType + basal

        scaling_factor = 1.0  # for distance scaling
        mean_apical_thin_interval = scaling_factor * 1.6
        mean_apical_mush_interval = scaling_factor * 5.0
        mean_basal_thin_interval = scaling_factor * 1.6
        mean_basal_mush_interval = scaling_factor * 5.0

        basalDistance = 0
        apicalDistance = 0
        apical_period = 300
        basal_period = 140
        apical_inhibitory_period = 250
        basal_inhibitory_period = 250

        self.spineArray = []
        self.inhibitoryArray = []

        for id in self.line_ids:
            branchType = self.point_lookup[id]['type']
            current_point = self.point_lookup[id]
            parent_id = self.point_lookup[id]['parent']
            if (int(parent_id) is not -1):
                parent_point = self.point_lookup[parent_id]
                parent_x = float(parent_point['siteX'])
                parent_y = float(parent_point['siteY'])
                parent_z = float(parent_point['siteZ'])

                x = float(current_point['siteX'])
                y = float(current_point['siteY'])
                z = float(current_point['siteZ'])

                distance = sqrt((x - parent_x)**2 + (y - parent_y)
                                ** 2 + (z - parent_z)**2)

                if int(branchType) == self.branchType["basal"]:
                    basalSpineCount = 0
                    self.genSpines(mean_basal_thin_interval, self.__class__.spineType[
                              "thin"], distance, basal_period, boutonTypeThinBasal, spineTypeThinBasal, basal)
                    self.genSpines(mean_basal_mush_interval, self.__class__.spineType[
                              "mush"], distance, basal_period, boutonTypeMushBasal, spineTypeMushBasal, basal)
                    genInhibitory(basal_inhibitory_period, distance)
                    basalDistance += distance

                elif int(branchType) == self.branchType["apical"]:
                    apicalSpineCount = 0
                    self.genSpines(mean_apical_thin_interval, self.__class__.spineType[
                              "thin"], distance, apical_period, boutonTypeThinApical, spineTypeThinApical, apical)
                    self.genSpines(mean_apical_mush_interval, self.__class__.spineType[
                              "mush"], distance, apical_period, boutonTypeMushApical, spineTypeMushApical, apical)
                    genInhibitory(apical_inhibitory_period, distance)
                    apicalDistance += distance

    def genSpines(self, mean_interval, spineMType, distance, period, boutonType, spineType, branchType):
        """

        Generate a spine

        @param mean_interval = mean distance of spines
        @type float
        @param spineMType    = the morphology type ()
        @type spineType data member
        @param distance      =
        """
        spine_location = 0
        siteR = 5
        bouton_include = 0
        spine_include = 0
        synapse_include = 0
        junctionRate = .03
        spineJunctionrate = .61
        mushroomJunctionRate = .21
        while True:
            # Assume spine appearance with adjacent-distance
            # follows exponential distribution
            interval = rnd.exponential(mean_interval)
            spine_location += interval
            if (spine_location < distance):
                siteX = np.interp(spine_location, [0, distance], [parent_x, x])
                siteY = np.interp(spine_location, [0, distance], [parent_y, y])
                siteZ = np.interp(spine_location, [0, distance], [parent_z, z])
                if spineMType == 2:
                    boutonMType = 4
                elif spineMType == 3:
                    boutonMType = 5
                    self.spineArray.append([str(branchType), str(spineMType), str(siteX), str(siteY), str(siteZ), str(siteR), str(period), str(
                        boutonType), str(spineType), str(bouton_include), str(spine_include), str(synapse_include), str(boutonMType)])
                    junctionRand = rnd.rand()
                    if junctionRand <= junctionRate:
                        junctionTypeRand = rnd.rand()
                        junctionX = siteX
                        junctionY = siteY
                        junctionZ = siteZ
                        if junctionTypeRand <= mushroomJunctionRate:
                            junctionMType = 2
                            boutonMType = 4
                            junctionBoutonType = bouton + thin + agingType + branchType
                            junctionSpineType = spine + thin + agingType + branchType
                        elif junctionTypeRand > mushroomJunctionRate:
                            junctionMType = 3
                            boutonMType = 5
                            junctionBoutonType = bouton + mush + agingType + branchType
                            junctionSpineType = spine + mush + agingType + branchType
                            self.spineArray.append([str(branchType), str(spineMType), str(siteX), str(siteY), str(siteZ), str(siteR), str(period), str(
                                junctionBoutonType), str(junctionSpineType), str(bouton_include), str(spine_include), str(synapse_include), str(boutonMType)])
                        else:
                            break

    def genInhibitory(self, period, distance):
        mean_inhibitory_interval = 3
        boutonMType = 1
        inhibitory_location = 0
        siteR = 5
        bouton_include = 0
        spine_include = 'N/A'
        synapse_include = 0
        boutonType = 'bouton_inhibitory'
        spineType = 'N/A'

        while True:
            interval = rnd.exponential(mean_inhibitory_interval)
            inhibitory_location += interval
            if (inhibitory_location < distance):
                siteX = np.interp(inhibitory_location, [
                                  0, distance], [parent_x, x])
                siteY = np.interp(inhibitory_location, [
                                  0, distance], [parent_y, y])
                siteZ = np.interp(inhibitory_location, [
                                  0, distance], [parent_z, z])
                self.inhibitoryArray.append([str(branchType), str(boutonMType), str(siteX), str(siteY), str(siteZ), str(siteR), str(
                    period), str(boutonType), str(spineType), str(bouton_include), str(spine_include), str(synapse_include)])
            else:
                break

    def rotateSpines(self):
        """
        mark the rotate index so that two adjacent spines have different rotation index

        1. rotate spine + bouton
        2. rotate GABA bouton

        """

        x = (len(self.spineArray))
        y = (len(self.inhibitoryArray))

        orientation = []
        results = itertools.cycle(self.rotation_indices)
        num = 0
        for value in results:
            orientation.append([value])
            num += 1
            if num >= x:
                break

        spineArray = np.asarray(self.spineArray)
        all_data = []
        all_data = np.hstack((spineArray, orientation))
        titles = ['branchType', 'boutonType', 'spineType',
                  'siteX', 'siteY', 'siteZ', 'siteR', 'period',
                  'boutonFileName','spineFileName',
                  'bouton_include', 'spine_include', 'synapse_include',
                  'orientation']

        self.outputSpinesData = []
        self.outputSpinesData = np.vstack((titles, all_data))
        self.excite_boutonspineData = all_data
        self.extitatory_titles = titles

        self.inhibitory_output = []
        if self.inhibitoryArray:
            inhibitory_orientation = []
            inhibitory_results = itertools.cycle(self.rotation_indices)
            inhib_num = 0
            for value in inhibitory_results:
                inhibitory_orientation.append([value])
                inhib_num += 1
                if inhib_num >= y:
                    break
            inhibitoryArray = np.asarray(self.inhibitoryArray)
            #inhibitory_data = []
            #inhibitory_data = np.hstack((inhibitoryArray, inhibitory_orientation))
            #inhibitory_titles = ['branchType', 'boutonMType', 'siteX', 'siteY', 'siteZ', 'siteR', 'period', 'boutonType', 'spineType', 'bouton_include', 'spine_include', 'synapse_include', 'orientation']
            #inhibitory_titles = ['branchType', 'boutonMType',
            #                     'siteX', 'siteY', 'siteZ', 'siteR', 'period',
            #                     'boutonType', 'spineType',
            #                     'bouton_include', 'spine_include', 'synapse_include']
            inhibitory_titles = ['branchType', 'boutonType', 'spineType',
                    'siteX', 'siteY', 'siteZ', 'siteR', 'period',
                    'boutonFileName','spineFileName',
                    'bouton_include', 'spine_include', 'synapse_include',
                    'orientation']
            self.inhibitory_output = np.vstack((inhibitory_titles, inhibitoryArray))
            self.inhibit_boutonData = inhibitoryArray
            self.inhibitory_titles = inhibitory_titles

    def saveExcitatoryBoutonSpineInfo(self):
        """
        e.g. spines.txt
        organize into multiple fields (space-delimited)
        14 fields
        """
        try:
            spineFile = open(self.spineFileName, "w")
        except IOError:
            print "Could not open file to write! Please check !" +  self.spineFileName
        np.savetxt(spineFile, self.outputSpinesData,
                   fmt='%s %s %s %s %s %s %s %s %s %s %s %s %s %s')
        spineFile.close()

    def saveInhibitBoutonInfo(self):
        """
        e.g. inhibitory.txt
        """
        if self.inhibitory_output:
            try:
                inhibitoryFile = open(self.inhibitoryFileName, "w")
            except IOError:
                print "Could not open file to write! Please check !" +  self.inhibitoryFileName
            np.savetxt(inhibitoryFile, self.inhibitory_output,
                    fmt='%s %s %s %s %s %s %s %s %s %s %s %s')
            inhibitoryFile.close()

    def saveStatisticsToFile(self):
        """
        Save the spine statistics to file
        """
        try:
            linesSpines = open(self.spineFileName).read().splitlines()
        except IOError:
            print "Could not open file to write! Please check !" +  self.spineFileName

        split_linesSpines = map(lambda x: x.split(' '), linesSpines)

        inputs = []

        for i, line in enumerate(split_linesSpines[1:]):
            inputs.append({'dendriteIdent': line[0], 'spineIdent': line[1]})

        df = pd.DataFrame.from_dict(inputs)
        apicalThin = (df['dendriteIdent'] == '4') & (df['spineIdent'] == '2')
        apicalMush = (df['dendriteIdent'] == '4') & (df['spineIdent'] == '3')
        basalThin = (df['dendriteIdent'] == '3') & (df['spineIdent'] == '2')
        basalMush = (df['dendriteIdent'] == '3') & (df['spineIdent'] == '3')
        apicalDensityAll = (sum(apicalThin) + sum(apicalMush)) / apicalDistance
        apicalDensityThin = sum(apicalThin) / apicalDistance
        apicalDensityMush = sum(apicalMush) / apicalDistance
        basalDensityAll = (sum(basalThin) + sum(basalMush)) / basalDistance
        basalDensityThin = sum(basalThin) / basalDistance
        basalDensityMush = sum(basalMush) / basalDistance

        statsFile = open(self.statFileName, "w")
        statsFile.write('type' + "\t" + 'apicalLength' + "\t" + 'basalLength' + "\t" + 'apicalDensityAll' + "\t" + 'apicalDensityThin' +
                        "\t" + 'apicalDensityMush' + "\t" + 'basalDensityAll' + "\t" + 'basalDensityThin' + "\t" + 'basalDensityMush' + "\n")
        statsArray = [str(agingType), str(apicalDistance), str(basalDistance), str(apicalDensityAll), str(
            apicalDensityThin), str(apicalDensityMush), str(basalDensityAll), str(basalDensityThin), str(basalDensityMush)]
        statsFile.write('\t'.join(statsArray) + "\n")
        statsFile.close()

    def saveSpines(self, spine_filename='spines.txt', inhibit_filename="inhibitory.txt",
                   stat_filename="spine_stats.txt"):
        """
        save to  file
        """
        # Part 1
        self.spineFileName = spine_filename
        self.saveExcitatoryBoutonSpineInfo()

        # Part 2
        self.inhibitoryFileName = inhibit_filename
        self.saveInhibitBoutonInfo()

        # Part 3: statistical analysis
        #self.statFileName = stat_filename
        #self.saveStatisticsToFile()

    def genboutonspineSWCFiles_MSN(self, targetFolder="neurons"):
        # NOTE: assume 8:9 columns are boutonFileName|spineFileName
        boutonspines = self.excite_boutonspineData[:, [8,9]]
        boutonspines = np.vstack({tuple(row) for row in boutonspines})
        if (not targetFolder.startswith("/")) and (not targetFolder.startswith(".")):
            targetFolder = "./" + targetFolder
        if (cur_version >= (3,2)):
            os.makedirs(os.path.dirname(targetFolder), exist_ok=True)
        else:
            if not os.path.exists(os.path.dirname(targetFolder)):
                try:
                    os.makedirs(os.path.dirname(targetFolder))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
        spineNeckRadius = 0.2 # micrometer
        spineNeckLen    = 0.99 # micrometer
        spineHeadRadius = 0.35/2 # micrometer
        boutonHeadRadius = 0.273 # micrometer
        # NOTE:
        # 2=use axon-convention
        # 3=use basal-den convention
        # thinBouton = mushroomBouton = 2
        #boutonSWC = 2
        boutonSWC = self.__class__.branchType["bouton"]
        # spineBasal = spineApical = 3
        spineNeckSWC = self.__class__.branchType["basal"]
        spineHeadSWC = self.__class__.branchType["soma"]
        for row in boutonspines:
            spineFN = row[1]
            boutonFN = row[0]
            for index, angle in enumerate(self.rotation_angles):
                self.genSWC (targetFolder, spineFN, boutonFN,
                             spineNeckSWC,
                             spineHeadSWC,
                             boutonSWC,
                             spineNeckRadius,
                             spineNeckLen,
                             spineHeadRadius,
                             boutonHeadRadius,
                             angle, index)

    def genSWC(self, targetFolder, spineFN, boutonFN,
               spineNeckSWC, spineHeadSWC,
               boutonSWC,
               spineNeckRadius,
               spineNeckLen,
               spineHeadRadius,
               boutonHeadRadius,
               degrees, orientation):
        """
        @param targetFolder = location of all boutons.swc and spines.wc files
        @param spineFN =  string telling the filename prefix of spine
                e.g. spine_generic, spine_thin_aged_basal, spine_thin_aged_apical
        @param boutonFN = string telling the filename prefix of spine

        @param spineNeckSWC =   the type of the compartment representing the spine neck
                in .SWC file
        @param spineHeadSWC =   the type of the compartment representing the spine head
                in .SWC file
        @param boutonSWC = the type of the compartment representing the bouton neck
                in .SWC file

        @param spineNeckRadius = radius of the spine neck
        @param spineNeckLen =
        @param spineHeadRadius = radius of the spine head
        @param boutonHeadRadius = radius of the spherical bouton neck
        @param degrees =  [radian] the orientation
                5 groups: 0, 144, 288, 72, and 216 [rad]
        @param orientation = an index indicating what degree of orientation being used
                0, 1, 2, 3, and 4
        """
        spinename = targetFolder +"/"+str(spineFN)+"_"+str(orientation)+".swc"
        boutonname = targetFolder+"/"+str(boutonFN)+"_"+str(orientation)+".swc"
        SWCFileSpine = open(spinename, "w")
        SWCFileBouton = open(boutonname, "w")
        spineDistance1 = .1 #neck distance to shaft
        #spineDistance2 = (spineDistance1 + typeDTS - spineHeadRadius)
        spineDistance2 = (spineDistance1 + spineNeckLen)
        #radius1 = .2
        radius1 = spineNeckRadius
        radius2 = spineHeadRadius
        #boutonRadius = 0.05
        #boutonRadius =  boutonHeadRadius

        #################
        # start generating data
        spineArray = []
        boutonArray = []

        ##################
        # SPINE
        siteX1 = 0.0
        siteY1 = (spineDistance1)*(math.cos(degrees))
        siteZ1 = (spineDistance1)*(math.sin(degrees))
        siteX2 = 0.0
        siteY2 = (spineDistance2)*(math.cos(degrees))
        siteZ2 = (spineDistance2)*(math.sin(degrees))

        spineArray.append(['1', str(spineHeadSWC),
                           str(siteX2), str(siteY2),
                           str(siteZ2), str(radius2), '-1'])
        spineArray.append(['2', str(spineNeckSWC),
                           str(siteX1), str(siteY1),
                           str(siteZ1), str(radius1), '1'])
        spineArray = np.asarray(spineArray)
        np.savetxt(spinename, spineArray, fmt='%s')

        ###################
        ## BOUTON
        GenAxonSoma = 1
        GenBoutonAxonSoma = 2
        GenMethod = GenAxonSoma  # choose one of the values above
        if (GenMethod == GenAxonSoma):
            boutonDistance1 = spineDistance2 + boutonHeadRadius
            boutonDistance2 = boutonDistance1 + boutonHeadRadius

            boutonSiteX2 = 0.0
            boutonSiteY2 = (boutonDistance2)*math.cos(degrees)
            boutonSiteZ2 = (boutonDistance2)*math.sin(degrees)
            boutonSiteX1 = 0.0
            boutonSiteY1 = (boutonDistance1)*math.cos(degrees)
            boutonSiteZ1 = (boutonDistance1)*math.sin(degrees)


            #presynSomaRadius = 5.0 # micrometer
            presynSomaRadius = 0.6 # micrometer
            boutonArray.append(['1', '1',
                                str(boutonSiteX2),
                                str(boutonSiteY2),
                                str(boutonSiteZ2),
                                str(presynSomaRadius), '-1'])

            boutonArray.append(['2', '2',
                                str(boutonSiteX1),
                                str(boutonSiteY1),
                                str(boutonSiteZ1),
                                str(boutonHeadRadius), '1'])
        elif (GenMethod == GenBoutonAxonSoma):
            axon_length = 10.0
            boutonDistance1 = spineDistance2 + boutonHeadRadius
            boutonDistance2 = boutonDistance1 + boutonHeadRadius
            boutonDistance3 = boutonDistance2 + axon_length

            boutonSiteX3 = 0.0
            boutonSiteY3 = (boutonDistance3)*math.cos(degrees)
            boutonSiteZ3 = (boutonDistance3)*math.sin(degrees)
            boutonSiteX2 = 0.0
            boutonSiteY2 = (boutonDistance2)*math.cos(degrees)
            boutonSiteZ2 = (boutonDistance2)*math.sin(degrees)
            boutonSiteX1 = 0.0
            boutonSiteY1 = (boutonDistance1)*math.cos(degrees)
            boutonSiteZ1 = (boutonDistance1)*math.sin(degrees)


            presynSomaRadius = 5.0 # micrometer
            boutonArray.append(['1', '1',
                                str(boutonSiteX3),
                                str(boutonSiteY3),
                                str(boutonSiteZ3),
                                str(presynSomaRadius), '-1'])

            # presynAxonRadius = 1.0 # micrometer

            presynTelodendriaRadius = 0.2
            boutonArray.append(['2', '2',
                                str(boutonSiteX2),
                                str(boutonSiteY2),
                                str(boutonSiteZ2),
                                str(presynTelodendriaRadius), '1'])
            boutonArray.append(['3', boutonSWC,
                                str(boutonSiteX1),
                                str(boutonSiteY1),
                                str(boutonSiteZ1),
                                str(boutonHeadRadius), '2'])
        else:
            print("NON accept bouton setting")
            assert(0)

        boutonArray = np.asarray(boutonArray)
        np.savetxt(boutonname, boutonArray, fmt='%s')

        SWCFileSpine.close()
        SWCFileBouton.close()

    def genTissueText(self, tissueFileName ="neurons.txt"):
        """
        Create the tissue file (e.g. neurons.txt)
        with all boutons _+ spines information based on the template
        neurons.txt.template
        """
        if (not tissueFileName.startswith("/")) and (not tissueFileName.startswith(".")):
            tissueFileName = "./" + tissueFileName
        self.tissueFileName = tissueFileName

        try:
            lines = open(self.spineFileName).read().splitlines()
        except IOError:
            print "Could not open file to write! Please check !" +  self.spineFileName
        split_lines = map(lambda x: x.split(' '), lines)

        inputs = []

        for i, line in enumerate(split_lines[1:]):
            inputs.append({'name': (i+1),
                           'bouton_index': (((i+1)*2)-1),
                           'spine_index': ((i+1)*2),
                           'branchType': line[0],
                           'boutonType': line[1],
                           'spineMType': line[2],
                           'siteX': line[3],
                           'siteY': line[4],
                           'siteZ': line[5],
                           'siteR': line[6],
                           'period': line[7],
                           'boutonFileName': line[8],
                           'spineFileName': line[9],
                           'bouton_include': line[10]=='1',
                           'spine_include': line[11]=='1',
                           'synapse_include': line[12]=='1',
                           'orientation': line[13]})

        try:
            linesInhibitory = open(self.inhibitoryFileName).read().splitlines()
        except IOError:
            print "Could not open file to write! Please check !" +  self.spineFileName
        split_linesInhibitory = map(lambda x: x.split(' '), linesInhibitory)

        inputsInhibitory = []

        x = len(open(self.spineFileName).readlines())

        for i, line in enumerate(split_linesInhibitory[1:]):
            inputsInhibitory.append({'name': (i+x),
                                     'bouton_index': ((i+(2*x))-1),
                                     'branchType': line[0],
                                     'boutonType': line[1],
                                     'siteX': line[2],
                                     'siteY': line[3],
                                     'siteZ': line[4],
                                     'siteR': line[5],
                                     'period': line[6],
                                     'boutonFileName': line[7],
                                     'spineFileName': line[8],
                                     'bouton_include': line[9]=='1',
                                     'spine_include': line[10]=='1',
                                     'synapse_include': line[11]=='1'})


        with open("neurons.txt.template", "r") as file:
            txt = file.read()
            newFile = open(self.tissueFileName, "w")
            newFile.write(pystache.render(txt,
                                          {'inputs': inputs,
                                           'inputsInhibitory': inputsInhibitory}))
            newFile.close()

    def genModelGSL(self, modelfile="model.gsl"):
        """

        """
        print("""IMPORTANT:
              The convention for generating is:
              suppose the model file is <model.gsl>
              within that file it needs to include
                #include "stimulus_<model.gsl>"
                #include "recording_<model.gsl>"
                #include "connect_<model.gsl>"
              at the proper position
              """)
        try:
            lines = open(self.spineFileName).read().splitlines()
        except IOError:
            print "Could not open file to write! Please check !" +  self.spineFileName
        split_lines = map(lambda x: x.split(' '), lines)

        inputs = []

        for i, line in enumerate(split_lines[1:]):
            inputs.append({'name': (i+1),
                           'bouton_index': (((i+1)*2)-1),
                           'spine_index': ((i+1)*2),
                           'branchType': line[0],
                           'boutonType': line[1],
                           'spineMType': line[2],
                           'siteX': line[3],
                           'siteY': line[4],
                           'siteZ': line[5],
                           'siteR': line[6],
                           'period': line[7],
                           'boutonFileName': line[8],
                           'spineFileName': line[9],
                           'bouton_include': line[10]=='1',
                           'spine_include': line[11]=='1',
                           'synapse_include': line[12]=='1',
                           'orientation': line[13]})
        try:
            linesInhibitory = open(self.inhibitoryFileName).read().splitlines()
        except IOError:
            print "Could not open file to write! Please check !" +  self.spineFileName
        split_linesInhibitory = map(lambda x: x.split(' '), linesInhibitory)

        inputsInhibitory = []

        x = len(open(self.spineFileName).readlines())

        for i, line in enumerate(split_linesInhibitory[1:]):
            inputsInhibitory.append({'name': (i+x),
                                     'bouton_index': ((i+(2*x))-1),
                                     'branchType': line[0],
                                     'boutonType': line[1],
                                     'siteX': line[2],
                                     'siteY': line[3],
                                     'siteZ': line[4],
                                     'siteR': line[5],
                                     'period': line[6],
                                     'boutonFileName': line[7],
                                     'spineFileName': line[8],
                                     'bouton_include': line[9]=='1',
                                     'spine_include': line[10]=='1',
                                     'synapse_include': line[11]=='1'})
        #with open("model.gsl.template", "r") as file:
        #    gsl = file.read()
        #    newFile = open("model.gsl", "w")
        #    newFile.write(pystache.render(gsl, {'inputs': inputs, 'inputsInhibitory': inputsInhibitory}))
        #    newFile.close()
        with open("stimulus_model.gsl.template", "r") as file:
            gsl = file.read()
            targetFile = "./" + "stimulus_" + modelfile
            newFile = open(targetFile, "w")
            newFile.write(pystache.render(gsl, {'inputs': inputs,
                                                'inputsInhibitory': inputsInhibitory}))
            newFile.close()

        with open("recording_model.gsl.template", "r") as file:
            gsl = file.read()
            targetFile = "./" + "recording_" + modelfile
            newFile = open(targetFile, "w")
            newFile.write(pystache.render(gsl, {'inputs': inputs,
                                                'inputsInhibitory': inputsInhibitory}))
            newFile.close()

        with open("connect_model.gsl.template", "r") as file:
            gsl = file.read()
            targetFile = "./" + "connect_" + modelfile
            newFile = open(targetFile, "w")
            newFile.write(pystache.render(gsl, {'inputs': inputs,
                                                'inputsInhibitory': inputsInhibitory}))

if __name__ == '__main__':
    genSpine = SomeClass('neurons/neuron_test.swc')
