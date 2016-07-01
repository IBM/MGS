__author__ = "Hoang Trong Minh Tuan"
__copyright__ = "Copyright 2016, IBM"
__credits__ = []  # list of people who file pugs
__version__ = "0.1"
__maintainer__ = "Hoang Trong Minh Tuan"
__email__ = "tmhoangt@us.ibm.com"
__status__ = "Prototype"  # (Prototype, Development, Production)
import pdb
import itertools
import pandas as pd
import numpy as np
import numpy.random as rnd
import pystache
import sys
from math import *
import sympy as sp
import time

from sympy import Point3D
from sympy.abc import L
from sympy import Line3D, Segment3D
import glob

sys.setrecursionlimit(100000)
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

cwd = os.path.dirname(os.path.realpath(__file__))

branchType = {"soma": 1, "axon": 2, "basal": 3, "apical": 4,
              "AIS": 5, "tufted": 6,"bouton":7 }
def execute(command):
  # print(command)
  print(os.popen('cd ' + cwd + '; ' + command).read())
def getSpineVector(dx, dy, dz, orientation):
  angle = (2*pi/5) * ((2*orientation) % 5)
  v1 = [dx,dy,dz] # dendrite vector
  v2 = [dy,dz,dx] # not the dendrite vector
  v3 = np.cross(v1,v2) # perpendicular to the dendrite vector
  v4 = rotateVector(v3,angle,v1) # unit vector for bouton and spine
  return normaliseVector(v4)
def rotateVector(v,theta,axis):
  return np.dot(rotationMatrix(axis,theta), v)
def rotationMatrix(axis, theta):
  axis = normaliseVector(np.asarray(axis))
  theta = np.asarray(theta)
  a = cos(theta/2.0)
  b, c, d = -axis*sin(theta/2.0)
  aa, bb, cc, dd = a*a, b*b, c*c, d*d
  bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
  return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                   [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                   [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
def normaliseVector(v1):
  return v1/sqrt(np.dot(v1, v1))

def splitLine(line):
  return [x for x in line.split(' ') if len(x) > 0]
def readLines(filename):
  lines = open(filename).read().splitlines()
  all_lines_split = map(splitLine, lines)
  return [line for line in all_lines_split[1:] if line[0][0] != '#']

def renderFile(templateFile, outputFile, content):
  with open(templateFile, "r") as file:
    newFile = open(outputFile, "w")
    newFile.write(pystache.render(file.read(), content))
    newFile.close()
  return

def getNeuronStatisticsFromTissueFile(pathToFile='./neurons.txt', ignoreCommentedLine=True):
  """
  Return the statistics (surface area, volume ...)
  from a given neuron based on the morphology
  Spines can also be added by using tissue text file
  """
  #pathToFile = './neurons.txt'
  try:
      # file object
      myfile = open(pathToFile, "r+")
      # or "a+", whatever you need
  except IOError:
      print "Could not open file! Please check " + pathToFile
  lines = myfile.read().splitlines()
  split_lines = map(lambda x: x.strip().split(' '), lines)
  #ignoreCommentedLine = True
  #ignoreCommentedLine = False
  start = 1
  #neuronStat = NeuronStatistics()
  print("IMPORTANT: It is assumed all .swc files belong to the same neuron")
  print(" So comment out those from presynaptic neurons")
  neuronStat = {}
  global type2Analyse
  type2Analyze = "all" # 'all', 'surfaceArea', 'volume'
  if (type2Analyze== 'all' or type=='surfaceArea'):
    neuronStat['surfaceArea'] = 0.0
  if (type2Analyze== 'all' or type=='volume'):
    neuronStat['volume'] = 0.0
  for line in split_lines[start:]:
    file = line[0].lstrip()
    if (ignoreCommentedLine and file[0] == '#'):
      # skip commented line
      continue
    else:
      if (file[0] == '#'):
        file = file[1:]
    #if (int(line[2]) == 0):
    #  # skip main neuron
    #  continue
    xoffset = 0.0
    yoffset = 0.0
    zoffset = 0.0
    if (line[7] == 'R'):
      xoffset = float(line[4])
      yoffset = float(line[5])
      zoffset = float(line[6])
    spine  = SomeClass(file, False)
    result = spine.getStatistics(type=type2Analyze)
    if (type2Analyze== 'all' or type=='surfaceArea'):
      neuronStat['surfaceArea'] += result['surfaceArea']
    if (type2Analyze== 'all' or type=='volume'):
      neuronStat['volume'] += result['volume']
  #print neuronStat
  #print 'TUAN'
  for key,val in neuronStat.items():
    print(key, ': ', val)

def getNeuronStatisticsFromSWC(pathToFile='./neurons/neurons.swc', ignoreCommentedLine=True):
  """
  Return the statistics (surface area, volume ...)
  from a given neuron based on the morphology
  """
  #ignoreCommentedLine = True
  #ignoreCommentedLine = False
  start = 1
  #neuronStat = NeuronStatistics()
  print("IMPORTANT: It is assumed all .swc files belong to the same neuron")
  print(" So comment out those from presynaptic neurons")
  neuronStat = {}
  global type2Analyse
  type2Analyze = "all" # 'all', 'surfaceArea', 'volume'
  if (type2Analyze== 'all' or type=='surfaceArea'):
    neuronStat['surfaceArea'] = 0.0
  if (type2Analyze== 'all' or type=='volume'):
    neuronStat['volume'] = 0.0
  spine  = SomeClass(pathToFile, False)
  branchTypes=[
#                      branchType['soma'],
#                      branchType['axon'],
                      branchType['basal'],
                      branchType['apical'],
#                      branchType['AIS'],
                      branchType['tufted'],
                      branchType['bouton'],
                      ]
  result = spine.getStatistics(type=type2Analyze, branchType2Find = branchTypes)
  if (type2Analyze== 'all' or type=='surfaceArea'):
    neuronStat['surfaceArea'] += result['surfaceArea']
  if (type2Analyze== 'all' or type=='volume'):
    neuronStat['volume'] += result['volume']
  #print neuronStat
  for key,val in neuronStat.items():
    print(key, ': ', val)

"""
NOTE: branchOrder being store starts from zero
"""
class SomeClass(object):
    branchType = {"soma": 1, "axon": 2, "basal": 3, "apical": 4,
                  "AIS": 5, "tufted": 6,"bouton":7 }
    # NOTE: The regular neuron is going to use '0' in MTYPE
    # so we should use a different MTYPE for spine and bouton
    # The different set of configuration that we can use
    #   spine="generic"= 1  ; bouton = "excitatory=2" or "inhibitory=3"
    spineType = {"generic": 2}
    boutonType = {"excitatory":1, "inhibitory":3}

    """
    spineType = {"thin": 2, "mush": 3}
    #boutonType = {"excitatory":4, "inhibitory":5}
    boutonTypeOnSpine = {"thin":4, "mush":5} #as a function of spine-type
    """
    numFields = 7   # number of fields in SWC file
    #REF: Data members
    # self.rotation_indices []
    # self.rotation_angles []
    # self.line_ids = []   a vector containing the index of every line (i.e.
    #                      the value of the first column of SWC file)
    # self.swc_filename = name of swc file
    #      swcFolder    = location of swc file
    #      line_ids = [id1, id2, ...] holding line-index
    #      point_lookup[id] = map{ 'type': branchType,
    #                               'siteX': x,
    #                               'siteY': y,
    #                               'siteZ': z,
    #                               'siteR': r,
    #                               'parent': ??,
    #                               'dist2soma': ??,
    #                               'dist2branchPoint': ??,
    #                               'branchOrder': ??,
    #                               'numChildren': ??
    #                            }
    # self.spineArray[] each element is a list of
    #                    (branchType, boutonType, spineType,
    #                     X, Y, Z, //location at which spine stems from
    #                         //bouton is created after spine
    #                     Rradius_of_stimulus,
    #                     period_of_stimulus,
    #                     boutonFileName, spineFileName,
    #                     isRecordBouton,
    #                     isRecordSpine,
    #                     isRecordSynapse,
    #                    )
    # self.inhibitoryArray[] each line same structure, except
    #                  spineType get value "N/A"
    #


    # def __new__(cls, *args, **kwargs):
    #   pass

    def __init__(self, x, verbose=True):
        """
        GOAL: accept the path+name of .swc file
             and then parse it (self.parse_file() method)
        Pass in the path and file name of .swc file
        @param x = .swc filename (including path)

        """
        self.verbose = verbose
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
        self.swcFolder = os.path.dirname(x)
        self.parse_file()
        if (verbose):
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
                self.genboutonspineSWCFiles_PL5b(folder)
                self.genboutonspineSWCFiles_PL2_3(folder)
                5. generate the neurons.txt/tissue.txt file
                self.genTissueText()
                6. generate the GSL component files
                self.genModelGSL()

                """)


    def convertBranch(self, startIndex, newBranch, write2File=True, fileSuffix="_changeBranch.swc"):
      """
      Convert all points starting from the given 'startIndex'
      to the new branchType 'newBranch'
      NOTE: Some swc files does not discriminate between apical and basal den
            So this function can be useful
      """
      lines2Change = []
      lines2Change.extend(startIndex)
      tmpPointLookup = deepcopy(self.point_lookup)
      lineArray = []
      index = 0
      mapNewId = {}
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          x =self.point_lookup[id]['siteX']
          y =self.point_lookup[id]['siteY']
          z =self.point_lookup[id]['siteZ']
          r =self.point_lookup[id]['siteR']
          dist2soma= self.point_lookup[id]['dist2soma']
          dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
          branchOrder = self.point_lookup[id]['branchOrder']
          numChildren = self.point_lookup[id]['numChildren']
          if ((int(id) in lines2Change) or
             (int(parent_id) in lines2Change)):
            # start to delete from this line
            lines2Change.append(int(id))
            brType = newBranch
            tmpPointLookup[str(id)] = {'type': brType,
                                     'siteX': x,
                                     'siteY': y,
                                     'siteZ': z,
                                     'siteR': r, 'parent': str(parent_id),
                                     'dist2soma': dist2soma,
                                     'dist2branchPoint': dist2branchPoint,
                                     'branchOrder': branchOrder,
                                     'numChildren': numChildren}
          lineArray.append([str(id), str(brType),
                          str(x), str(y),
                          str(z), str(r), str(parent_id)])
      self.point_lookup = tmpPointLookup
      lineArray = np.asarray(lineArray)
      if (write2File):
        PL5bFileName = self.swc_filename+fileSuffix
        np.savetxt(PL5bFileName, lineArray, fmt='%s')
        print("Write to file: ", PL5bFileName)


    def genPL5b(self):
      """
      1. convert multiple-point soma to 1point
      2. remove nearbyPoint
      3. convert to tufted region
      """
      pass
      self.reviseSomaSWCFile(write2File=False)
      dist = 0.0
      #self.removeNearbyPoints(dist, write2File=True,fileSuffix='_revised.swc')
      self.removeNearbyPoints(dist, write2File=False)
      self.convertToTufted()


    def convertToTufted(self):
        """
        Convert a region of distal apical dendrites to thick tufted dendrite
        NOTE:
            thresholdDistance = 600.0 um
        Convert a region of axon to AIS
        NOTE:
            proximalAISDistance2Soma = 23.0 um
            distalAISDistance2Soma = 50 um
        """
        #Put branchType = 5 to represent the region of high CaLVA, CaHVA
        PL5bFileName = self.swc_filename+"_new.swc"
        SWCFileSpine = open(PL5bFileName, "w")

        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x_soma =float(self.point_lookup[id]['siteX'])
            y_soma =float(self.point_lookup[id]['siteY'])
            z_soma =float(self.point_lookup[id]['siteZ'])
            r_soma =float(self.point_lookup[id]['siteR'])
            if (int(parent_id) == -1):
              break

        thresholdTuftedZone = 550.0 + r_soma # [um]
        #proximalAISDistance2Soma = 20.0 # [um]
        proximalAISDistance2Soma = 10.0 + r_soma # [um]
        distalAISDistance2Soma = 45.0 + r_soma # [um]  - before the onset of myelination
        lineArray = []
        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x =float(self.point_lookup[id]['siteX'])
            y =float(self.point_lookup[id]['siteY'])
            z =float(self.point_lookup[id]['siteZ'])
            r =self.point_lookup[id]['siteR']
            dist2soma= self.point_lookup[id]['dist2soma']
            ## NOTE: we use direct distance
            dist2soma_straight = sqrt((x_soma - x)**2 +
                        (y_soma - y)**2 + (z_soma - z)**2)

            if (float(dist2soma_straight) > thresholdTuftedZone and int(brType) == self.branchType["apical"]):
                brType = self.branchType["tufted"]
            elif (float(dist2soma) <= distalAISDistance2Soma
                and float(dist2soma) >= proximalAISDistance2Soma
                and int(brType) == self.branchType["axon"]):
                brType = self.branchType["AIS"]
                if (float(dist2soma) >= proximalAISDistance2Soma and
                    float(dist2soma) < (proximalAISDistance2Soma)+13.0):
                  print(dist2soma, x,y,z)

            lineArray.append([id, str(brType),
                            str(x), str(y),
                            str(z), str(r), str(parent_id)])
        lineArray = np.asarray(lineArray)
        np.savetxt(PL5bFileName, lineArray, fmt='%s')
        print("Write to file: ", PL5bFileName)

    def checkSiteApical(self, startDist=615.0, endDist=625.0):
        """
        This is useful for find the recording site or stimulus site
        Return the point on apical den at a given distance to soma
        """
        #thresholdfrom = 615.0 # [um]
        #thresholdto = 625.0 # [um]
        thresholdfrom = startDist # [um]
        thresholdto = endDist # [um]
        print("Display points on the tree (apical dendrite) whose")
        print("coordinate fall within: [", thresholdfrom, ',', thresholdto, ']')
        print('---> dist2soma, x,y,z, r : ')
        maxR = 0.0
        point = []
        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x = self.point_lookup[id]['siteX']
            y = self.point_lookup[id]['siteY']
            z = self.point_lookup[id]['siteZ']
            r = self.point_lookup[id]['siteR']
            dist2soma= self.point_lookup[id]['dist2soma']
            dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
            ## parent
            if (int(parent_id) == -1):
                continue
            parent_x1 = self.point_lookup[parent_id]['siteX']
            parent_y1 = self.point_lookup[parent_id]['siteY']
            parent_z1 = self.point_lookup[parent_id]['siteZ']
            parent_dist2soma= self.point_lookup[parent_id]['dist2soma']
            parent_dist2branchPoint = \
                float(self.point_lookup[parent_id]['dist2branchPoint'])
            ## halfway to previous point
            gap = dist2soma - parent_dist2soma
            dis2points = self.find_distance(id, parent_id)
            halfwayX = np.interp(gap-gap/2.0, [0, dis2points], [parent_x1, x])
            halfwayY = np.interp(gap-gap/2.0, [0, dis2points], [parent_y1, y])
            halfwayZ = np.interp(gap-gap/2.0, [0, dis2points], [parent_z1, z])

            if (float(dist2soma) > thresholdfrom and
                float(dist2soma) < thresholdto and
                (int(brType) == self.branchType["apical"] or
                int(brType) == self.branchType["tufted"]
                )):
                print(dist2soma, x,y,z, r)
                print('... its halfway to proximal is (dist,coordinates):')
                print('    ', dist2soma-gap/2, halfwayX, halfwayY, halfwayZ)
                if (float(r) > maxR):
                    maxR= float(r)
                    point =[dist2soma, x,y,z,r]
        ##Point with largest radius (mostlikely the main branch)
        print("Apical Final: ", point)

    def checkSiteBasal(self, startDist=615.0, endDist=625.0):
        """
        This is useful for find the recording site or stimulus site
        Return the point on basal den at a given distance to soma
        """
        #thresholdfrom = 615.0 # [um]
        #thresholdto = 625.0 # [um]
        thresholdfrom = startDist # [um]
        thresholdto = endDist # [um]
        print("Display points on the tree (apical dendrite) whose")
        print("coordinate fall within: [", thresholdfrom, ',', thresholdto, ']')
        print('---> dist2soma, x,y,z, r : ')
        maxR = 0.0
        point = []
        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x = self.point_lookup[id]['siteX']
            y = self.point_lookup[id]['siteY']
            z = self.point_lookup[id]['siteZ']
            r = self.point_lookup[id]['siteR']
            dist2soma= self.point_lookup[id]['dist2soma']
            dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
            ## parent
            if (int(parent_id) == -1):
                continue
            parent_x1 = self.point_lookup[parent_id]['siteX']
            parent_y1 = self.point_lookup[parent_id]['siteY']
            parent_z1 = self.point_lookup[parent_id]['siteZ']
            parent_dist2soma= self.point_lookup[parent_id]['dist2soma']
            parent_dist2branchPoint = \
                float(self.point_lookup[parent_id]['dist2branchPoint'])
            ## halfway to previous point
            gap = dist2soma - parent_dist2soma
            dis2points = self.find_distance(id, parent_id)
            halfwayX = np.interp(gap-gap/2.0, [0, dis2points], [parent_x1, x])
            halfwayY = np.interp(gap-gap/2.0, [0, dis2points], [parent_y1, y])
            halfwayZ = np.interp(gap-gap/2.0, [0, dis2points], [parent_z1, z])

            if (float(dist2soma) > thresholdfrom and
                float(dist2soma) < thresholdto and
                (int(brType) == self.branchType["basal"]
                )):
                print(dist2soma, x,y,z, r)
                print('... its halfway to proximal is (dist,coordinates):')
                print('    ', dist2soma-gap/2, halfwayX, halfwayY, halfwayZ)
                if (float(r) > maxR):
                    maxR= float(r)
                    point =[dist2soma, x,y,z,r]
        ##Point with largest radius (mostlikely the main branch)
        print("Basal Final: ", point)

    def checkSiteAxon(self, startDist=615.0, endDist=625.0):
        """
        This is useful for find the recording site or stimulus site
        Return the point on basal den at a given distance to soma
        """
        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x_soma =float(self.point_lookup[id]['siteX'])
            y_soma =float(self.point_lookup[id]['siteY'])
            z_soma =float(self.point_lookup[id]['siteZ'])
            r_soma =float(self.point_lookup[id]['siteR'])
            if (int(parent_id) == -1):
              break
        #thresholdfrom = 615.0 # [um]
        #thresholdto = 625.0 # [um]
        #thresholdfrom = startDist # [um]
        #thresholdto = endDist # [um]
        thresholdfrom = startDist + r_soma # [um]
        thresholdto = endDist + r_soma # [um]
        print("Display points on the tree (apical dendrite) whose")
        print("coordinate fall within: [", thresholdfrom, ',', thresholdto, ']')
        print('---> dist2soma, x,y,z, r : ')
        maxR = 0.0
        point = []
        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x = self.point_lookup[id]['siteX']
            y = self.point_lookup[id]['siteY']
            z = self.point_lookup[id]['siteZ']
            r = self.point_lookup[id]['siteR']
            dist2soma= self.point_lookup[id]['dist2soma']
            dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
            ## parent
            if (int(parent_id) == -1):
                continue
            parent_x1 = self.point_lookup[parent_id]['siteX']
            parent_y1 = self.point_lookup[parent_id]['siteY']
            parent_z1 = self.point_lookup[parent_id]['siteZ']
            parent_dist2soma= self.point_lookup[parent_id]['dist2soma']
            parent_dist2branchPoint = \
                float(self.point_lookup[parent_id]['dist2branchPoint'])
            ## halfway to previous point
            gap = dist2soma - parent_dist2soma
            dis2points = self.find_distance(id, parent_id)
            halfwayX = np.interp(gap-gap/2.0, [0, dis2points], [parent_x1, x])
            halfwayY = np.interp(gap-gap/2.0, [0, dis2points], [parent_y1, y])
            halfwayZ = np.interp(gap-gap/2.0, [0, dis2points], [parent_z1, z])

            if (float(dist2soma) > thresholdfrom and
                float(dist2soma) < thresholdto and
                (int(brType) == self.branchType["axon"] or
                int(brType) == self.branchType["AIS"]
                )):
                print(dist2soma, x,y,z, r)
                print('... its halfway to proximal is (dist,coordinates):')
                print('    ', dist2soma-gap/2, halfwayX, halfwayY, halfwayZ)
                if (float(r) > maxR):
                    maxR= float(r)
                    #point =[dist2soma, x,y,z,r]
                    point =[dist2soma-r_soma, x,y,z,r]
        ##Point with largest radius (mostlikely the main branch)
        print("Axon/AIS Final: ", point)


    def getDist2Soma(self, site):
      """
      This is useful for find the approximate distance form a given
      recording site or stimulus site to the soma
      """
      px = float(site[0])
      py = float(site[1])
      pz = float(site[2])
      pr = float(site[3])
      print('dist2soma, x,y,z : ')
      for ix in range(len(self.point_lookup)):
        id = str(ix+1)
        parent_id = self.point_lookup[id]['parent']
        brType =self.point_lookup[id]['type']
        x = float(self.point_lookup[id]['siteX'])
        y = float(self.point_lookup[id]['siteY'])
        z = float(self.point_lookup[id]['siteZ'])
        r = float(self.point_lookup[id]['siteR'])
        dist2soma= self.point_lookup[id]['dist2soma']
        distance = sqrt((x - px)**2 +
                    (y - py)**2 + (z - pz)**2)
        if (distance  <= pr):
          print(dist2soma, x,y,z)

    def getCapsuleWithGivenDist2Soma(self, distance, tolerance=2.0):
      """
      This is useful for find the capsule on the neuron tree with
      dist2soma closed to the given criteria
      'distance'  = in micrometer
      tolerance = in micrometer
      """
      print ("Criteria: dist2soma = ", distance, "; tolerance = ", tolerance, "[um]")
      print('branchType, dist2soma, x,y,z : ')
      for ix in range(len(self.point_lookup)):
        id = str(ix+1)
        parent_id = self.point_lookup[id]['parent']
        brType =self.point_lookup[id]['type']
        x = float(self.point_lookup[id]['siteX'])
        y = float(self.point_lookup[id]['siteY'])
        z = float(self.point_lookup[id]['siteZ'])
        r = float(self.point_lookup[id]['siteR'])
        dist2soma= self.point_lookup[id]['dist2soma']
        if (abs(distance-dist2soma)  <= abs(tolerance)):
          print(brType, dist2soma, x,y,z)

    def withSomaInside(self, x,y,z,r, xoffset=0.0, yoffset=0.0, zoffset=0.0):
      """
      Return TRUE if the soma falls within the given sphere defined by
      (x,y,z,radius)
      """
      center=[]
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          px =float(self.point_lookup[id]['siteX']) + xoffset
          py =float(self.point_lookup[id]['siteY']) + yoffset
          pz =float(self.point_lookup[id]['siteZ']) + zoffset
          pr =float(self.point_lookup[id]['siteR'])
          dist2soma= self.point_lookup[id]['dist2soma']
          if (int(brType) == self.branchType["soma"] ):
            distance = sqrt((x - px)**2 +
                        (y - py)**2 + (z - pz)**2)
            if (distance  <= r):
              center = [px, py, pz]
              return [True, center]

      return [False, center]

    def getIndexOfFileInTissueFile(self, filename, ignoreCommentedLine=True):
      """
      Return the file name (*.swc) correspond to spine structure
      that has the coordinate of spine head falls inside
      the given site (x,y,z,r)
      """
      pathTofiles = './neurons.txt'
      try:
          # file object
          myfile = open(pathTofiles, "r+")
          # or "a+", whatever you need
      except IOError:
          print "Could not open file! Please check " + pathTofiles
      lines = myfile.read().splitlines()
      split_lines = map(lambda x: x.strip().split(' '), lines)
      #ignoreCommentedLine = True
      #ignoreCommentedLine = False
      start = 1
      index = -1
      for line in split_lines[start:]:
        file = line[0].lstrip()
        if (ignoreCommentedLine and file[0] == '#'):
          # skip commented line
          continue
        else:
          if (file[0] == '#'):
            file = file[1:]
        index = index+1
        if (file == filename):
          print(filename, index)

    def getSpineHeadFromTissueFile(self, x,y,z, r, ignoreCommentedLine=True):
      """
      Return the file name (*.swc) correspond to spine structure
      that has the coordinate of spine head falls inside
      the given site (x,y,z,r)
      """
      pathTofiles = './neurons.txt'
      try:
          # file object
          myfile = open(pathTofiles, "r+")
          # or "a+", whatever you need
      except IOError:
          print "Could not open file! Please check " + pathTofiles
      lines = myfile.read().splitlines()
      split_lines = map(lambda x: x.strip().split(' '), lines)
      #ignoreCommentedLine = True
      #ignoreCommentedLine = False
      start = 1
      for line in split_lines[start:]:
        file = line[0].lstrip()
        if (ignoreCommentedLine and file[0] == '#'):
          # skip commented line
          continue
        else:
          if (file[0] == '#'):
            file = file[1:]
        if (int(line[2]) == 0):
          # skip main neuron
          continue
        xoffset = 0.0
        yoffset = 0.0
        zoffset = 0.0
        if (line[7] == 'R'):
          xoffset = float(line[4])
          yoffset = float(line[5])
          zoffset = float(line[6])
        spine  = SomeClass(file, False)
        [result, center] = spine.withSomaInside(x,y,z,r, xoffset, yoffset, zoffset)
        # if (spine.withSomaInside(x,y,z,r, xoffset, yoffset, zoffset)):
        if (result):
          print (file, 'head center: ', center)

    def reviseSWCFile(self):
      """
      1. remove 2 points with same coordinate
      2. convert multiple point soma into 1point

      """
      self.reviseSomaSWCFile(write2File=False)
      dist = 0.0
      self.removeNearbyPoints(dist, write2File=True,fileSuffix='_revised.swc')

    def removeNearbyPoints(self, dist_criteria=0.0, write2File=True,
                           fileSuffix='_trimmedClosedPoints.swc'):
      """
      Remove points that are too closed to another one based on
      the given 'dist_criteria' distance criteria
      NOTE:
        we can revise this
      """
      listDuplicatedPoints = []
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          x =float(self.point_lookup[id]['siteX'])
          y =float(self.point_lookup[id]['siteY'])
          z =float(self.point_lookup[id]['siteZ'])
          r =float(self.point_lookup[id]['siteR'])
          if (int(parent_id) != -1):
            parent_info = self.point_lookup[parent_id]
            parent_x = float(parent_info['siteX'])
            parent_y = float(parent_info['siteY'])
            parent_z = float(parent_info['siteZ'])
            dist2soma= self.point_lookup[id]['dist2soma']
            distance = sqrt((x - parent_x)**2 + (y - parent_y)**2 +
                            (z - parent_z)**2)
            if (distance <= dist_criteria):
              listDuplicatedPoints.append(int(parent_id))
            #print(id, x,y,z,parent_id)
      self.listDuplicatedPoints= list(set(listDuplicatedPoints))
      ##
      #Step 2: delete these points and update others
      ##....
      lines2Delete = []
      lines2Delete.extend(self.listDuplicatedPoints)
      lineArray = []
      index = 0
      mapNewId = {}
      tmpPointLookup = deepcopy(self.point_lookup)
      print("Before delete: ", len(self.point_lookup), " points")
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          x =self.point_lookup[id]['siteX']
          y =self.point_lookup[id]['siteY']
          z =self.point_lookup[id]['siteZ']
          r =self.point_lookup[id]['siteR']
          dist2soma= self.point_lookup[id]['dist2soma']
          dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
          branchOrder = self.point_lookup[id]['branchOrder']
          numChildren = self.point_lookup[id]['numChildren']
          if (int(id) in lines2Delete):
            # start to delete from this line
            del tmpPointLookup[id]
            mapNewId[id] = parent_id   # important if we delete intermediate points
            #print("line deleted ", id, "its parent: ", parent_id)
          else:
            index += 1
            mapNewId[id] = index
            if (parent_id in mapNewId):
              parent_id = mapNewId[parent_id]
            lineArray.append([str(index), str(brType),
                            str(x), str(y),
                            str(z), str(r), str(parent_id)])
            tmpPointLookup[str(index)] = {'type': brType,
                                     'siteX': x,
                                     'siteY': y,
                                     'siteZ': z,
                                     'siteR': r, 'parent': str(parent_id),
                                     'dist2soma': dist2soma,
                                     'dist2branchPoint': dist2branchPoint,
                                     'branchOrder': branchOrder,
                                     'numChildren': numChildren}
      for ix in range(index, len(self.point_lookup)):
        id = str(ix+1)
        del tmpPointLookup[id]
        #print ix

      print("After delete: ", len(tmpPointLookup), " points")
      self.point_lookup = tmpPointLookup
      lineArray = np.asarray(lineArray)
      if (write2File):
        PL5bFileName = self.swc_filename+fileSuffix
        np.savetxt(PL5bFileName, lineArray, fmt='%s')
        print("Write to file: ", PL5bFileName)

    def removeGivenPoints(self, lineIndex, write2File=True,
                           fileSuffix='_trimmedPoints.swc'):
      """
      Remove one or many points based on the given line index
      Typically these are points closed to soma of small capsule length
      Its parent point is taken over
      """
      ##
      #Step 1: delete these points and update others
      ##....
      lines2Delete = []
      lines2Delete.extend(lineIndex)
      lineArray = []
      index = 0
      mapNewId = {}
      tmpPointLookup = deepcopy(self.point_lookup)
      print("Before delete: ", len(self.point_lookup), " points")
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          x =self.point_lookup[id]['siteX']
          y =self.point_lookup[id]['siteY']
          z =self.point_lookup[id]['siteZ']
          r =self.point_lookup[id]['siteR']
          dist2soma= self.point_lookup[id]['dist2soma']
          dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
          branchOrder = self.point_lookup[id]['branchOrder']
          numChildren = self.point_lookup[id]['numChildren']
          if (int(id) in lines2Delete):
            # start to delete from this line
            del tmpPointLookup[id]
            mapNewId[id] = parent_id   # important if we delete intermediate points
            #print("line deleted ", id, "its parent: ", parent_id)
          else:
            index += 1
            mapNewId[id] = index
            if (parent_id in mapNewId):
              parent_id = mapNewId[parent_id]
            lineArray.append([str(index), str(brType),
                            str(x), str(y),
                            str(z), str(r), str(parent_id)])
            tmpPointLookup[str(index)] = {'type': brType,
                                     'siteX': x,
                                     'siteY': y,
                                     'siteZ': z,
                                     'siteR': r, 'parent': str(parent_id),
                                     'dist2soma': dist2soma,
                                     'dist2branchPoint': dist2branchPoint,
                                     'branchOrder': branchOrder,
                                     'numChildren': numChildren}
      for ix in range(index, len(self.point_lookup)):
        id = str(ix+1)
        del tmpPointLookup[id]
        #print ix

      print("After delete: ", len(tmpPointLookup), " points")
      self.point_lookup = tmpPointLookup
      lineArray = np.asarray(lineArray)
      if (write2File):
        PL5bFileName = self.swc_filename+fileSuffix
        np.savetxt(PL5bFileName, lineArray, fmt='%s')
        print("Write to file: ", PL5bFileName)


    def reviseSomaSWCFile(self,write2File=True, fileSuffix="_revisedSoma.swc"):
      """
      Revise multiple points soma into a single point soma

      Ref:
        1. http://stackoverflow.com/questions/15785428/how-do-i-fit-3d-data
      """
      somaPoints = [] # set of soma points
      pX = 0.0
      pY = 0.0
      pZ = 0.0
      central = [0, 0, 0, 0] #(x,y,z,r)
      # find all soma points (ignore the first one, i.e. the line with parent -1)
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          if (int(brType) == self.branchType["soma"] and
             int(parent_id) != -1
             ):
            somaPoints.append(int(id))
            x =self.point_lookup[id]['siteX']
            y =self.point_lookup[id]['siteY']
            z =self.point_lookup[id]['siteZ']
            pX += float(x)
            pY += float(y)
            pZ += float(z)
          if (int(brType) == self.branchType["soma"] and
              int(parent_id) == -1):
            x =self.point_lookup[id]['siteX']
            y =self.point_lookup[id]['siteY']
            z =self.point_lookup[id]['siteZ']
            r =self.point_lookup[id]['siteR']
            central[0]= float(x)
            central[1]= float(y)
            central[2]= float(z)
            central[3]= float(r)
      print("soma points: ", somaPoints)
      print("total points: ", len(somaPoints))
      ####
      #Step 1: find the new center and radius for soma
      # Find the central point
      # .. which is the center of all these points
      if (len(somaPoints) > 0):
        central[0] = pX / len(somaPoints)
        central[1] = pY / len(somaPoints)
        central[2] = pZ / len(somaPoints)
      # Find the radius
      # .. which is the furthest distance for this point
      # we can keep other candidate (i.e. the shortest, the mean, and the max)
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          if (int(id) in somaPoints):
            x =float(self.point_lookup[id]['siteX'])
            y =float(self.point_lookup[id]['siteY'])
            z =float(self.point_lookup[id]['siteZ'])
            dist =  sqrt((x-central[0])**2 + (y-central[1])**2 + (z-central[2])**2)
            if (dist > central[3]):
                central[3] = dist


      #####
      #Step 2: find the point
      # remove points in somaPoints and update other lines
      lines2Delete = []
      lines2Delete.extend(somaPoints)
      tmpPointLookup = deepcopy(self.point_lookup)
      print("Before delete: ", len(self.point_lookup), " points")
      lineArray = []
      index = 0
      mapNewId = {}
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          x =self.point_lookup[id]['siteX']
          y =self.point_lookup[id]['siteY']
          z =self.point_lookup[id]['siteZ']
          r =self.point_lookup[id]['siteR']
          dist2soma= self.point_lookup[id]['dist2soma']
          dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
          branchOrder = self.point_lookup[id]['branchOrder']
          numChildren = self.point_lookup[id]['numChildren']
          if (int(parent_id) == -1):
              # update x,y,z
              x = round(central[0],3)
              y = round(central[1],3)
              z = round(central[2],3)
              r = round(central[3],3)

          if ((int(id) in lines2Delete)
             ):
            # start to delete from this line
            del tmpPointLookup[id]
            mapNewId[id] = parent_id   # important if we delete intermediate points
          else:
            index += 1
            mapNewId[id] = index
            if (int(parent_id) in somaPoints):
              parent_id = 1
            else:
              if (parent_id in mapNewId):
                parent_id = mapNewId[parent_id]
            lineArray.append([str(index), str(brType),
                            str(x), str(y),
                            str(z), str(r), str(parent_id)])
            tmpPointLookup[str(index)] = {'type': brType,
                                     'siteX': str(x),
                                     'siteY': str(y),
                                     'siteZ': str(z),
                                     'siteR': str(r), 'parent': str(parent_id),
                                     'dist2soma': dist2soma,
                                     'dist2branchPoint': dist2branchPoint,
                                     'branchOrder': branchOrder,
                                     'numChildren': numChildren}
      for ix in range(index, len(self.point_lookup)):
        id = str(ix+1)
        del tmpPointLookup[id]
        #print ix

      print("After delete: ", len(tmpPointLookup), " points")
      self.point_lookup = deepcopy(tmpPointLookup)
      lineArray = np.asarray(lineArray)
      if (write2File):
        PL5bFileName = self.swc_filename+fileSuffix
        np.savetxt(PL5bFileName, lineArray, fmt='%s')
        print("Write to file: ", PL5bFileName)
        print("IMPORTANT: Please revise the coordinate and radius of the soma point")

    def removeBranch(self, lineIndex, write2File=True, fileSuffix="_trimmed.swc"):
      """
      Remove a branch starting from the given line index
      """
      lines2Delete = []
      lines2Delete.extend(lineIndex)
      tmpPointLookup = deepcopy(self.point_lookup)
      print("Before delete: ", len(self.point_lookup), " points")
      lineArray = []
      index = 0
      mapNewId = {}
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          x =self.point_lookup[id]['siteX']
          y =self.point_lookup[id]['siteY']
          z =self.point_lookup[id]['siteZ']
          r =self.point_lookup[id]['siteR']
          dist2soma= self.point_lookup[id]['dist2soma']
          dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
          branchOrder = self.point_lookup[id]['branchOrder']
          numChildren = self.point_lookup[id]['numChildren']
          if ((int(id) in lines2Delete) or
             (int(parent_id) in lines2Delete)):
            # start to delete from this line
            del tmpPointLookup[id]
            lines2Delete.append(int(id))
          else:
            index += 1
            mapNewId[id] = index
            if (parent_id in mapNewId):
              parent_id = mapNewId[parent_id]
            lineArray.append([str(index), str(brType),
                            str(x), str(y),
                            str(z), str(r), str(parent_id)])
            tmpPointLookup[str(index)] = {'type': brType,
                                     'siteX': x,
                                     'siteY': y,
                                     'siteZ': z,
                                     'siteR': r, 'parent': str(parent_id),
                                     'dist2soma': dist2soma,
                                     'dist2branchPoint': dist2branchPoint,
                                     'branchOrder': branchOrder,
                                     'numChildren': numChildren}

      print("After delete: ", len(tmpPointLookup), " points")
      self.point_lookup = tmpPointLookup
      #idx = 0
      #for ix in range(len(tmpPointLookup)):
      #    idx += 1
      #    id = str(ix+1)
      #    parent_id = tmpPointLookup[id]['parent']
      #    x =tmpPointLookup[id]['siteX']
      #    y =tmpPointLookup[id]['siteY']
      #    z =tmpPointLookup[id]['siteZ']
      #    lineArray.append([id, str(brType),
      #                    str(x), str(y),
      #                    str(z), str(r), str(parent_id)])


      #    lineArray.append([id, str(brType),
      #                    str(x), str(y),
      #                    str(z), str(r), str(parent_id)])
      lineArray = np.asarray(lineArray)
      if (write2File):
        PL5bFileName = self.swc_filename+fileSuffix
        np.savetxt(PL5bFileName, lineArray, fmt='%s')
        print("Write to file: ", PL5bFileName)

    def removeTerminalPoints(self,write2File=True, fileSuffix="_trimmedTerminalPoints.swc"):
      """
      Remove all points at terminal (hopefully to remove points of small volumes
      after resampling)
      """
      tmpPointLookup = deepcopy(self.point_lookup)
      print("Before delete: ", len(self.point_lookup), " points")
      branchpoint_list = []
      # start with all distal ends to examine toward soma
      for id in self.line_ids:
          numChildren = int(self.point_lookup[id]['numChildren'])
          if numChildren == 0:
              branchpoint_list.append(int(id))

      branchpoint_list = list(set(branchpoint_list))
      lines2Delete = []
      lines2Delete.extend(branchpoint_list)
      # start
      lineArray = []
      index = 0
      mapNewId = {}
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          x =self.point_lookup[id]['siteX']
          y =self.point_lookup[id]['siteY']
          z =self.point_lookup[id]['siteZ']
          r =self.point_lookup[id]['siteR']
          dist2soma= self.point_lookup[id]['dist2soma']
          dist2branchPoint = self.point_lookup[id]['dist2branchPoint']
          branchOrder = self.point_lookup[id]['branchOrder']
          numChildren = self.point_lookup[id]['numChildren']
          if (int(id) in lines2Delete):
            # start to delete from this line
            del tmpPointLookup[id]
            mapNewId[id] = parent_id   # important if we delete intermediate points
          else:
            index += 1
            mapNewId[id] = index
            if (parent_id in mapNewId):
              parent_id = mapNewId[parent_id]
            lineArray.append([str(index), str(brType),
                            str(x), str(y),
                            str(z), str(r), str(parent_id)])
            tmpPointLookup[str(index)] = {'type': brType,
                                     'siteX': x,
                                     'siteY': y,
                                     'siteZ': z,
                                     'siteR': r, 'parent': str(parent_id),
                                     'dist2soma': dist2soma,
                                     'dist2branchPoint': dist2branchPoint,
                                     'branchOrder': branchOrder,
                                     'numChildren': numChildren}

      print("After delete: ", len(tmpPointLookup), " points")
      self.point_lookup = tmpPointLookup
      lineArray = np.asarray(lineArray)
      if (write2File):
        PL5bFileName = self.swc_filename+"_trimmedTerminalPoints.swc"
        np.savetxt(PL5bFileName, lineArray, fmt='%s')
        print("Write to file: ", PL5bFileName)


    def getSpineHeadFromTissueFileAndMark(self, x,y,z, r, ignoreCommentedLine=True):
      """
      Return the file name (*.swc) correspond to spine structure
      that has the coordinate of spine head falls inside
      the given site (x,y,z,r)
      """
      pathTofiles = './neurons.txt'
      pathToNewFile = pathTofiles + '_new'
      try:
          # file object
          myfile = open(pathTofiles, "r+")
          # or "a+", whatever you need
      except IOError:
          print "Could not open file! Please check " + pathTofiles
      lines = myfile.read().splitlines()
      newlines = deepcopy(lines)
      split_lines = map(lambda x: x.strip().split(' '), lines)
      #ignoreCommentedLine = True
      #ignoreCommentedLine = False
      start = 1
      index = start-1
      count = 0
      for line in split_lines[start:]:
        index += 1
        file = line[0].lstrip()
        if (ignoreCommentedLine and file[0] == '#'):
          # skip commented line
          continue
        else:
          if (file[0] == '#'):
            file = file[1:]
        if (int(line[2]) == 0):
          # skip main neuron
          continue
        xoffset = 0.0
        yoffset = 0.0
        zoffset = 0.0
        if (line[7] == 'R'):
          xoffset = float(line[4])
          yoffset = float(line[5])
          zoffset = float(line[6])
        spine  = SomeClass(file, False)
        [result, center] = spine.withSomaInside(x,y,z,r, xoffset, yoffset, zoffset)
        # if (spine.withSomaInside(x,y,z,r, xoffset, yoffset, zoffset)):
        if (result):
          newlines[index-1] = "x" + newlines[index-1]
          count += 1
          #print(newlines[index-1])
          #print (file, 'head center: ', center)
      print ("Therea re total ", count, " spines got triggered")
      myfile = open(pathToNewFile, "w")
      for item in newlines:
        myfile.write("%s\n" % item)

    def getStatistics(self, type='all', branchType2Find=[
                                                    branchType['soma'],
                                                    branchType['axon'],
                                                    branchType['basal'],
                                                    branchType['apical'],
                                                    branchType['AIS'],
                                                    branchType['tufted'],
                                                    branchType['bouton'],
                                                    ]):
      """
      This is used to find statistics for a given neuron
      type = 'all', 'surfaceArea', 'volume'

      If spine neck:
          subtract the part covered by the spine neck on the shaft
      """
      result = {}
      if (type == 'all' or type=='surfaceArea'):
        result['surfaceArea'] = 0.0
      if (type == 'all' or type=='volume'):
        result['volume'] = 0.0
      for ix in range(len(self.point_lookup)):
        id = str(ix+1)
        parent_id = self.point_lookup[id]['parent']
        brType =self.point_lookup[id]['type']
        x = float(self.point_lookup[id]['siteX'])
        y = float(self.point_lookup[id]['siteY'])
        z = float(self.point_lookup[id]['siteZ'])
        r = float(self.point_lookup[id]['siteR'])
        dist2soma= self.point_lookup[id]['dist2soma']
        if (not int(brType) in branchType2Find):
          continue
        parent_dist2soma = 0.0
        new_r = r
        if (int(parent_id) != -1):
          #not the soma
          parent_dist2soma == self.point_lookup[parent_id]['dist2soma']
          lenSeg = float(dist2soma) - float(parent_dist2soma)
          assert(lenSeg >= 0.0)
          parent_brType = self.point_lookup[parent_id]['type']
          if (parent_brType == self.branchType['soma']):
            new_r = r
          else:
            parent_r = float(self.point_lookup[parent_id]['siteR'])
            new_r = (r + parent_r)/2.0
        else:
          lenSeg = 0
        if (int(brType) == self.branchType["soma"] ):
          if (type == 'all' or type=='surfaceArea'):
            result['surfaceArea'] += 4 * pi * new_r * new_r
          if (type == 'all' or type=='volume'):
            result['volume'] += 4.0/3.0 * pi * new_r * new_r * new_r
        else :
          if (type == 'all' or type=='surfaceArea'):
            result['surfaceArea'] +=  2 * pi * new_r * lenSeg
          if (type == 'all' or type=='volume'):
            result['volume'] +=  pi * new_r * new_r * lenSeg
      #print result
      #time.sleep(10)
      #sys.exit
      return result


    def getSpineHeadFromFolder(self, x,y,z, r):
      """
      Return the file name (*.swc) correspond to spine structure
      that has the coordinate of spine head falls inside
      the given site (x,y,z,r)
      """
      pathTofiles = './spines/s*.swc'
      for file in glob.glob(pathTofiles):
        spine  = SomeClass(file, False)
        if (spine.withSomaInside(x,y,z,r)):
          print (file)

    def getPreSynapticSomaHeadFromTissueFile(self, x,y,z, r, ignoreCommentedLine=True):
      """
      Return the file name (*.swc) correspond to presynaptic structure
      that has the coordinate of associated spine head falls inside
      the given site (x,y,z,r)
      """
      pathTofiles = './neurons.txt'
      try:
          # file object
          myfile = open(pathTofiles, "r+")
          # or "a+", whatever you need
      except IOError:
          print "Could not open file! Please check " + pathTofiles
      lines = myfile.read().splitlines()
      split_lines = map(lambda x: x.strip().split(' '), lines)
      #ignoreCommentedLine = True
      #ignoreCommentedLine = False
      start = 1
      for line in split_lines[start:]:
        file = line[0].lstrip()
        if (ignoreCommentedLine and file[0] == '#'):
          # skip commented line
          continue
        else:
          if (file[0] == '#'):
            file = file[1:]
        if (int(line[2]) == 0):
          # skip main neuron
          continue
        boutonMType = line[2]
        xoffset = 0.0
        yoffset = 0.0
        zoffset = 0.0
        if (line[7] == 'R'):
          xoffset = float(line[4])
          yoffset = float(line[5])
          zoffset = float(line[6])
        spine  = SomeClass(file, False)
        # if (spine.withSomaInside(x,y,z,r, xoffset, yoffset, zoffset)):
        if (spine.withAxonInside(x,y,z,r, xoffset, yoffset, zoffset)):
          print (file, ", boutonMType: ", boutonMType)

    def withAxonInside(self, x,y,z,r, xoffset=0.0, yoffset=0.0, zoffset=0.0):
      """
      Return TRUE if the soma falls within the given sphere defined by
      (x,y,z,radius)
      """
      for ix in range(len(self.point_lookup)):
          id = str(ix+1)
          parent_id = self.point_lookup[id]['parent']
          brType =self.point_lookup[id]['type']
          px =float(self.point_lookup[id]['siteX']) + xoffset
          py =float(self.point_lookup[id]['siteY']) + yoffset
          pz =float(self.point_lookup[id]['siteZ']) + zoffset
          pr =float(self.point_lookup[id]['siteR'])
          dist2soma= self.point_lookup[id]['dist2soma']
          if (int(brType) == self.branchType["axon"] ):
            distance = sqrt((x - px)**2 +
                        (y - py)**2 + (z - pz)**2)
            if (distance  <= r):
              id = parent_id
              somax =float(self.point_lookup[id]['siteX']) + xoffset
              somay =float(self.point_lookup[id]['siteY']) + yoffset
              somaz =float(self.point_lookup[id]['siteZ']) + zoffset
              print("soma bouton loc: ", somax, somay, somaz )
              return True
      return False


    def reviseNeuron(self):
        """
        Remove unused value in the second column
        5 = fork point
        6 = end point
        GOAL: make them get the value of the parent point
        """
        neuronFileName = self.swc_filename+"_new.swc"
        SWCFileSpine = open(neuronFileName, "w")

        thresholdTuftedZone = 600.0 # [um]
        proximalAISDistance2Soma = 20.0 # [um]
        distalAISDistance2Soma = 30.0 # [um]
        lineArray = []
        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x =self.point_lookup[id]['siteX']
            y =self.point_lookup[id]['siteY']
            z =self.point_lookup[id]['siteZ']
            r =self.point_lookup[id]['siteR']
            dist2soma= self.point_lookup[id]['dist2soma']

            if (int(brType) == 5):
              brType = self.point_lookup[parent_id]["type"]
            elif (int(brType) == 6):
              brType = self.point_lookup[parent_id]["type"]
            lineArray.append([id, str(brType),
                            str(x), str(y),
                            str(z), str(r), str(parent_id)])
        lineArray = np.asarray(lineArray)
        np.savetxt(neuronFileName, lineArray, fmt='%s')

    def update_dist2branchPoint_and_branchOrder(self, ids_HaveChanged):
        """
        Update the dist2soma for all points
                (currently absorbed in the self.point_lookup so far)
                based on the  given index 'id' of a new branching point

        @param ids_HaveChanged
        @type list

        STRATEGY:
            check for all ids with parent_id in the list ids_HaveChanged
        """
        ids_toUpdate = deepcopy(ids_HaveChanged)
        for idparent in ids_HaveChanged:
            #idparent = the id of the line that need to be update (due to
            #           it becomes a branchpoint)
            for id in self.line_ids:
                parent_id = self.point_lookup[id]['parent']
                if (parent_id == idparent):
                    # update current dist2branchPoint of  'id'
                    parent_dist2branchPoint = \
                        self.point_lookup[idparent]['dist2branchPoint']
                    if (self.point_lookup[id]['numChildren'] == 1):
                        self.point_lookup[id]['dist2branchPoint'] = \
                            self.find_distance(id, parent_id) + \
                            parent_dist2branchPoint
                        self.point_lookup[id]['branchOrder'] = \
                            self.point_lookup[parent_id]['branchOrder']
                    else:
                        self.point_lookup[id]['branchOrder'] = \
                            self.point_lookup[parent_id]['branchOrder'] + 1
                    ids_toUpdate.append(id)
            ids_toUpdate = list(set(ids_toUpdate))
            ids_toUpdate.remove(idparent)
        if (ids_toUpdate):
            self.update_dist2branchPoint_and_branchOrder(ids_toUpdate)

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

        GOAL: pass in the swc file of neuron
        put data to
        self.line_ids
        self.point_lookup
            organize the file in the form
            a list of 'line'
            NOTE:
            each 'line' is organized as a map with
            - key = line index,
            - value = a map with asssociated key for each column's value
            KEYS are
            'type' = swc branchType
            'siteX'
            'siteY'
            'siteZ'
            'siteR'
            'parent'
            'dist2soma'
            'dist2branchPoint'
            'branchOrder'
            'numChildren'

        """
        try:
            # file object
            myfile = open(self.swc_filename, "r+")
            # or "a+", whatever you need
        except IOError:
            print "Could not open file! Please check " + self.swc_filename

        # a list of strings, each string represent a line
        lines = myfile.read().splitlines()

        # a list of list of field-values
        # each line (as a list) now is broken down into fields
        split_lines = map(lambda x: x.strip().split(' '), lines)


        # a vector containing each line in the form of point_lookup
        self.line_ids = []
        # another map to enable
        # access to individual field (column)
        # take the line index as the key; and
        # value is
        self.point_lookup = {}

        maxDist2BranchPoint = 0.0
        maxDist2Soma = 0.0
        maxChildren = 0
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

        ## second patch = add data first
        #NO NEED THIS SECTION
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

        #self.listDuplicatedPoints = []
        for line in split_lines[start:]:
            if (len(line) != self.__class__.numFields):
                continue
            id = line[0]
            dist2soma = 0.0
            dist2branchPoint = 0.0
            branchOrder = 0  # branch just stemming from soma
            numChildren = 0

            do_update = False
            ids_as_new_branchpoint = []
            #if (id == '3441' or id == '3447' or id == '6782'):
            #    pdb.set_trace()
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
                if (distance <= 0.1):#don't want points to closed to each other
                  print("WARNING: there are duplicated points")
                  #self.listDuplicatedPoints.append(id)
                  #print(id, x,y,z,parent_id)
                #assert(distance > 0)
                self.point_lookup[parent_id]["numChildren"] += 1
                if (self.point_lookup[parent_id]["numChildren"] == 2) and \
                        (int(parent_id) != 1):
                    # NOTE: only update if a point become a branching
                    #  if it already a branching point, ignore the update
                    self.point_lookup[parent_id]["dist2branchPoint"] = 0.0
                    self.point_lookup[parent_id]["branchOrder"] += 1
                    # update dist2branchPoint and branchOrder of all its children
                    ids_as_new_branchpoint = [parent_id]
                    do_update = True

                dist2branchPoint = distance + \
                    self.point_lookup[parent_id]["dist2branchPoint"]
                if (int(parent_id) == 1):
                    branchOrder = self.point_lookup[parent_id]["branchOrder"]+1
                else:
                    branchOrder = self.point_lookup[parent_id]["branchOrder"]
            else:
                pass

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
                    self.update_dist2branchPoint_and_branchOrder(ids_as_new_branchpoint)

        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            point_info = self.point_lookup[id]
            numChildren = point_info['numChildren']
            if (int(numChildren)> maxChildren):
                maxChildren = numChildren
                idOfMaxChildren = id
        if (self.verbose):
          print("point with max-Dist2Soma is ", maxDist2Soma, " and id =", idOfMaxDist2Soma)
          print("point with max-Children is ", maxChildren, " and id =", idOfMaxChildren)
        # NOTE: keep this code for debug purpose
        """
        for ix in range(len(self.point_lookup)):
            #if (self.point_lookup[str(ix+1)]['dist2branchPoint'] == 0.0):
            print self.point_lookup[str(ix+1)]['dist2branchPoint']
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
        Generate the data for spines+bouton only at branchpoint
        OUTPUT:
            self.spineArray[]
            [not implement]//self.inhibitoryArray[]

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
        ###################
        # NOTE: The regular neuron is going to use '0' in MTYPE
        # so we should use a different MTYPE for spine and bouton
        # The different set of configuration that we can use
        #   spine="generic"= 1  ; bouton = "excitatory=2" or "inhibitory=3"
        self.spineMType = {"generic": 2}
        self.boutonMType = {"excitatory":1, "inhibitory":3}
        ############
        # start the spine generation process
        self.spineArray = []
        self.inhibitoryArray = []
        basalSpineCount = 0
        apicalSpineCount = 0
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

                    boutonType = self.boutonMType["excitatory"]
                    spineType = self.spineMType["generic"]
                    boutonFileName = "bouton_generic"
                    spineFileName = "spine_generic"
                    stimR = 5  # radius (micrometer) of stimulus taking effect
                    period = 300 # period of stimulus
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
                                        str(stimR),
                                        str(period),
                                        str(boutonFileName),
                                        str(spineFileName),
                                        str(bouton_include),
                                        str(spine_include),
                                        str(synapse_include)
                                        ])
                    ## END
                elif (int(branchType) == self.__class__.branchType["apical"]):
                    apicalSpineCount += 1
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
            on dendritic branches of different branch orders

        NOTE: Currently generate spines on 'basal' dendrite
        self.spineArray[] each line with
                        ([(branchType),
                        (boutonType),
                        (spineType),
                        (siteX),
                        (siteY),
                        (siteZ),
                        (stimR),
                        (period),
                        (boutonFileName),
                        (spineFileName),
                        (bouton_include),  # if =1 then recording bouton
                        (spine_include),
                        (synapse_include)
                        ])

        self.inhibitoryArray[] with same structure, except
             spineType="N/A"  (i.e. no spine to use)
             There are two options: GABA bouton come proximal or distal side
             of spines
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

        ###################
        # NOTE: The regular neuron is going to use '0' in MTYPE
        # so we should use a different MTYPE for spine and bouton
        # The different set of configuration that we can use
        #   spine="generic"= 1  ; bouton = "excitatory=2" or "inhibitory=3"
        self.spineMType = {"generic": 2}
        self.boutonMType = {"excitatory":1, "inhibitory":3}
        ############
        # start the spine generation process
        self.spineArray = []
        self.inhibitoryArray = []
        basalSpineCount = 0
        apicalSpineCount = 0
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
                        spineLocationAsDistancetoDistalEndCurrentSlot =\
                            np.random.uniform(0, slot_len)
                        spine_distance2distalpoint =\
                            i * slot_len + \
                            spineLocationAsDistancetoDistalEndCurrentSlot
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

                        boutonType = self.boutonMType["excitatory"]
                        spineType = self.spineMType["generic"]
                        boutonFileName = "bouton_generic"
                        spineFileName = "spine_generic"
                        # TUAN: TO MODIFY
                        stimR = 5  # radius (micrometer) of stimulus taking effect
                        period = 300 # period of stimulus
                        bouton_include= 0  #record data for this bouton?
                        spine_include = 0
                        synapse_include= 0 # record receptor for this synapse

                        self.spineArray.append([
                                           str(branchType),
                                           str(boutonType),
                                           str(spineType),
                                           str(round(siteX,3)),
                                           str(round(siteY,3)),
                                           str(round(siteZ,3)),
                                           str(stimR),
                                           str(period),
                                           str(boutonFileName),
                                           str(spineFileName),
                                           str(bouton_include),
                                           str(spine_include),
                                           str(synapse_include)
                                           ])
                        ## END
                        if (self.point_lookup[pid]['dist2branchPoint'] == 0.0 and
                            pid != '-1'):
                            tmplist.append(pid)
                        # the we feed this to tree's points to generate the
                        # spine location

                        #self.spineArray.append([(branchType), (spineMType), (siteX), (siteY), (siteZ), (stimR), (period), (
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
            at different distance bins to the soma
        GOAL: put data to
        self.spineArray []
        self.inhibitoryArray []
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
        ###################
        # NOTE: The regular neuron is going to use '0' in MTYPE
        # so we should use a different MTYPE for spine and bouton
        # The different set of configuration that we can use
        #   spine="generic"= 1  ; bouton = "excitatory=2" or "inhibitory=3"
        self.spineMType = {"generic": 2}
        self.boutonMType = {"excitatory":1, "inhibitory":3}
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

                        boutonType = self.boutonMType["excitatory"]
                        spineType = self.spineMType["generic"]
                        boutonFileName = "bouton_generic"
                        spineFileName = "spine_generic"
                        # TUAN: TO MODIFY
                        stimR = 5  # radius (micrometer) of stimulus taking effect
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
                                           str(stimR),
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

    def genSpines_Poisson(self, mean_interval, spineMType, distance,
                          period, boutonType, spineType, branchType, ageType):
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
        stimR = 5
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
                self.spineArray.append([str(branchType), str(spineMType), str(siteX), str(siteY), str(siteZ), str(stimR), str(period), str(
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
                        junctionBoutonType = bouton + thin + ageType + branchType
                        junctionSpineType = spine + thin + ageType + branchType
                    elif junctionTypeRand > mushroomJunctionRate:
                        junctionMType = 3
                        boutonMType = 5
                        junctionBoutonType = bouton + mush + ageType + branchType
                        junctionSpineType = spine + mush + ageType + branchType
                        self.spineArray.append([str(branchType), str(spineMType), str(siteX), str(siteY), str(siteZ), str(stimR), str(period), str(
                            junctionBoutonType), str(junctionSpineType), str(bouton_include), str(spine_include), str(synapse_include), str(boutonMType)])
                    else:
                        break
    def _isChildOf(self, listCandidate, parentCandidate ):
        """
        Tell if a point in the list is a child of the given point
        """
        for id in listCandidate:
            point_info = self.point_lookup[id]
            parent_id =  point_info['parent']
            stored_id = id
            while (parent_id != '1'):
                if (parent_id == parentCandidate):
                    print("OOOOOOOOOOOOOOOOOOOOWW", stored_id, id)
                    id = parent_id
                    point_info = self.point_lookup[id]
                    parent_id =  point_info['parent']

    def genSpine_PyramidalL5(self):
        """
        Call this function to generate spines with statistics for Pyramidal LayerV neuron
        OUTPUT:
            self.spineArray[]
            self.inhibitoryArray[]

        """
        ###################################
        ## YOU CAN MODIFY HERE
        ##NOTE: focus on young data
        ageType = 'young'  # 'young', 'aged', 'general'
        data2use = "MorrisonLab_L23"  # 'MorrisonLab_L23'
        useMean = True # True, False  [where we generate random size for spines or not]

        # NOTE: mean distance from spines (of a given type)
        #    to the adjacent one (regardless of type)
        if (data2use == "MorrisonLab_L23"):
            ##the same for any age
            scaling_factor = 1.3  # for distance scaling [um]
            mean_apical_thin_interval = scaling_factor * 0.81
            mean_apical_mush_interval = scaling_factor * 0.81
            mean_basal_thin_interval = scaling_factor * 0.81
            mean_basal_mush_interval = scaling_factor * 0.81
            inhFactor = 2.6
            meanApicalInhInterval  = inhFactor*1.0
            meanBasalInhInterval   = inhFactor*1.0
        else:
            print("ERROR: unexpected input")
            return
        ## NOTE: between Thin and Mush, it is assumed
        #   the chance is 70% for having Mush
        #   and 30% for Thin
        proportionMushBasal = 0.7
        proportionMushApical = 0.7
        #########
        stimR = 5 # radius of stimulus [um]
        bouton_include = 0 # I/O on bouton?
        spine_include = 0
        synapse_include = 0
        ## rate of having spine on branchpoint
        #junctionRate = .03
        #spineJunctionrate = .61
        #mushroomJunctionRate = .21
        ##General data
        ## NOTE: spine of MSN is bigger than pyramidal neuron
        rNeckMean = 0.1
        rNeckSTD = 0.03
        lNeckMean = 1.5
        lNeckSTD = 0.3
        rHeadMean = 0.7
        rHeadSTD = 0.3
        ##Age-dependent data
        rNeckMeanYoung = 0.7
        rNeckSTDYoung = 0.3
        lNeckMeanYoung = 1.5
        lNeckSTDYoung = 0.3
        #
        rNeckMeanAged = 0.7
        rNeckSTDAged = 0.3
        lNeckMeanAged = 1.5
        lNeckSTDAged = 0.3
        ##Age-dependent and Type-dependent  data
        if (data2use == "MorrisonLab_L23"):
            #NOTE: neck (len+radius), head (only radius)
            rNeckMeanThinYoung = 0.1
            rNeckSTDThinYoung = 0.03
            lNeckMeanThinYoung = 1.897
            lNeckSTDThinYoung = 0.3
            rHeadMeanThinYoung = 0.591
            rHeadSTDThinYoung = 0.03
            rNeckMeanMushYoung = 0.1
            rNeckSTDMushYoung = 0.03
            lNeckMeanMushYoung = 1.934
            lNeckSTDMushYoung = 0.3
            rHeadMeanMushYoung = 0.610
            rHeadSTDMushYoung = 0.03
            #
            rNeckMeanThinAged = 0.1
            rNeckSTDThinAged = 0.03
            lNeckMeanThinAged = 1.802
            lNeckSTDThinAged = 0.3
            rHeadMeanThinAged = 0.615
            rHeadSTDThinAged = 0.03
            rNeckMeanMushAged = 0.1
            rNeckSTDMushAged = 0.03
            lNeckMeanMushAged = 1.899
            lNeckSTDMushAged = 0.3
            rHeadMeanMushAged = 0.637
            rHeadSTDMushAged = 0.03

        ####
        if (ageType == 'young'):
            rNeckMeanThin= rNeckMeanThinYoung
            rNeckSTDThin= rNeckSTDThinYoung
            lNeckMeanThin= lNeckMeanThinYoung
            lNeckSTDThin= lNeckSTDThinYoung
            rHeadMeanThin= rHeadMeanThinYoung
            rHeadSTDThin= rHeadSTDThinYoung
            rNeckMeanMush= rNeckMeanMushYoung
            rNeckSTDMush= rNeckSTDMushYoung
            lNeckMeanMush= lNeckMeanMushYoung
            lNeckSTDMush= lNeckSTDMushYoung
            rHeadMeanMush= rHeadMeanMushYoung
            rHeadSTDMush= rHeadSTDMushYoung
        elif  (ageType == 'aged'):
            rNeckMeanThin= rNeckMeanThinAged
            rNeckSTDThin= rNeckSTDThinAged
            lNeckMeanThin= lNeckMeanThinAged
            lNeckSTDThin= lNeckSTDThinAged
            rHeadMeanThin= rHeadMeanThinAged
            rHeadSTDThin= rHeadSTDThinAged
            rNeckMeanMush= rNeckMeanMushAged
            rNeckSTDMush= rNeckSTDMushAged
            lNeckMeanMush= lNeckMeanMushAged
            lNeckSTDMush= lNeckSTDMushAged
            rHeadMeanMush= rHeadMeanMushAged
            rHeadSTDMush= rHeadSTDMushAged
        else:
            #handle here
            print 'Unknown age type: '
            return

        ### END
        #############################################

        ###################
        # NOTE: The regular neuron is going to use '0' in MTYPE
        # so we should use a different MTYPE for spine and bouton
        # The different set of configuration that we can use
        self.spineMType = {"thin": 2, "mush": 3}
        self.boutonMType= {"thin":4, "mush":5, "GABA": 1} #as a function of spine-type
        ###################
        ## SETTING 1
        bouton = 'bouton_'
        spine = 'spine_'
        thin = 'thin_'
        mush = 'mushroom_'
        apical = '_apical'
        basal = '_basal'

        #define the prefix for *.swc filenames
        boutonTypeThinApical = bouton + thin + ageType + apical
        spineTypeThinApical = spine + thin + ageType + apical
        boutonTypeMushApical = bouton + mush + ageType + apical
        spineTypeMushApical = spine + mush + ageType + apical
        boutonTypeThinBasal = bouton + thin + ageType + basal
        spineTypeThinBasal = spine + thin + ageType + basal
        boutonTypeMushBasal = bouton + mush + ageType + basal
        spineTypeMushBasal = spine + mush + ageType + basal

        basalDistance = 0
        apicalDistance = 0
        ###################
        ## SETTING 2
        ## Folder setup
        ## remove files
        self.spineFolder = self.swcFolder + '/spines'
        execute('mkdir -p ' + self.spineFolder)
        # execute('rm spines/*')
        execute('find '+self.spineFolder +' -maxdepth 1 -name "*.swc" -print0 | xargs -0 rm')

        ###################
        ## SETTING 3
        ## APICAL + BASAL DEN
        #period of stimulus signal
        apical_period = 300
        basal_period = 140
        apical_inhibitory_period = 250
        basal_inhibitory_period = 250


        ######
        # Suppose spine occurrence follow Poisson distribution
        # we use the mean adjacent distance for each spine type (thin, mush)
        # to find the location of next spine, starting from the most distal
        # points
        self.spineArray = []
        self.inhibitoryArray = []
        self.spineCount = 0

        self.locPreviousSpineOrSynapse=  {}

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

        holdpoint_list = list(set(holdpoint_list))

        #### Hold number of times visited a point
        #[pointID] = times
        timeVisitPoint = {}

        #####Find soma
        for ix in range(len(self.point_lookup)):
            id = str(ix+1)
            parent_id = self.point_lookup[id]['parent']
            brType =self.point_lookup[id]['type']
            x_soma =float(self.point_lookup[id]['siteX'])
            y_soma =float(self.point_lookup[id]['siteY'])
            z_soma =float(self.point_lookup[id]['siteZ'])
            r_soma =float(self.point_lookup[id]['siteR'])
            if (int(parent_id) == -1):
              break
        print("soma radius: ", r_soma)
        ############
        # start the spine generation process
        """
        Basically, what the algorithm does is
          1. start with all the distal points
          2. for each point in the list
            generate the next spine using Poisson distribution
            put the coordinate in the spineArray[]
            .... in the holdpoint_list
        """
        point_lookupBackUp = deepcopy(self.point_lookup)
        while holdpoint_list:
            tmplist = []

            for id in holdpoint_list:
                if (int(id) == -1):
                    break
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                dist2soma = float(point_info['dist2soma'])
                numChildren =  int(point_info['numChildren'])
                if (dist2branchPoint == 0.0):
                    assert(numChildren > 0)
                #assert(dist2branchPoint==0.0 && numChildren > 0)
                #spine_filename = 'spine' +
                pid = parent_id
                cid = id
                if (int (pid) == -1):
                    continue
                if  (dist2soma <= r_soma):
                    continue

                distance=self.find_distance(cid, pid)
                randUniform = random.random()
                #genSpineType = "none"
                spineMType = self.spineMType["thin"]
                mean_spine_distance = 0.0
                if int(branchType) == self.branchType["basal"]:
                    stimPeriod = basal_period
                    if (randUniform <= proportionMushBasal):
                        #genSpineType = "mush"
                        mean_spine_distance = mean_basal_mush_interval
                        spineFileName = spineTypeMushBasal
                        boutonFileName = boutonTypeMushBasal
                        spineMType = self.spineMType["mush"]
                        rNeckMean= rNeckMeanMush
                        rNeckSTD= rNeckSTDMush
                        lNeckMean= lNeckMeanMush
                        lNeckSTD= lNeckSTDMush
                        rHeadMean= rHeadMeanMush
                        rHeadSTD= rHeadSTDMush
                    else:
                        mean_spine_distance = mean_basal_thin_interval
                        spineFileName = spineTypeThinBasal
                        boutonFileName = boutonTypeThinBasal
                        spineMType = self.spineMType["thin"]
                        rNeckMean= rNeckMeanThin
                        rNeckSTD= rNeckSTDThin
                        lNeckMean= lNeckMeanThin
                        lNeckSTD= lNeckSTDThin
                        rHeadMean= rHeadMeanThin
                        rHeadSTD= rHeadSTDThin
                elif int(branchType) == self.branchType["apical"] or \
                    int(branchType) == self.branchType["tufted"] :
                    stimPeriod = apical_period
                    if (randUniform <= proportionMushApical):
                        #genSpineType = "mush"
                        mean_spine_distance = mean_apical_mush_interval
                        spineFileName = spineTypeMushApical
                        boutonFileName = boutonTypeMushApical
                        spineMType = self.spineMType["mush"]
                        rNeckMean= rNeckMeanMush
                        rNeckSTD= rNeckSTDMush
                        lNeckMean= lNeckMeanMush
                        lNeckSTD= lNeckSTDMush
                        rHeadMean= rHeadMeanMush
                        rHeadSTD= rHeadSTDMush
                    else:
                        mean_spine_distance = mean_apical_thin_interval
                        spineFileName = spineTypeThinApical
                        boutonFileName = boutonTypeThinApical
                        spineMType = self.spineMType["thin"]
                        rNeckMean= rNeckMeanThin
                        rNeckSTD= rNeckSTDThin
                        lNeckMean= lNeckMeanThin
                        lNeckSTD= lNeckSTDThin
                        rHeadMean= rHeadMeanThin
                        rHeadSTD= rHeadSTDThin
                else:
                    tmplist.append(pid)
                    continue

                # Assume spine appearance with adjacent-distance
                # follows exponential distribution
                interval = rnd.exponential(mean_spine_distance)
                #interval = self.nextLocation(mean_spine_distance)
                #print(interval)
                assert(interval > 0)
                assert(self.spineCount< 20000)
                #assert(self.spineCount< 30000)
                # Find the pair of points
                keepGoing = False
                firstUse = True
                while (interval >= distance and int(pid) != -1):
                    firstUse = False
                    # update interval and find new cid, pid
                    interval = interval - distance
                    cid = pid
                    point_info = self.point_lookup[cid]
                    pid =  point_info['parent']
                    if (int(pid) == -1):
                        break
                    distance=self.find_distance(cid, pid)
                    dist2branchPoint = float(point_info['dist2branchPoint'])
                    dist2soma = float(point_info['dist2soma'])
                    numChildren =  int(point_info['numChildren'])
                    if (dist2branchPoint == 0.0):
                        keepGoing = True
                        if (cid in timeVisitPoint):
                            timeVisitPoint[cid] += 1
                        else:
                            timeVisitPoint[cid] = 1
                        if (timeVisitPoint[cid] == int(numChildren)):
                            tmplist.append(cid)
                        #else:
                            #timeVisitPoint[cid] -= 1 #as the point will be visit again
                            #print (cid + ": child = " + str(numChildren) + " while it is "
                            #       #+ str(timeVisitPoint[cid]))
                            #       + str(timeVisitPoint))
                            #timeVisitPoint[cid] += 1 #as the point will be visit again
                            #keepGoing = True
                        break
                if  (keepGoing):
                    continue

                if (int(pid) == -1):
                   continue

                if  (dist2soma-interval <= r_soma):
                    continue

                #now we ensured the spine in between cid and pid
                x1 = float(self.point_lookup[cid]['siteX'])
                y1 = float(self.point_lookup[cid]['siteY'])
                z1 = float(self.point_lookup[cid]['siteZ'])
                parent_x1 = float(self.point_lookup[pid]['siteX'])
                parent_y1 = float(self.point_lookup[pid]['siteY'])
                parent_z1 = float(self.point_lookup[pid]['siteZ'])
                # generate new point for spine
                siteX = np.interp(distance-interval, [0, distance], [parent_x1, x1])
                siteY = np.interp(distance-interval, [0, distance], [parent_y1, y1])
                siteZ = np.interp(distance-interval, [0, distance], [parent_z1, z1])
                if spineMType == self.spineMType["thin"]:
                    boutonMType = self.boutonMType["thin"]
                elif spineMType == self.spineMType["mush"]:
                    boutonMType = self.boutonMType["mush"]
                rand = np.random.uniform()
                #if (rand < chance2HaveSpine):
                if (1):
                    # limits # spines to be generated
                    period = stimPeriod
                    self.spineArray.append([str(branchType),
                                            str(boutonMType),
                                            str(spineMType),
                                            str(siteX), str(siteY), str(siteZ),
                                            str(stimR), str(period),
                                            str(boutonFileName), str(spineFileName),
                                            str(bouton_include),
                                            str(spine_include),
                                            str(synapse_include)
                                            ]
                                            )
                    self.spineCount += 1
                    index = self.spineCount
                    dend_dx = parent_x1 - x1
                    dend_dy = parent_y1 - y1
                    dend_dz = parent_z1 - z1
                    dx, dy, dz = getSpineVector(dend_dx, dend_dy, dend_dz, index)
                    #genSpine(array, index, spineMType, x, y, z, dx, dy, dz, period, ageType, branchType)
                    ######################
                    if (useMean == True):
                        rNeck = rNeckMean  # (um)
                        lNeck = lNeckMean  # (um)
                        rHead = rHeadMean  # (um)
                    else:
                        rNeck = np.radom.normal(rNeckMean, rNeckSTD)  # (um)
                        lNeck = np.radom.normal(lNeckMean, lNeckSTD)  # (um)
                        rHead = np.radom.normal(rHeadMean, rHeadSTD)  # (um)
                    #if int(spineMType) == self.spineMType["thin"]: # thin
                    #    boutonMType = self.boutonMType["thin"]
                    #    # not significant betweeen apical and basal
                    #    # based on L2/3 data
                    #    rHead = 0.591 if ageType == 'young' else 0.615
                    #    lNeck = 1.897 if ageType == 'young' else 1.802
                    #    #if int(branchType) == self.branchType["basal"]: # basal
                    #    #    rHead = 0.1282 if ageType == 'young' else 0.1408
                    #    #    lNeck = 1.4360 if ageType == 'young' else 1.2663
                    #    #elif int(branchType) == 4 or int(branchType) == 6: # apical
                    #    #    rHead = 0.1255 if ageType == 'young' else 0.1360
                    #    #    lNeck = 1.4289 if ageType == 'young' else 1.3820
                    #    #else:
                    #    #    print 'Unknown branch type: ' + str(branchType)
                    #    #    return
                    #elif int(spineMType) == self.spineMType["mush"]: # mushroom
                    #    boutonMType = self.boutonMType["mush"]
                    #    # not significant betweeen apical and basal
                    #    # based on L2/3 data
                    #    rHead = 0.610 if ageType == 'young' else 0.637
                    #    lNeck = 1.934 if ageType == 'young' else 1.899
                    #    #if int(branchType) == 3:
                    #    #    rHead = 0.2377 if ageType == 'young' else 0.2366
                    #    #    lNeck = 1.4819 if ageType == 'young' else 1.4476
                    #    #elif int(branchType) == 4 or int(branchType) == 6: #apical or tufted
                    #    #    rHead = 0.2358 if ageType == 'young' else 0.2382
                    #    #    lNeck = 1.4906 if ageType == 'young' else 1.4571
                    #    #else:
                    #    #    print 'Unknown branch type: ' + str(branchType)
                    #    #    return
                    #else:
                    #    print 'Unknown spine type: ' + str(spineMType)
                    #    return
                    offset = 0.0
                    self.createSpineSWC(index, dx, dy, dz, rHead, rNeck, lNeck,
                                        offset)
                    ### CHECK ANGLE (90-degreee)
                    #p1 = Point3D(x1,y1,z1)
                    #p2 = Point3D(parent_x1, parent_y1, parent_z1)
                    #l1 = Line3D(p1,p2)
                    #p3 = Point3D(0,0,0)
                    #p4 = Point3D(dx*lNeck, dy*lNeck, dz*lNeck)
                    #l2 = Line3D(p3,p4)
                    #print(l1.is_perpendicular(l2), " ", math.degrees(l1.angle_between(l2)))
                    #print (l1.distance())
                    #angle = math.degrees(l1.angle_between(l2))
                    #if (angle < 80 or angle > 99):
                    #    print (angle)
                    ##END CHECK
                    #createBoutonSWC(index, dx, dy, dz, lNeck + 0.1)
                    self.createBoutonSWC(index, dx, dy, dz, lNeck + 0.0)


                if (interval > 0.05):#tolerance distance
                    # NOTE: this if is important
                    newid = str(len(self.point_lookup)+1)
                    newid_dist2soma = float(self.point_lookup[cid]['dist2soma']) - interval
                    newid_dist2branchPoint = newid_dist2soma -\
                        self.point_lookup[pid]['dist2soma'] + \
                        self.point_lookup[pid]['dist2branchPoint']
                    #pdb.set_trace()
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
                    tmplist.append(newid)
                else:
                    tmplist.append(cid)
            holdpoint_list = list(set(tmplist))
        #print("spines count: ", basalSpineCount)
        self.point_lookup = deepcopy(point_lookupBackUp) # restore

        ## {'X':0, 'Y':0, 'Z':0}
        #### IMPORTANT: Assume first element is the soma
        ## otherwise, we need to make sure searching from soma
        #for id in self.line_ids:
        #    branchType = self.point_lookup[id]['type']
        #    current_point = self.point_lookup[id]
        #    parent_id = self.point_lookup[id]['parent']
        #    if (int(parent_id) == -1):
        #        continue # skip
        #    parent_point = self.point_lookup[parent_id]
        #    parent_x = float(parent_point['siteX'])
        #    parent_y = float(parent_point['siteY'])
        #    parent_z = float(parent_point['siteZ'])
        #    if (not self.locPreviousSpineOrSynapse):
        #        location = {}
        #        location['X'] = parent_x
        #        location['Y'] = parent_y
        #        location['Z'] = parent_z
        #        self.locPreviousSpineOrSynapse["thin"] = location
        #        self.locPreviousSpineOrSynapse["mush"] = location
        #        self.locPreviousSpineOrSynapse["inhibit"] = location
        #    x = float(current_point['siteX'])
        #    y = float(current_point['siteY'])
        #    z = float(current_point['siteZ'])
        #    #segment length (on which we put spines)
        #    distance = sqrt((x - parent_x)**2 + (y - parent_y)
        #                    ** 2 + (z - parent_z)**2)
        #    if int(branchType) == self.branchType["basal"]:
        #        basalSpineCount = 0
        #        self.genSpines(mean_basal_thin_interval,
        #                       self.spineMType["thin"], distance, basal_period,
        #                       boutonTypeThinBasal, spineTypeThinBasal, basal)
        #        self.genSpines(mean_basal_mush_interval,
        #                       self.spineMType["mush"], distance, basal_period,
        #                       boutonTypeMushBasal, spineTypeMushBasal, basal)
        #        genInhibitory(basal_inhibitory_period, distance)
        #        basalDistance += distance
        #    elif int(branchType) == self.branchType["apical"] or \
        #        int(branchType) == self.branchType["tufted"] :
        #        apicalSpineCount = 0
        #        self.genSpines(mean_apical_thin_interval,
        #                       self.spineMType["thin"], distance, apical_period,
        #                       boutonTypeThinApical, spineTypeThinApical, apical)
        #        self.genSpines(mean_apical_mush_interval,
        #                       self.spineMType["mush"], distance, apical_period,
        #                       boutonTypeMushApical, spineTypeMushApical, apical)
        #        genInhibitory(apical_inhibitory_period, distance)
        #        apicalDistance += distance

        print("Total excit. spines: ", self.spineCount)
        numSpines = self.spineCount
        if (0):# make 0 to not generate GABA input
          boutonTypeGABA = "GABABouton_" + ageType + apical
          listBranches = [self.branchType["apical"], self.branchType["tufted"]]
          self._genInhibitory(listBranches, meanApicalInhInterval,
                              apical_inhibitory_period,
                              boutonTypeGABA,
                              "N/A",
                              self.branchType["apical"],
                              "N/A",
                              ageType
                              )
          boutonTypeGABA = "GABABouton_" + ageType + basal
          listBranches = [self.branchType["basal"],]
          self._genInhibitory(listBranches, meanBasalInhInterval,
                              basal_inhibitory_period,
                              boutonTypeGABA,
                              "N/A",
                              self.branchType["basal"],
                              "N/A",
                              ageType
                              )
        print("Total GABA inputs: ", self.spineCount-numSpines)
        self.rotateSpines()
        self.saveSpines()
        self.genboutonspineSWCFiles_PL5b()

    def genSpine_PL5_new(self, use_mean=True):
        """
        Call this function to generate spines with statistics for Pyramidal LayerV neuron
        OUTPUT:
            self.spineArray[]
            self.inhibitoryArray[]
        """
        bouton = 'bouton_'
        spine = 'spine_'
        thin = 'thin_'
        mush = 'mushroom_'
        apical = '_apical'
        basal = '_basal'
        ##NOTE: focus on young data
        ageType = 'young'
        # ageType = 'aged'

        #define the prefix for *.swc filenames
        boutonTypeThinApical = bouton + thin + ageType + apical
        spineTypeThinApical = spine + thin + ageType + apical
        boutonTypeMushApical = bouton + mush + ageType + apical
        spineTypeMushApical = spine + mush + ageType + apical
        boutonTypeThinBasal = bouton + thin + ageType + basal
        spineTypeThinBasal = spine + thin + ageType + basal
        boutonTypeMushBasal = bouton + mush + ageType + basal
        spineTypeMushBasal = spine + mush + ageType + basal

        # NOTE: mean distance from spines (of a given type)
        #    to the adjacent one (regardless of type)
        scaling_factor = 1.4  # for distance scaling [um]
        mean_apical_thin_interval = scaling_factor * 1.6
        mean_apical_mush_interval = scaling_factor * 5.0
        mean_basal_thin_interval = scaling_factor * 1.6
        mean_basal_mush_interval = scaling_factor * 5.0
        inhFactor = 2.6
        meanApicalInhInterval  = inhFactor*1.0
        meanBasalInhInterval   = inhFactor*1.0

        basalDistance = 0
        apicalDistance = 0
        ######
        ## Folder setup
        ## remove files
        self.spineFolder = self.swcFolder + '/spines'
        execute('mkdir -p ' + self.spineFolder)
        # execute('rm spines/*')
        execute('find '+self.spineFolder +' -maxdepth 1 -name "*.swc" -print0 | xargs -0 rm')

        ###################
        ## APICAL + BASAL DEN
        #period of stimulus signal
        apical_period = 300
        basal_period = 140
        apical_inhibitory_period = 250
        basal_inhibitory_period = 250

        ###################
        # NOTE: The regular neuron is going to use '0' in MTYPE
        # so we should use a different MTYPE for spine and bouton
        # The different set of configuration that we can use
        self.spineMType = {"thin": 2, "mush": 3}
        self.boutonMType= {"thin":4, "mush":5, "GABA": 1} #as a function of spine-type

        ######
        # Suppose spine occurrence follow Poisson distribution
        # we use the mean adjacent distance for each spine type (thin, mush)
        # to find the location of next spine, starting from the most distal
        # points
        self.spineArray = []
        self.inhibitoryArray = []
        self.spineCount = 0
        ########################
        listBranches = [self.branchType["basal"],]
        self._genSpineOnBranch(listBranches, mean_basal_thin_interval,
                         basal_period, boutonTypeThinBasal,
                         spineTypeThinBasal,
                         self.branchType["basal"],
                         self.spineMType["thin"], ageType
                               )
        listBranches = [self.branchType["basal"],]
        self._genSpineOnBranch(listBranches, mean_basal_mush_interval,
                         basal_period, boutonTypeMushBasal,
                         spineTypeMushBasal,
                         self.branchType["basal"],
                         self.spineMType["mush"], ageType
                               )
        listBranches = [self.branchType["apical"], self.branchType["tufted"]]
        self._genSpineOnBranch(listBranches, mean_apical_thin_interval,
                         apical_period, boutonTypeThinApical,
                         spineTypeThinApical,
                         self.branchType["apical"],
                         self.spineMType["thin"], ageType
                               )
        listBranches = [self.branchType["apical"], self.branchType["tufted"]]
        self._genSpineOnBranch(listBranches, mean_apical_mush_interval,
                         apical_period, boutonTypeMushApical,
                         spineTypeMushApical,
                         self.branchType["apical"],
                         self.spineMType["mush"], ageType
                               )

        print("Total excit. spines: ", self.spineCount)
        numSpines = self.spineCount
        ####
        if (0):# make 0 to not generate GABA input
          boutonTypeGABA = "GABABouton_" + ageType + apical
          listBranches = [self.branchType["apical"], self.branchType["tufted"]]
          self._genInhibitory(listBranches, meanApicalInhInterval,
                              apical_inhibitory_period,
                              boutonTypeGABA,
                              "N/A",
                              self.branchType["apical"],
                              "N/A",
                              ageType
                              )
          boutonTypeGABA = "GABABouton_" + ageType + basal
          listBranches = [self.branchType["basal"],]
          self._genInhibitory(listBranches, meanBasalInhInterval,
                              basal_inhibitory_period,
                              boutonTypeGABA,
                              "N/A",
                              self.branchType["basal"],
                              "N/A",
                              ageType
                              )
        print("Total GABA inputs: ", self.spineCount-numSpines)
        self.rotateSpines()
        self.saveSpines()
        self.genboutonspineSWCFiles_PL5b()
        ######################
        #apicalPeriod = 300
        #basalPeriod = 140
        #apicalInhPeriod = 250
        #basalInhPeriod = 250
        #spineArrayHeader = 'index spineMType boutonMType x y z period boutonInclude spineInclude synapseInclude'
        #inhArrayHeader = 'index boutonMType x y z period boutonInclude synapseInclude'
        #spinesFile = open('spines.txt', 'w')
        #spinesFile.write(spineArrayHeader+"\n")
        #for spine in spineArray:
        #    spinesFile.write(' '.join(spine)+"\n")
        #spinesFile.close()
        #inhFile = open('inhibitory.txt', 'w')
        #inhFile.write(inhArrayHeader+"\n")
        #for inhSynapse in inhArray:
        #    inhFile.write(' '.join(inhSynapse)+"\n")
        #inhFile.close()
        ######################

    def genSpines(self, mean_interval, spineMType, distance, period,
                  boutonFileName, spineFileName, branchType):
        """

        Generate a spine

        @param mean_interval = mean distance of spines of that type
        @type float
        @param spineMType    = the integer represent the morphology type ()
        @type self.spineMType data member
        @param distance      = segment length on which spines are placed
        @type float
        @param period       = period of stimulus of bouton on that spine
        @type float           [ms]
        @param boutonFileName   =
        @type string
        @param spineFileName   =
        @type string
        @param branchType  =
        """
        stimR = 5 # radius of stimulus [um]
        bouton_include = 0 # I/O on bouton?
        spine_include = 0
        synapse_include = 0
        # rate of having spine on branchpoint
        junctionRate = .03
        spineJunctionrate = .61
        mushroomJunctionRate = .21
        ######################
        # START
        spine_location = 0
        while True:
            # Assume spine appearance with adjacent-distance
            # follows exponential distribution
            interval = rnd.exponential(mean_interval)
            spine_location += interval
            if (spine_location < distance):
                siteX = np.interp(spine_location, [0, distance], [parent_x, x])
                siteY = np.interp(spine_location, [0, distance], [parent_y, y])
                siteZ = np.interp(spine_location, [0, distance], [parent_z, z])
                if spineMType == self.spineMType["thin"]:
                    boutonMType = self.boutonMType["thin"]
                elif spineMType == self.spineMType["mush"]:
                    boutonMType = self.boutonMType["mush"]
                self.spineArray.append([str(branchType),
                                        str(boutonMType),
                                        str(spineMType),
                                        str(siteX), str(siteY), str(siteZ),
                                        str(stimR), str(period),
                                        str(boutonFileName), str(spineFileName),
                                        str(bouton_include),
                                        str(spine_include),
                                        str(synapse_include)
                                        ]
                                        )
                # GENERATE SPINE at JUNCTION BRANCH POINT
                junctionRand = rnd.rand()
                if junctionRand <= junctionRate:
                    junctionTypeRand = rnd.rand()
                    junctionX = siteX
                    junctionY = siteY
                    junctionZ = siteZ
                    if junctionTypeRand <= mushroomJunctionRate:
                        junctionMType = self.spineMType["thin"]
                        boutonMType = self.boutonMType["thin"]
                        junctionBoutonType = bouton + thin + ageType + branchType
                        junctionSpineType = spine + thin + ageType + branchType
                    elif junctionTypeRand > mushroomJunctionRate:
                        junctionMType = self.spineMType["mush"]
                        boutonMType = self.boutonMType["mush"]
                        junctionBoutonType = bouton + mush + ageType + branchType
                        junctionSpineType = spine + mush + ageType + branchType
                        self.spineArray.append([str(branchType),
                                                str(boutonMType),
                                                str(spineMType),
                                                str(siteX), str(siteY), str(siteZ),
                                                str(stimR), str(period),
                                                str(junctionBoutonType),
                                                str(junctionSpineType),
                                                str(bouton_include),
                                                str(spine_include),
                                                str(synapse_include)
                                                ])
                    else:
                        break

    def genInhibitory(self, period, distance):
        mean_inhibitory_interval = 3
        boutonMType = 1
        inhibitory_location = 0
        stimR = 5
        bouton_include = 0
        spine_include = 'N/A'
        synapse_include = 0
        boutonFileName = 'bouton_inhibitory'
        spineFileName = 'N/A'

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
                self.inhibitoryArray.append([str(branchType),
                                             str(boutonMType),
                                             str(siteX), str(siteY), str(siteZ),
                                             str(stimR), str(period),
                                             str(boutonFileName),
                                             str(spineFileName),
                                             str(bouton_include),
                                             str(spine_include),
                                             str(synapse_include)])
            else:
                break

    def _genInhibitory(self, listBranches, mean_symsynapse_distance, stimPeriod,
                          boutonFileName, spineFileName, branchType,
                          spineMType, ageType):
        stimR = 5 # radius of stimulus [um]
        boutonMType = 1
        bouton_include = 0 # I/O on bouton?
        spine_include = 'N/A'
        synapse_include = 0
        # rate of having spine on branchpoint
        chance2HaveGABAInput = 0.60
        #chance2HaveGABAInput = self.chance2HaveGABAInput
        spineFileName = 'N/A'
        spineMType = 'N/A'

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

        holdpoint_list = list(set(holdpoint_list))
        # print(holdpoint_list)
        ############
        # start the spine generation process
        """
        Basically, what the algorithm does is
          1. start with all the distal points
          2. for each point in the list
            generate the next spine using Poisson distribution
            put the coordinate in the spineArray[]
            .... in the holdpoint_list
        """
        point_lookupBackUp = deepcopy(self.point_lookup)
        while holdpoint_list:
            tmplist = []
            for id in holdpoint_list:
                if (int(id) == -1):
                    break
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                dist2soma = float(point_info['dist2soma'])
                #spine_filename = 'spine' +
                pid = parent_id
                cid = id
                if (int (pid) == -1):
                    continue
                if (int(pid) != -1 and
                    not self._branchInList(branchType, listBranches)):
                    tmplist.append(pid)
                    continue

                distance=self.find_distance(cid, pid)

                # Assume spine appearance with adjacent-distance
                # follows exponential distribution
                interval = rnd.exponential(mean_symsynapse_distance)
                #interval = self.nextLocation(mean_symsynapse_distance)
                #print(interval)
                assert(interval > 0)
                #assert(self.spineCount< 10000)
                # Find the pair of points
                while (interval > distance and int(pid) != -1):
                    # update interval and find new cid, pid
                    interval = interval - distance
                    cid = pid
                    point_info = self.point_lookup[cid]
                    pid =  point_info['parent']
                    if (int(pid) == -1):
                        break
                    distance=self.find_distance(cid, pid)
                    dist2branchPoint = float(point_info['dist2branchPoint'])
                    if (dist2branchPoint == 0.0):
                        tmplist.append(cid)
                        break

                if (dist2branchPoint == 0.0 and interval > distance):
                    continue
                if (int(pid) == -1):
                   continue

                #now we ensure the spine in between cid and pid
                x1 = float(self.point_lookup[cid]['siteX'])
                y1 = float(self.point_lookup[cid]['siteY'])
                z1 = float(self.point_lookup[cid]['siteZ'])
                parent_x1 = float(self.point_lookup[pid]['siteX'])
                parent_y1 = float(self.point_lookup[pid]['siteY'])
                parent_z1 = float(self.point_lookup[pid]['siteZ'])
                # generate new point for spine
                siteX = np.interp(distance-interval, [0, distance], [parent_x1, x1])
                siteY = np.interp(distance-interval, [0, distance], [parent_y1, y1])
                siteZ = np.interp(distance-interval, [0, distance], [parent_z1, z1])
                rand = np.random.uniform()
                if (rand < chance2HaveGABAInput):
                # if (1):
                    # limits # spines to be generated
                    period = stimPeriod
                    self.inhibitoryArray.append([str(branchType),
                                            str(boutonMType),
                                            str(spineMType),
                                            str(siteX), str(siteY), str(siteZ),
                                            str(stimR), str(period),
                                            str(boutonFileName), str(spineFileName),
                                            str(bouton_include),
                                            str(spine_include),
                                            str(synapse_include)
                                            ]
                                            )
                    self.spineCount += 1
                    index = self.spineCount
                    dend_dx = parent_x1 - x1
                    dend_dy = parent_y1 - y1
                    dend_dz = parent_z1 - z1
                    dx, dy, dz = getSpineVector(dend_dx, dend_dy, dend_dz, index)
                    #genSpine(array, index, spineMType, x, y, z, dx, dy, dz, period, ageType, branchType)
                    ######################
                    offset = 0.0
                    ### CHECK ANGLE (90-degreee)
                    #p1 = Point3D(x1,y1,z1)
                    #p2 = Point3D(parent_x1, parent_y1, parent_z1)
                    #l1 = Line3D(p1,p2)
                    #p3 = Point3D(0,0,0)
                    #p4 = Point3D(dx*lNeck, dy*lNeck, dz*lNeck)
                    #l2 = Line3D(p3,p4)
                    #print(l1.is_perpendicular(l2), " ", math.degrees(l1.angle_between(l2)))
                    #print (l1.distance())
                    #angle = math.degrees(l1.angle_between(l2))
                    #if (angle < 80 or angle > 99):
                    #    print (angle)
                    ##END CHECK
                    #createBoutonSWC(index, dx, dy, dz, lNeck + 0.1)
                    self.createBoutonSWC(index, dx, dy, dz, 0.0)

                newid = str(len(self.point_lookup)+1)
                newid_dist2soma = float(self.point_lookup[cid]['dist2soma']) - interval
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
                tmplist.append(newid)
            holdpoint_list = list(set(tmplist))
        #print("spines count: ", basalSpineCount)
        self.point_lookup = deepcopy(point_lookupBackUp) # restore

    def _genInhibitory_PL2_3(self, listBranches, mean_symsynapse_distance, stimPeriod,
                          boutonFileName, spineFileName, branchType,
                          spineMType, ageType):
        stimR = 5 # radius of stimulus [um]
        boutonMType = 1
        bouton_include = 0 # I/O on bouton?
        spine_include = 'N/A'
        synapse_include = 0
        # rate of having spine on branchpoint
        #chance2HaveGABAInput = 0.60
        chance2HaveGABAInput = self.chance2HaveGABAInput
        spineFileName = 'N/A'
        spineMType = 'N/A'

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

        holdpoint_list = list(set(holdpoint_list))
        # print(holdpoint_list)
        ############
        # start the spine generation process
        """
        Basically, what the algorithm does is
          1. start with all the distal points
          2. for each point in the list
            generate the next spine using Poisson distribution
            put the coordinate in the spineArray[]
            .... in the holdpoint_list
        """
        point_lookupBackUp = deepcopy(self.point_lookup)
        while holdpoint_list:
            tmplist = []
            for id in holdpoint_list:
                if (int(id) == -1):
                    break
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                dist2soma = float(point_info['dist2soma'])
                #spine_filename = 'spine' +
                pid = parent_id
                cid = id
                if (int (pid) == -1):
                    continue
                if (int(pid) != -1 and
                    not self._branchInList(branchType, listBranches)):
                    tmplist.append(pid)
                    continue

                distance=self.find_distance(cid, pid)

                # Assume spine appearance with adjacent-distance
                # follows exponential distribution
                interval = rnd.exponential(mean_symsynapse_distance)
                #interval = self.nextLocation(mean_symsynapse_distance)
                #print(interval)
                assert(interval > 0)
                #assert(self.spineCount< 10000)
                # Find the pair of points
                while (interval > distance and int(pid) != -1):
                    # update interval and find new cid, pid
                    interval = interval - distance
                    cid = pid
                    point_info = self.point_lookup[cid]
                    pid =  point_info['parent']
                    if (int(pid) == -1):
                        break
                    distance=self.find_distance(cid, pid)
                    dist2branchPoint = float(point_info['dist2branchPoint'])
                    if (dist2branchPoint == 0.0):
                        tmplist.append(cid)
                        break

                if (dist2branchPoint == 0.0 and interval > distance):
                    continue
                if (int(pid) == -1):
                   continue

                #now we ensure the spine in between cid and pid
                x1 = float(self.point_lookup[cid]['siteX'])
                y1 = float(self.point_lookup[cid]['siteY'])
                z1 = float(self.point_lookup[cid]['siteZ'])
                parent_x1 = float(self.point_lookup[pid]['siteX'])
                parent_y1 = float(self.point_lookup[pid]['siteY'])
                parent_z1 = float(self.point_lookup[pid]['siteZ'])
                # generate new point for spine
                siteX = np.interp(distance-interval, [0, distance], [parent_x1, x1])
                siteY = np.interp(distance-interval, [0, distance], [parent_y1, y1])
                siteZ = np.interp(distance-interval, [0, distance], [parent_z1, z1])
                rand = np.random.uniform()
                if (rand < chance2HaveGABAInput):
                # if (1):
                    # limits # spines to be generated
                    period = stimPeriod
                    self.inhibitoryArray.append([str(branchType),
                                            str(boutonMType),
                                            str(spineMType),
                                            str(siteX), str(siteY), str(siteZ),
                                            str(stimR), str(period),
                                            str(boutonFileName), str(spineFileName),
                                            str(bouton_include),
                                            str(spine_include),
                                            str(synapse_include)
                                            ]
                                            )
                    self.spineCount += 1
                    index = self.spineCount
                    dend_dx = parent_x1 - x1
                    dend_dy = parent_y1 - y1
                    dend_dz = parent_z1 - z1
                    dx, dy, dz = getSpineVector(dend_dx, dend_dy, dend_dz, index)
                    #genSpine(array, index, spineMType, x, y, z, dx, dy, dz, period, ageType, branchType)
                    ######################
                    offset = 0.0
                    ### CHECK ANGLE (90-degreee)
                    #p1 = Point3D(x1,y1,z1)
                    #p2 = Point3D(parent_x1, parent_y1, parent_z1)
                    #l1 = Line3D(p1,p2)
                    #p3 = Point3D(0,0,0)
                    #p4 = Point3D(dx*lNeck, dy*lNeck, dz*lNeck)
                    #l2 = Line3D(p3,p4)
                    #print(l1.is_perpendicular(l2), " ", math.degrees(l1.angle_between(l2)))
                    #print (l1.distance())
                    #angle = math.degrees(l1.angle_between(l2))
                    #if (angle < 80 or angle > 99):
                    #    print (angle)
                    ##END CHECK
                    self.createBoutonSWC(index, dx, dy, dz, 0.0)

                newid = str(len(self.point_lookup)+1)
                newid_dist2soma = float(self.point_lookup[cid]['dist2soma']) - interval
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
                tmplist.append(newid)
            holdpoint_list = list(set(tmplist))
        #print("spines count: ", basalSpineCount)
        self.point_lookup = deepcopy(point_lookupBackUp) # restore


    def createSpineSWC(self, index, dx, dy, dz, rHead, rNeck, lNeck, offset):
        """
        create the SWC files for spine
        """
        swcFile = open(self.spineFolder+'/spine_%04.d.swc' % index, 'w')
        point = (rHead+lNeck+offset)
        swcFile.write(' '.join(['1', '1',
                                "{0:.3f}".format(dx*(point)),
                                "{0:.3f}".format(dy*(point)),
                                "{0:.3f}".format(dz*(point)),
                                str(rHead), '-1'])+"\n")
        #point = (lNeck+offset)
        #swcFile.write(' '.join(['2', '3',
        #                        "{0:.3f}".format(dx*(point)),
        #                        "{0:.3f}".format(dy*(point)),
        #                        "{0:.3f}".format(dz*(point)),
        #                        str(rNeck), '1'])+"\n")
        ##swcFile.write(' '.join(['2', '3', str(dx*offset), str(dy*offset), str(dz*offset), str(rNeck), '1'])+"\n")
        #swcFile.write(' '.join(['3', '3',
        #                        "{0:.3f}".format(offset),
        #                        "{0:.3f}".format(offset),
        #                        "{0:.3f}".format(offset),
        #                        str(rNeck), '2'])+"\n")
        #####IMPORTANT
        # to ensure proper spine-attachment
        # the spine must have 1 capsule of neck
        swcFile.write(' '.join(['2', '3',
                                "{0:.3f}".format(offset),
                                "{0:.3f}".format(offset),
                                "{0:.3f}".format(offset),
                                str(rNeck), '1'])+"\n")
        swcFile.close()

    def createBoutonSWC(self, index, dx, dy, dz, offset):
        """
        create the SWC files for  presynaptic neuron (soma + axon)
        """
        presynSomaRadius = 5.0 # (um)
        axonLen = 3.0  #(um)
        swcFile = open(self.spineFolder+'/bouton_%04.d.swc' % index, 'w')
        point = (offset+presynSomaRadius + axonLen)
        swcFile.write(' '.join(['1', '1',
                                "{0:.3f}".format(dx*(point)),
                                "{0:.3f}".format(dy*(point)),
                                "{0:.3f}".format(dz*(point)),
                                '5.0', '-1'])+"\n")
        point = (offset+axonLen)
        swcFile.write(' '.join(['2', '2',
                                "{0:.3f}".format(dx*(point)),
                                "{0:.3f}".format(dy*(point)),
                                "{0:.3f}".format(dz*(point)),
                                '1.0', '1'])+"\n")
        swcFile.write(' '.join(['3', '2',
                                "{0:.3f}".format(dx*offset),
                                "{0:.3f}".format(dy*offset),
                                "{0:.3f}".format(dz*offset),
                                '1.0', '2'])+"\n")
        swcFile.close()

    def getSpineVector(self, dx, dy, dz, orientation):
        num_rotation = 5 # make sure the same as
        angle = (2*pi/5) * ((2*orientation) % 5)
        v1 = [dx,dy,dz] # dendrite vector
        v2 = [dy,dz,dx] # not the dendrite vector
        v3 = np.cross(v1,v2) # perpendicular to the dendrite vector
        v4 = rotateVector(v3,angle,v1) # unit vector for bouton and spine
        return normaliseVector(v4)

    def genboutonspineSWCFiles_PL5b (self):
        """
        read data from 'spines.txt'
        and save to 'neurons.txt' tissue file
        """

        split_lines = readLines('spines.txt')
        offsetIdx = 1
        inputs = []
        for i, line in enumerate(split_lines):
            newInput = {}
            #newInput['index']          = int(line[0])
            newInput['index']          = int(i)+offsetIdx
            newInput['spineFileName']  = 'spines/spine_%04.d.swc' % newInput['index']
            newInput['boutonFileName'] = 'spines/bouton_%04.d.swc' % newInput['index']
            newInput['boutonIndex']    = (newInput['index']+1)*2 - 1
            newInput['spineIndex']     = (newInput['index']+1)*2
            newInput['spineMType']     = line[2]
            newInput['boutonMType']    = line[1]
            newInput['siteX']          = line[3]
            newInput['siteY']          = line[4]
            newInput['siteZ']          = line[5]
            #newInput['siteR']          = 5.0
            newInput['siteR']          = line[6]
            newInput['period']         = line[7]
            newInput['boutonInclude']  = line[8]=='1'
            newInput['spineInclude']   = line[9]=='1'
            newInput['synapseInclude'] = line[10]=='1'
            inputs.append(newInput)
        offsetIdx = i+1

        inputsInhibitory = []
        #split_linesInhibitory = readLines('inhibitory.txt')
        #for i, line in enumerate(split_linesInhibitory):
        #    newInput = {}
        #    #newInput['index']          = int(line[0])
        #    newInput['index']          = int(i)+offsetIdx
        #    newInput['boutonFileName'] = 'spines/bouton_%04.d.swc' % newInput['index']
        #    newInput['boutonIndex']    = newInput['index']
        #    newInput['boutonMType']    = line[1]
        #    newInput['siteX']          = line[3]
        #    newInput['siteY']          = line[4]
        #    newInput['siteZ']          = line[5]
        #    #newInput['siteR']          = 5.0
        #    newInput['siteR']          = line[6]
        #    newInput['period']         = line[7]
        #    newInput['boutonInclude']  = line[8]=='1'
        #    newInput['synapseInclude'] = line[10]=='1'
        #    inputsInhibitory.append(newInput)
        content = {'inputs': inputs, 'inputsInhibitory': inputsInhibitory}
        renderFile("neurons.txt.template_number", self.spineFolder+"/neurons.txt", content)

    def genboutonspineSWCFiles_PL2_3 (self):

        split_lines = readLines('spines.txt')
        offsetIdx = 1
        inputs = []
        for i, line in enumerate(split_lines):
            newInput = {}
            #newInput['index']          = int(line[0])
            newInput['index']          = int(i)+offsetIdx
            newInput['spineFileName']  = 'spines/spine_%04.d.swc' % newInput['index']
            newInput['boutonFileName'] = 'spines/bouton_%04.d.swc' % newInput['index']
            newInput['boutonIndex']    = (newInput['index']+1)*2 - 1
            newInput['spineIndex']     = (newInput['index']+1)*2
            newInput['spineMType']     = line[2]
            newInput['boutonMType']    = line[1]
            newInput['siteX']          = line[3]
            newInput['siteY']          = line[4]
            newInput['siteZ']          = line[5]
            #newInput['siteR']          = 5.0
            newInput['siteR']          = line[6]
            newInput['period']         = line[7]
            newInput['boutonInclude']  = line[8]=='1'
            newInput['spineInclude']   = line[9]=='1'
            newInput['synapseInclude'] = line[10]=='1'
            inputs.append(newInput)
        offsetIdx = i+1

        inputsInhibitory = []
        fileName  = 'inhibitory.txt'
        if os.path.exists((fileName)):
            split_linesInhibitory = readLines('inhibitory.txt')
            for i, line in enumerate(split_linesInhibitory):
                newInput = {}
                #newInput['index']          = int(line[0])
                newInput['index']          = int(i)+offsetIdx
                newInput['boutonFileName'] = 'spines/bouton_%04.d.swc' % newInput['index']
                newInput['boutonIndex']    = newInput['index']
                newInput['boutonMType']    = line[1]
                newInput['siteX']          = line[3]
                newInput['siteY']          = line[4]
                newInput['siteZ']          = line[5]
                #newInput['siteR']          = 5.0
                newInput['siteR']          = line[6]
                newInput['period']         = line[7]
                newInput['boutonInclude']  = line[8]=='1'
                newInput['synapseInclude'] = line[10]=='1'
                inputsInhibitory.append(newInput)
        content = {'inputs': inputs, 'inputsInhibitory': inputsInhibitory}
        renderFile("neurons.txt.template_number", self.spineFolder+"/neurons.txt", content)

    def _branchInList(self, branchType, listBranches):
        """
        Check if a branch type in the list
        """
        result = False
        for i, val in enumerate(listBranches):
            if (int(branchType) == int(val)):
                result = True
                break
        return result

    def _genSpineOnBranch(self, listBranches, mean_spine_distance, stimPeriod,
                          boutonFileName, spineFileName, branchType,
                          spineMType, ageType):
        """

        """
        stimR = 5 # radius of stimulus [um]
        bouton_include = 0 # I/O on bouton?
        spine_include = 0
        synapse_include = 0
        # rate of having spine on branchpoint
        junctionRate = .03
        spineJunctionrate = .61
        mushroomJunctionRate = .21
        chance2HaveSpine = 0.60
        #chance2HaveSpine = self.chance2HaveSpine

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

        holdpoint_list = list(set(holdpoint_list))
        # print(holdpoint_list)
        ############
        # start the spine generation process
        """
        Basically, what the algorithm does is
          1. start with all the distal points
          2. for each point in the list
            generate the next spine using Poisson distribution
            put the coordinate in the spineArray[]
            .... in the holdpoint_list
        """
        point_lookupBackUp = deepcopy(self.point_lookup)
        while holdpoint_list:
            tmplist = []
            for id in holdpoint_list:
                if (int(id) == -1):
                    break
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                dist2soma = float(point_info['dist2soma'])
                #spine_filename = 'spine' +
                pid = parent_id
                cid = id
                if (int (pid) == -1):
                    continue
                if (int(pid) != -1 and
                    not self._branchInList(branchType, listBranches)):
                    tmplist.append(pid)
                    continue

                distance=self.find_distance(cid, pid)

                # Assume spine appearance with adjacent-distance
                # follows exponential distribution
                interval = rnd.exponential(mean_spine_distance)
                #interval = self.nextLocation(mean_spine_distance)
                #print(interval)
                assert(interval > 0)
                #assert(self.spineCount< 10000)
                #assert(self.spineCount< 10000)
                # Find the pair of points
                while (interval > distance and int(pid) != -1):
                    # update interval and find new cid, pid
                    interval = interval - distance
                    cid = pid
                    point_info = self.point_lookup[cid]
                    pid =  point_info['parent']
                    if (int(pid) == -1):
                        break
                    distance=self.find_distance(cid, pid)
                    dist2branchPoint = float(point_info['dist2branchPoint'])
                    if (dist2branchPoint == 0.0):
                        tmplist.append(cid)
                        break

                if (dist2branchPoint == 0.0 and interval > distance):
                    continue
                if (int(pid) == -1):
                   continue

                #now we ensure the spine in between cid and pid
                x1 = float(self.point_lookup[cid]['siteX'])
                y1 = float(self.point_lookup[cid]['siteY'])
                z1 = float(self.point_lookup[cid]['siteZ'])
                parent_x1 = float(self.point_lookup[pid]['siteX'])
                parent_y1 = float(self.point_lookup[pid]['siteY'])
                parent_z1 = float(self.point_lookup[pid]['siteZ'])
                # generate new point for spine
                siteX = np.interp(distance-interval, [0, distance], [parent_x1, x1])
                siteY = np.interp(distance-interval, [0, distance], [parent_y1, y1])
                siteZ = np.interp(distance-interval, [0, distance], [parent_z1, z1])
                if spineMType == self.spineMType["thin"]:
                    boutonMType = self.boutonMType["thin"]
                elif spineMType == self.spineMType["mush"]:
                    boutonMType = self.boutonMType["mush"]
                rand = np.random.uniform()
                if (rand < chance2HaveSpine):
                # if (1):
                    # limits # spines to be generated
                    period = stimPeriod
                    self.spineArray.append([str(branchType),
                                            str(boutonMType),
                                            str(spineMType),
                                            str(siteX), str(siteY), str(siteZ),
                                            str(stimR), str(period),
                                            str(boutonFileName), str(spineFileName),
                                            str(bouton_include),
                                            str(spine_include),
                                            str(synapse_include)
                                            ]
                                            )
                    self.spineCount += 1
                    index = self.spineCount
                    dend_dx = parent_x1 - x1
                    dend_dy = parent_y1 - y1
                    dend_dz = parent_z1 - z1
                    dx, dy, dz = getSpineVector(dend_dx, dend_dy, dend_dz, index)
                    #genSpine(array, index, spineMType, x, y, z, dx, dy, dz, period, ageType, branchType)
                    ######################
                    rNeck = 0.1 # (um)
                    if int(spineMType) == self.spineMType["thin"]: # thin
                        boutonMType = self.boutonMType["thin"]
                        # not significant betweeen apical and basal
                        # based on L2/3 data
                        rHead = 0.591 if ageType == 'young' else 0.615
                        lNeck = 1.897 if ageType == 'young' else 1.802
                        #if int(branchType) == self.branchType["basal"]: # basal
                        #    rHead = 0.1282 if ageType == 'young' else 0.1408
                        #    lNeck = 1.4360 if ageType == 'young' else 1.2663
                        #elif int(branchType) == 4 or int(branchType) == 6: # apical
                        #    rHead = 0.1255 if ageType == 'young' else 0.1360
                        #    lNeck = 1.4289 if ageType == 'young' else 1.3820
                        #else:
                        #    print 'Unknown branch type: ' + str(branchType)
                        #    return
                    elif int(spineMType) == self.spineMType["mush"]: # mushroom
                        boutonMType = self.boutonMType["mush"]
                        # not significant betweeen apical and basal
                        # based on L2/3 data
                        rHead = 0.610 if ageType == 'young' else 0.637
                        lNeck = 1.934 if ageType == 'young' else 1.899
                        #if int(branchType) == 3:
                        #    rHead = 0.2377 if ageType == 'young' else 0.2366
                        #    lNeck = 1.4819 if ageType == 'young' else 1.4476
                        #elif int(branchType) == 4 or int(branchType) == 6: #apical or tufted
                        #    rHead = 0.2358 if ageType == 'young' else 0.2382
                        #    lNeck = 1.4906 if ageType == 'young' else 1.4571
                        #else:
                        #    print 'Unknown branch type: ' + str(branchType)
                        #    return
                    else:
                        print 'Unknown spine type: ' + str(spineMType)
                        return
                    offset = 0.0
                    self.createSpineSWC(index, dx, dy, dz, rHead, rNeck, lNeck,
                                        offset)
                    ### CHECK ANGLE (90-degreee)
                    #p1 = Point3D(x1,y1,z1)
                    #p2 = Point3D(parent_x1, parent_y1, parent_z1)
                    #l1 = Line3D(p1,p2)
                    #p3 = Point3D(0,0,0)
                    #p4 = Point3D(dx*lNeck, dy*lNeck, dz*lNeck)
                    #l2 = Line3D(p3,p4)
                    #print(l1.is_perpendicular(l2), " ", math.degrees(l1.angle_between(l2)))
                    #print (l1.distance())
                    #angle = math.degrees(l1.angle_between(l2))
                    #if (angle < 80 or angle > 99):
                    #    print (angle)
                    ##END CHECK
                    #createBoutonSWC(index, dx, dy, dz, lNeck + 0.1)
                    self.createBoutonSWC(index, dx, dy, dz, lNeck + 0.0)

                newid = str(len(self.point_lookup)+1)
                newid_dist2soma = float(self.point_lookup[cid]['dist2soma']) - interval
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
                tmplist.append(newid)
            holdpoint_list = list(set(tmplist))
        #print("spines count: ", basalSpineCount)
        self.point_lookup = deepcopy(point_lookupBackUp) # restore

    def _genSpineOnBranch_PL2_3(self, listBranches, mean_spine_distance, stimPeriod,
                          boutonFileName, spineFileName, branchType,
                          spineMType, ageType):
        """
        Pyramidal neuron Layer 2/3
        """
        stimR = 5 # radius of stimulus [um]
        bouton_include = 0 # I/O on bouton?
        spine_include = 0
        synapse_include = 0
        # rate of having spine on branchpoint
        junctionRate = .03
        spineJunctionrate = .61
        mushroomJunctionRate = .21
        # chance2HaveSpine = 0.60
        chance2HaveSpine = self.chance2HaveSpine

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

        holdpoint_list = list(set(holdpoint_list))
        # print(holdpoint_list)
        ############
        # start the spine generation process
        """
        Basically, what the algorithm does is
          1. start with all the distal points
          2. for each point in the list
            generate the next spine using Poisson distribution
            put the coordinate in the spineArray[]
            .... in the holdpoint_list
        """
        point_lookupBackUp = deepcopy(self.point_lookup)
        while holdpoint_list:
            tmplist = []
            for id in holdpoint_list:
                if (int(id) == -1):
                    break
                point_info = self.point_lookup[id]
                branchType = point_info['type']
                parent_id =  point_info['parent']
                branchOrder =  point_info['branchOrder']
                dist2branchPoint = float(point_info['dist2branchPoint'])
                dist2soma = float(point_info['dist2soma'])
                #spine_filename = 'spine' +
                pid = parent_id
                cid = id
                if (int (pid) == -1):
                    continue
                if (int(pid) != -1 and
                    not self._branchInList(branchType, listBranches)):
                    tmplist.append(pid)
                    continue

                distance=self.find_distance(cid, pid)

                # Assume spine appearance with adjacent-distance
                # follows exponential distribution
                interval = rnd.exponential(mean_spine_distance)
                #interval = self.nextLocation(mean_spine_distance)
                #print(interval)
                assert(interval > 0)
                assert(self.spineCount< 10000)
                # Find the pair of points
                while (interval > distance and int(pid) != -1):
                    # update interval and find new cid, pid
                    interval = interval - distance
                    cid = pid
                    point_info = self.point_lookup[cid]
                    pid =  point_info['parent']
                    if (int(pid) == -1):
                        break
                    distance=self.find_distance(cid, pid)
                    dist2branchPoint = float(point_info['dist2branchPoint'])
                    if (dist2branchPoint == 0.0):
                        tmplist.append(cid)
                        break

                if (dist2branchPoint == 0.0 and interval > distance):
                    continue
                if (int(pid) == -1):
                   continue

                #now we ensure the spine in between cid and pid
                x1 = float(self.point_lookup[cid]['siteX'])
                y1 = float(self.point_lookup[cid]['siteY'])
                z1 = float(self.point_lookup[cid]['siteZ'])
                parent_x1 = float(self.point_lookup[pid]['siteX'])
                parent_y1 = float(self.point_lookup[pid]['siteY'])
                parent_z1 = float(self.point_lookup[pid]['siteZ'])
                # generate new point for spine
                siteX = np.interp(distance-interval, [0, distance], [parent_x1, x1])
                siteY = np.interp(distance-interval, [0, distance], [parent_y1, y1])
                siteZ = np.interp(distance-interval, [0, distance], [parent_z1, z1])
                if spineMType == self.spineMType["thin"]:
                    boutonMType = self.boutonMType["thin"]
                elif spineMType == self.spineMType["mush"]:
                    boutonMType = self.boutonMType["mush"]
                rand = np.random.uniform()
                if (rand < chance2HaveSpine):
                # if (1):
                    # limits # spines to be generated
                    period = stimPeriod
                    self.spineArray.append([str(branchType),
                                            str(boutonMType),
                                            str(spineMType),
                                            str(siteX), str(siteY), str(siteZ),
                                            str(stimR), str(period),
                                            str(boutonFileName), str(spineFileName),
                                            str(bouton_include),
                                            str(spine_include),
                                            str(synapse_include)
                                            ]
                                            )
                    self.spineCount += 1
                    index = self.spineCount
                    dend_dx = parent_x1 - x1
                    dend_dy = parent_y1 - y1
                    dend_dz = parent_z1 - z1
                    dx, dy, dz = getSpineVector(dend_dx, dend_dy, dend_dz, index)
                    #genSpine(array, index, spineMType, x, y, z, dx, dy, dz, period, ageType, branchType)
                    ######################
                    rNeck = 0.1
                    if int(spineMType) == 2: # thin
                        boutonMType = 4
                        if int(branchType) == 3: # basal
                            rHead = 0.1282 if ageType == 'young' else 0.1408
                            lNeck = 1.4360 if ageType == 'young' else 1.2663
                        elif int(branchType) == 4 or int(branchType) == 6: # apical
                            rHead = 0.1255 if ageType == 'young' else 0.1360
                            lNeck = 1.4289 if ageType == 'young' else 1.3820
                        else:
                            print 'Unknown branch type: ' + str(branchType)
                            return
                    elif int(spineMType) == 3: # mushroom
                        boutonMType = 5
                        if int(branchType) == 3:
                            rHead = 0.2377 if ageType == 'young' else 0.2366
                            lNeck = 1.4819 if ageType == 'young' else 1.4476
                        elif int(branchType) == 4 or int(branchType) == 6: #apical or tufted
                            rHead = 0.2358 if ageType == 'young' else 0.2382
                            lNeck = 1.4906 if ageType == 'young' else 1.4571
                        else:
                            print 'Unknown branch type: ' + str(branchType)
                            return
                    else:
                        print 'Unknown spine type: ' + str(spineMType)
                        return
                    offset = 0.0
                    self.createSpineSWC(index, dx, dy, dz, rHead, rNeck, lNeck,
                                        offset)
                    ### CHECK ANGLE (90-degreee)
                    #p1 = Point3D(x1,y1,z1)
                    #p2 = Point3D(parent_x1, parent_y1, parent_z1)
                    #l1 = Line3D(p1,p2)
                    #p3 = Point3D(0,0,0)
                    #p4 = Point3D(dx*lNeck, dy*lNeck, dz*lNeck)
                    #l2 = Line3D(p3,p4)
                    #print(l1.is_perpendicular(l2), " ", math.degrees(l1.angle_between(l2)))
                    #print (l1.distance())
                    #angle = math.degrees(l1.angle_between(l2))
                    #if (angle < 80 or angle > 99):
                    #    print (angle)
                    ##END CHECK
                    #createBoutonSWC(index, dx, dy, dz, lNeck + 0.1)
                    self.createBoutonSWC(index, dx, dy, dz, lNeck + 0.0)

                newid = str(len(self.point_lookup)+1)
                newid_dist2soma = float(self.point_lookup[cid]['dist2soma']) - interval
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
                tmplist.append(newid)
            holdpoint_list = list(set(tmplist))
        #print("spines count: ", basalSpineCount)
        self.point_lookup = deepcopy(point_lookupBackUp) # restore

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
                  'siteX', 'siteY', 'siteZ', 'stimR', 'period',
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
            inhibitory_data = []
            inhibitory_data = np.hstack((inhibitoryArray, inhibitory_orientation))
            inhibitory_titles = ['branchType', 'boutonType', 'spineType',
                    'siteX', 'siteY', 'siteZ', 'stimR', 'period',
                    'boutonFileName','spineFileName',
                    'bouton_include', 'spine_include', 'synapse_include',
                    'orientation']
            self.inhibitory_output = np.vstack((inhibitory_titles, inhibitory_data))
            self.inhibit_boutonData = inhibitory_data
            self.inhibitory_titles = inhibitory_titles

    def genSpine_PL23(self, use_mean=True):
        """
        Call this function to generate spines with statistics for Pyramidal Layer 2/3 neuron
        OUTPUT:
            self.spineArray[]
            self.inhibitoryArray[]
        ## NOTE:Basically there is not difference in spine morphology between
        #  young and aged rhesus monkey, except that the spine numbers are less
        #  in aged subject
        # NOTE: This changes only found in PL2/3 neurons; but not in PL5
        """
        ##NOTE: focus on young data
        ageType = 'young'
        # ageType = 'aged'

        #############################
        # component of filenames
        bouton = 'bouton_'
        spine = 'spine_'
        thin = 'thin_'
        mush = 'mushroom_'
        apical = '_apical'
        basal = '_basal'
        #define the prefix for *.swc filenames
        boutonTypeThinApical = bouton + thin + ageType + apical
        spineTypeThinApical = spine + thin + ageType + apical
        boutonTypeMushApical = bouton + mush + ageType + apical
        spineTypeMushApical = spine + mush + ageType + apical
        boutonTypeThinBasal = bouton + thin + ageType + basal
        spineTypeThinBasal = spine + thin + ageType + basal
        boutonTypeMushBasal = bouton + mush + ageType + basal
        spineTypeMushBasal = spine + mush + ageType + basal

        ###############################################
        # NOTE: mean distance between spines
        if (ageType == 'young'):
          scaling_factor = 1.4  # for distance scaling [um]
        elif (ageType == 'aged'):
          scaling_factor = 1.8  # for distance scaling [um]
        mean_apical_thin_interval = scaling_factor * 1.6
        mean_apical_mush_interval = scaling_factor * 5.0
        mean_basal_thin_interval = scaling_factor * 1.6
        mean_basal_mush_interval = scaling_factor * 5.0
        ## GABAergic input
        inhFactor = 2.6
        meanApicalInhInterval  = inhFactor*1.0
        meanBasalInhInterval   = inhFactor*1.0

        #basalDistance = 0
        #apicalDistance = 0
        ######################################
        # FILE RESET
        ## remove files
        execute('mkdir -p spines')
        # execute('rm spines/*')
        execute('find spines -maxdepth 1 -name "*.swc" -print0 | xargs -0 rm')

        ###################
        ### STIMULUS
        #period of stimulus signal
        ## APICAL + BASAL DEN
        apical_period = 300
        basal_period = 140
        apical_inhibitory_period = 250
        basal_inhibitory_period = 250

        ###################
        # NOTE: The regular neuron is going to use '0' in MTYPE
        # so we should use a different MTYPE for spine and bouton
        # The different set of configuration that we can use
        self.spineMType = {"thin": 2, "mush": 3}
        self.boutonMType= {"thin":4, "mush":5, "GABA": 1} #as a function of spine-type

        ######
        # Suppose spine occurrence follow Poisson distribution
        # we use the mean adjacent distance for each spine type (thin, mush)
        # to find the location of next spine, starting from the most distal
        # points
        self.spineArray = []
        self.inhibitoryArray = []
        self.spineCount = 0
        ########################
        self.chance2HaveSpine = 1.0
        listBranches = [self.branchType["basal"],]
        self._genSpineOnBranch_PL2_3(listBranches, mean_basal_thin_interval,
                         basal_period, boutonTypeThinBasal,
                         spineTypeThinBasal,
                         self.branchType["basal"],
                         self.spineMType["thin"], ageType
                               )
        self.chance2HaveSpine = 1.0
        listBranches = [self.branchType["basal"],]
        self._genSpineOnBranch_PL2_3(listBranches, mean_basal_mush_interval,
                         basal_period, boutonTypeMushBasal,
                         spineTypeMushBasal,
                         self.branchType["basal"],
                         self.spineMType["mush"], ageType
                               )
        self.chance2HaveSpine = 1.0
        listBranches = [self.branchType["apical"], self.branchType["tufted"]]
        self._genSpineOnBranch_PL2_3(listBranches, mean_apical_thin_interval,
                         apical_period, boutonTypeThinApical,
                         spineTypeThinApical,
                         self.branchType["apical"],
                         self.spineMType["thin"], ageType
                               )
        self.chance2HaveSpine = 1.0
        listBranches = [self.branchType["apical"], self.branchType["tufted"]]
        self._genSpineOnBranch_PL2_3(listBranches, mean_apical_mush_interval,
                         apical_period, boutonTypeMushApical,
                         spineTypeMushApical,
                         self.branchType["apical"],
                         self.spineMType["mush"], ageType
                               )

        self.chance2HaveGABAInput = 1.0
        print("Total excit. spines: ", self.spineCount)
        numSpines = self.spineCount
        boutonTypeGABA = "GABABouton_" + ageType + apical
        listBranches = [self.branchType["apical"], self.branchType["tufted"]]
        self._genInhibitory_PL2_3(listBranches, meanApicalInhInterval,
                            apical_inhibitory_period,
                            boutonTypeGABA,
                            "N/A",
                            self.branchType["apical"],
                            "N/A",
                            ageType
                            )
        boutonTypeGABA = "GABABouton_" + ageType + basal
        listBranches = [self.branchType["basal"],]
        self._genInhibitory_PL2_3(listBranches, meanBasalInhInterval,
                            basal_inhibitory_period,
                            boutonTypeGABA,
                            "N/A",
                            self.branchType["basal"],
                            "N/A",
                            ageType
                            )
        print("Total GABA inputs: ", self.spineCount-numSpines)
        self.rotateSpines()
        self.saveSpines()
        self.genboutonspineSWCFiles_PL2_3()

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
        if (self.inhibitoryArray):
            try:
                inhibitoryFile = open(self.inhibitoryFileName, "w")
                np.savetxt(inhibitoryFile, self.inhibitory_output,
                        fmt='%s %s %s %s %s %s %s %s %s %s %s %s %s %s')
                inhibitoryFile.close()
            except IOError:
                print "Could not open file to write! Please check !" +  self.inhibitoryFileName

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
        statsArray = [str(ageType), str(apicalDistance), str(basalDistance), str(apicalDensityAll), str(
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
        """
        Save thousands of bouton/spine .SWC files to a given folders
        """
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
        spineDistance1 = 0.0#.34 #neck distance to shaft
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
        #siteY1 = (spineDistance1)*(math.cos(degrees))
        #siteZ1 = (spineDistance1)*(math.sin(degrees))
        siteY1 = float(spineDistance1)
        siteZ1 = float(spineDistance1)
        siteX2 = 0.0
        siteY2 = (spineDistance2)*(math.cos(degrees))
        siteZ2 = (spineDistance2)*(math.sin(degrees))
        spineHeadX = siteX2
        spineHeadY = siteY2
        spineHeadZ = siteZ2

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
            #boutonDistance1 = spineDistance2 + boutonHeadRadius
            boutonDistance1 = spineDistance2 #+ boutonHeadRadius
            #boutonDistance1 =  boutonHeadRadius
            boutonDistance2 = boutonDistance1 + boutonHeadRadius

            boutonSiteX2 = 0.0
            boutonSiteY2 = (boutonDistance2)*math.cos(degrees)
            boutonSiteZ2 = (boutonDistance2)*math.sin(degrees)
            #boutonSiteX1 = spineHeadX + boutonHeadRadius
            #boutonSiteY1 = spineHeadY
            #boutonSiteZ1 = spineHeadZ
            boutonSiteX1 = 0.0
            #boutonSiteY1 = (boutonDistance1)
            #boutonSiteZ1 = (boutonDistance1)
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
                           'stimR': line[6],
                           'period': line[7],
                           'boutonFileName': line[8],
                           'spineFileName': line[9],
                           'bouton_include': line[10]=='1',
                           'spine_include': line[11]=='1',
                           'synapse_include': line[12]=='1',
                           'orientation': line[13],
                           'folder': self.swcFolder})

        inputsInhibitory = []
        try:
            linesInhibitory = open(self.inhibitoryFileName).read().splitlines()
            split_linesInhibitory = map(lambda x: x.split(' '), linesInhibitory)


            x = len(open(self.spineFileName).readlines())

            for i, line in enumerate(split_linesInhibitory[1:]):
                inputsInhibitory.append({'name': (i+x),
                                        'bouton_index': ((i+(2*x))-1),
                                        'branchType': line[0],
                                        'boutonType': line[1],
                                        'siteX': line[2],
                                        'siteY': line[3],
                                        'siteZ': line[4],
                                        'stimR': line[5],
                                        'period': line[6],
                                        'boutonFileName': line[7],
                                        'spineFileName': line[8],
                                        'bouton_include': line[9]=='1',
                                        'spine_include': line[10]=='1',
                                        'synapse_include': line[11]=='1',
                                        'folder': self.swcFolder})
        except IOError:
            print "Could not open file to write! Please check !" +  self.inhibitoryFileName
            isContinue = self.getConfirmation()
            assert(isContinue)


        mainNeuronFile = []
        mainNeuronFile.append({'neuronFile': self.swc_filename,
                               }
                              )
        with open("neurons.txt.template", "r") as file:
            txt = file.read()
            newFile = open(self.tissueFileName, "w")
            newFile.write(pystache.render(txt,
                                          { 'mainNeuron': mainNeuronFile,
                                            'inputs': inputs,
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
                           'stimR': line[6],
                           'period': line[7],
                           'boutonFileName': line[8],
                           'spineFileName': line[9],
                           'bouton_include': line[10]=='1',
                           'spine_include': line[11]=='1',
                           'synapse_include': line[12]=='1',
                           'orientation': line[13]})
        inputsInhibitory = []
        try:
            linesInhibitory = open(self.inhibitoryFileName).read().splitlines()
            split_linesInhibitory = map(lambda x: x.split(' '), linesInhibitory)

            x = len(open(self.spineFileName).readlines())

            for i, line in enumerate(split_linesInhibitory[1:]):
                inputsInhibitory.append({'name': (i+x),
                                        'bouton_index': ((i+(2*x))-1),
                                        'branchType': line[0],
                                        'boutonType': line[1],
                                        'siteX': line[2],
                                        'siteY': line[3],
                                        'siteZ': line[4],
                                        'stimR': line[5],
                                        'period': line[6],
                                        'boutonFileName': line[7],
                                        'spineFileName': line[8],
                                        'bouton_include': line[9]=='1',
                                        'spine_include': line[10]=='1',
                                        'synapse_include': line[11]=='1'})
        except IOError:
            print "Could not open file to write! Please check !" +  self.inhibitoryFileName
            isContinue = self.getConfirmation()
            assert(isContinue)
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

        with open("connect_recording_model.gsl.template", "r") as file:
            gsl = file.read()
            targetFile = "./" + "connect_recording_" + modelfile
            newFile = open(targetFile, "w")
            newFile.write(pystache.render(gsl, {'inputs': inputs,
                                                'inputsInhibitory': inputsInhibitory}))

        with open("connect_stimulus_model.gsl.template", "r") as file:
            gsl = file.read()
            targetFile = "./" + "connect_stimulus_" + modelfile
            newFile = open(targetFile, "w")
            newFile.write(pystache.render(gsl, {'inputs': inputs,
                                                'inputsInhibitory': inputsInhibitory}))
    def getConfirmation(self, msg="Continue (y/n)?"):
        # raw_input returns the empty string for "enter"
        yes = set(['yes','y', 'ye', ''])
        no = set(['no','n'])

        print(msg)
        choice = raw_input().lower()
        isValid = False
        while (not isValid):
            if choice in yes:
                return True
            elif choice in no:
                return False
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")

if __name__ == '__main__':
    genSpine = SomeClass('neurons/neuron_test.swc')
