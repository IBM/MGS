from genSpines import SomeClass
from genSpines import branchType
targetDir = "neurons"

##############
# The goal is to modify multiple-point soma into single-point soma
# to be used by IBM NTS
##############
targetDir = "neurons"
#neuron = SomeClass(targetDir +"/hay1.swc_original")
#neuron.reviseSomaSWCFile()

##############
# remove branches
##############
targetDir = "neurons"
neuron = SomeClass(targetDir +"/neuron.swc")
#jindices =[2306, 2203, 1924, 57, 60]
#indices =[2306, 2203, 1924, 60, 2360, 1904, 283, 474, 514, 634 ]
#neuron.removeBranch(indices)
dist = 1.5
#self.removeNearbyPoints(dist, write2File=True,fileSuffix='_revised.swc')
neuron.removeNearbyPointsByDist(dist )



##############
# convert the branch supposed to be apical to apical
# Make sure to update 'startIndex' and 'newBranch'
##############
#targetDir = "neurons"
#neuron = SomeClass(targetDir +"/hay1.swc")
## convert points from line 3442 to apical (4)
##startIndex = [3442]# hay2
#startIndex = [1663] #hay1
#newBranch=4   #apical-den
#neuron.convertBranch(startIndex, newBranch)

##############
# Revise radius of branches
##############
#neuron = SomeClass(targetDir +"/neuron.swc_tufted.swc")
minR = 0.1
#neuron.reviseRadius(minR)

##############
#  Remove branch too far from soma
##############
#neuron = SomeClass(targetDir +"/neuron.swc")
branchType2Find=[
            branchType['soma'],
            branchType['axon'],
            branchType['basal'], ##############
            branchType['apical'],# Remove some branches from the tree
            branchType['AIS'],   ##############
            branchType['tufted'],#neuron = SomeClass(targetDir +"/neuron.swc")
            branchType['bouton'],##lineIndex =[116, 89, 50, 105, 153]
            ]
maxDist2Keep =060
#neuron.removeBranchTooFarFromSoma(branchType2Find, maxDist2Keep)


##############
# Remove some branches (only from branching point) from the tree
# if the proximal-side branchpoint of that branch is outside the sphere
# based on the given max-distance to soma
##############
#neuron = SomeClass(targetDir +"/neuron.swc")
dist=150.0 #[um]
#neuron.removeApicalBranch(dist)
#neuron.removeBasalBranch(dist)
#neuron.removeApicalBasalBranch(dist)


##############
# Remove terminal points in a tree
##############
#neuron = SomeClass(targetDir +"/neuron.swc")
#neuron.removeTerminalPoints()

##############
# Remove points that are adjacent to each other
##############
#neuron = SomeClass(targetDir +"/neuron.swc")
#distcriteria = 0.0
#neuron.removeNearbyPoints(distcriteria)

##############
# Remove points based on line index
##############
#neuron = SomeClass(targetDir +"/neuron.swc")
#lineIndex = [110]
#neuron.removeGivenPoints(lineIndex)


##############
# revise soma + closedpoints
##############
#targetDir = "neurons"
#neuron = SomeClass(targetDir +"/hay2.swc")
#neuron.reviseSWCFile()

