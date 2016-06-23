from genSpines import SomeClass
targetDir = "neurons"

##############
# Remove some branches from the tree
##############
#neuron = SomeClass(targetDir +"/neuron.swc_new.swc")
##lineIndex =[116, 89, 50, 105, 153]
#lineIndex =[981, 983]
#neuron.removeBranch(lineIndex)

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
neuron = SomeClass(targetDir +"/neuron.swc_new.swc")
lineIndex = [1510]
neuron.removeGivenPoints(lineIndex)

##############
# The goal is to modify multiple-point soma into single-point soma
# to be used by IBM NTS
##############
#targetDir = "neurons"
#neuron = SomeClass(targetDir +"/hay2.swc")
#neuron.reviseSomaSWCFile()

##############
# revise soma + closedpoints
##############
#targetDir = "neurons"
#neuron = SomeClass(targetDir +"/hay2.swc")
#neuron.reviseSWCFile()

