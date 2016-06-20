from genSpines import SomeClass
###
targetDir = "neurons"
neuron = SomeClass(targetDir +"/neuron.swc")
neuron.genPL5b()

##############
# convert the branch supposed to be apical to apical
# Make sure to update 'startIndex' and 'newBranch'
##############
targetDir = "neurons"
neuron = SomeClass(targetDir +"/hay2.swc_revised.swc")
# convert points from line 3442 to apical (4)
startIndex = [3442]
newBranch=4   #apical-den
neuron.convertBranch(startIndex, newBranch)
