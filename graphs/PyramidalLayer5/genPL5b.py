from genSpines import SomeClass

##########
# convert to tufted (and AIS) region
##############
targetDir = "neurons"
#neuron = SomeClass(targetDir +"/neuron.swc")
#neuron = SomeClass(targetDir +"/hay2_0.developed.swc")
#neuron.genPL5b()
#neuron = SomeClass(targetDir +"/hay1.swc_original_revisedSoma.swc")
neuron = SomeClass(targetDir +"/hay1.swc_original_revisedSoma.swc_changedBranch.swc")
neuron.genPL5b_hay1()
dist=0.0 #[um]
neuron = SomeClass(targetDir +"/hay1.swc_original_revisedSoma.swc_changedBranch.swc_scaledBasal.swc")
neuron.removeBasalBranch(dist)

##############
# convert the branch supposed to be apical to apical
# Make sure to update 'startIndex' and 'newBranch'
##############
#targetDir = "neurons"
#neuron = SomeClass(targetDir +"/hay1.swc")
## convert points from line 3442 to apical (4)
##startIndex = [3442]# hay2
startIndex = [1663] #hay1
newBranch=4   #apical-den
#neuron.convertBranch(startIndex, newBranch)
