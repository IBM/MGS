from genSpines import SomeClass
import sys
targetDir = "neurons"
neuron = SomeClass(targetDir +"/neuron.swc")
########################
## Get the point on the apical dendrite with dist2soma
## within a given range
# from = 615.0 # um
# to = 625.0 # um
# neuron.checkSiteApical(from, to )
dfrom = 50.0 # um
dto = 72.0 # um
#neuron.checkSiteApical(dfrom, dto )
#sys.exit()

# RETURN THE FILE NAME with spinehead at that location
#radius = 5.1
#neuron.getSpineHeadFromFolder(20.25, 607.39,1.049, radius )

#############################################
## DO THIS
# RETURN THE FILE NAME with spinehead at that location
# Use this if offset is put separate in tissuefile
#neuron.getSpineHeadFromTissueFile(20.25, 607.39,1.049, radius )
#neuron.getSpineHeadFromTissueFile(-41, 233, 6, radius )
#neuron.getSpineHeadFromTissueFile(28, 678, -16, radius )

############################
## DO THIS
## if we want to trigger a certain area
##  we only want to add presynaptic neurons on that region to the simulation
## Here, it add an 'x' mark at the end of the line containing the presynaptic neuron
## and we can easily apply removal command to uncomment these lines
# e.g. : (28,678,-16) is the distal region
radius = 10
neuron.getSpineHeadFromTissueFileAndMark(28, 678, -16, radius )

#############################################
## DO THIS
# RETURN THE FILE NAME of the bouton structure with
# the associates spine head at that location
# Use this if offset is put separate in tissuefile
#radius = 5.1
#neuron.getPreSynapticSomaHeadFromTissueFile(20.25, 607.39,1.049, radius )
radius = 5.1
#neuron.getPreSynapticSomaHeadFromTissueFile(28.46, 678.08,-16.08, radius )
radius = 5.1
#neuron.getPreSynapticSomaHeadFromTissueFile(-38.69, 236.84, 4.88, radius )
neuron.getPreSynapticSomaHeadFromTissueFile(-41, 233, 6, radius )

#############################################
## DO THIS
## Help to find out the index of the neuron
#neuron.getIndexOfFileInTissueFile('neurons/neuron.swc_new.swc')
#neuron.getIndexOfFileInTissueFile('spines/bouton_1691.swc' )
