
from genSpines import SomeClass
targetDir = "neurons"
neuron = SomeClass(targetDir +"/neuron.swc")

distance = 822.0 # um
#distance = 290.0 # um
#distance = 250.0 # um
#distance = 275.0 # um
#distance = 350.0 # um
#distance = 325.0 # um
tolerance = 7.0
neuron.getCapsuleWithGivenDist2Soma(distance, tolerance)
