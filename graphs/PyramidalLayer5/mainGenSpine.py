from genSpines import SomeClass

################
#
# Pyramidal neuron
# OPTION 1: full tree
################
##if (0):
##    targetDir = "neurons"
##    #spines = SomeClass("neurons/neuron_test.swc")
##    #spines = SomeClass("neurons/neuron.swc")
##    spines = SomeClass(targetDir +"/hay2_3K.swc")
##    spines.genSpine_PL5_new()

if (1): #use this
    targetDir = "neurons"
    modelFile = "model_somethingHay2011.gsl"
    tissueFile = "neurons_pyramidalL5.txt"
    #spines = SomeClass(targetDir +"/neuron.swc")
    spines = SomeClass(targetDir +"/hay2_3K.swc")
    spines.genSpine_PyramidalL5()  #better

################
#
# Pyramidal neuron
# OPTION 1: trimmed tree
################
targetDir = "neurons"
#spines = SomeClass(targetDir +"/neuron.swc_trimmedBranchMaxDistance.swc")
#spines.genSpine_PyramidalL5()  #better






print("Now is the time to modify spines.txt for what to do I/O: *_include set to 1")
dummy = raw_input("Edit and save the file before press Enter...")
#spines.genModelGSL(modelFile)  # 3 model files:  stimulus_<model.gsl>,
              # recording_<model.gsl>
              # connect_<model.gsl>

# print spines.__dict__.keys()
# print spines.point_lookup
#print spines
