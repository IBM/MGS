from genSpines import SomeClass

#spines = SomeClass("neurons/neuron_test.swc")
#spines = SomeClass("neurons/neuron.swc")
targetDir = "neurons"
modelFile = "model_somethingHay2011.gsl"
tissueFile = "neurons_pyramidalL5.txt"
spines = SomeClass(targetDir +"/neuron.swc")
#spines.genSpine_PL5_new()
spines.genSpine_PL23()

#spines.rotateSpines()
#spines.saveSpines()  # default:spines.txt
print("Now is the time to modify spines.txt for what to do I/O: *_include set to 1")
dummy = raw_input("Edit and save the file before press Enter...")
#spines.genModelGSL(modelFile)  # 3 model files:  stimulus_<model.gsl>,
              # recording_<model.gsl>
              # connect_<model.gsl>

# print spines.__dict__.keys()
# print spines.point_lookup
#print spines
