from genSpines import SomeClass

#spines = SomeClass("neurons/neuron_test.swc")
#spines = SomeClass("neurons/neuron.swc")
targetDir = "neurons_msn"
modelFile = "model_Tuan2016.gsl"
spines = SomeClass(targetDir +"/MSN_denda_0.swc")
spines.genSpine_MSN_branchorder_based()
#spines.genSpine_at_branchpoint()
#spines.genSpine_MSN_distance_based()

spines.rotateSpines()
spines.saveSpines()  # default:spines.txt
#spines.genboutonspineSWCFiles_MSN() # defalt location: /neurons/.
spines.genboutonspineSWCFiles_MSN(targetDir) # defalt location: /neurons/.
spines.genTissueText()  # default: neurons.txt
spines.genModelGSL(modelFile)  # 3 model files:  stimulus_<model.gsl>,
              # recording_<model.gsl>
              # connect_<model.gsl>

# print spines.__dict__.keys()
# print spines.point_lookup
#print spines
