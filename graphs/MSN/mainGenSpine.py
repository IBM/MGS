from genSpines import SomeClass

################
# MSN neuron
################
#spines = SomeClass("neurons/neuron_test.swc")
#spines = SomeClass("neurons/neuron.swc")

### IMPORTANT: use genSpine_MSN_distance_based or genSpine_PyramidalL5
cond = 2
if (cond==1):
    targetDir = "neurons_msn0"
    modelFile = "model_Tuan2016.gsl"
    tissueFile = "neurons_msn.txt"
    spines = SomeClass(targetDir +"/neuron.swc")
    #spines.genSpine_MSN_branchorder_based()
    #spines.genSpine_at_branchpoint()
    spines.genSpine_MSN_distance_based()
    #spines.genSpine_PyramidalL5()  #better
elif (cond==2):
    targetDir = "neurons_msn0"
    modelFile = "model_Tuan2016.gsl"
    tissueFile = "neurons_msn.txt"
    spines = SomeClass(targetDir +"/MSN_denda_0_1180.swc")
    #spines.genSpine_PyramidalL5()  #better
    #spines.genSpine_MSN_branchorder_based()
    spines.genSpine_MSN_distance_based()
    #spines.genSpine_at_branchpoint()

#spines.rotateSpines()
#spines.saveSpines()  # default:spines.txt
#spines.genboutonspineSWCFiles_MSN() # defalt location: /neurons/.
#spines.genboutonspineSWCFiles_MSN(targetDir) # defalt location: /neurons/.
#spines.genTissueText(tissueFile)  # default: neurons.txt
print("Now is the time to modify spines.txt for what to do I/O: *_include set to 1")
dummy = raw_input("Edit and save the file before press Enter...")
#spines.genModelGSL(modelFile)  # 3 model files:  stimulus_<model.gsl>,
              # recording_<model.gsl>
              # connect_<model.gsl>

# print spines.__dict__.keys()
# print spines.point_lookup
#print spines
