from genSpines import SomeClass

#spines = SomeClass("neurons/neuron_test.swc")
spines = SomeClass("neurons/neuron.swc")
spines.genSpine_MSN_branchorder_based()
#spines.genSpine_at_branchpoint()
#spines.genSpine_MSN_distance_based()

spines.rotateSpines()
spines.saveSpines()
spines.genboutonspineSWCFiles_MSN()
spines.genTissueText()
spines.genModelGSL()

# print spines.__dict__.keys()
# print spines.point_lookup
#print spines
