from genSpines import SomeClass
targetDir = "neurons"
neuron = SomeClass(targetDir +"/neuron.swc")
# GOAL: remove unused feature: 5=fork point, 6=endpoint
neuron.reviseNeuron()
