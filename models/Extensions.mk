#{{{CONSTANT
CONSTANT_MODULES += PointNeuronStimulator \
#}}}

#{{{INTERFACE
INTERFACE_MODULES += IntValueProducer \
	ValueProducer \
	OutputProducer \
	PlasticOutputProducer \
	SpikeProducer \
	PlasticSpikeProducer \
	ThresholdProducer \
	LFPProducer \
	NeurotransmitterIAFProducer \
	AvailableNeurotransmitterIAFProducer \
	SynapticCurrentIAFProducer \
	CaIAFProducer \
	eCBIAFProducer \
	CB1RIAFProducer \
	SORNSpikeProducer \
	WaveProducer \
	FiringRateProducer \
	VoltageIAFProducer \
	MotoneuronSimpleProducer \
	MotoneuronProducer \
	ReceptiveFieldProducer \
	GoodwinProducer \
#}}}

#{{{NODE
NODE_MODULES += LifeNode \
#}}}

#{{{STRUCT
STRUCT_MODULES += Input \
	StructuralInput \
	PlasticInput \
	SpikeInput \
	SynapseInput \
	SORNSynapseInput \
	ModulatedSynapseInput \
	GJInput \
	PSPInput \
	PlasticPSPInput \
	NeurotransmitterIAFInput \
	SynapticCurrentIAFInput \
	eCBIAFInput \
	WaveInput \
	VoltageIAFInput \
	GoodwinInput \
#}}}

#{{{VARIABLE
VARIABLE_MODULES += LifeDataCollector \
#}}}

