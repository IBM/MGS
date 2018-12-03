#{{{CONSTANT
CONSTANT_MODULES += PointNeuronStimulator \
#}}}

#{{{INTERFACE
INTERFACE_MODULES += ValueProducer \
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
	LinskerInfomaxUnit_LNOutputProducer \
	LypFeed \
	DataFeed \
#}}}

#{{{NODE
NODE_MODULES += BengioRateInterneuron \
	BengioRatePyramidal \
	BitmapPhenotype \
	BoutonIAFUnit \
	CleftAstrocyteIAFUnit \
	FileDriverUnit \
	GatedThalamicUnit \
	GatedThalamoCorticalUnit \
	Goodwin \
	IzhikUnit \
	LeakyIAFUnit \
	LifeNode \
	LinskerInfomaxUnit \
	LypCollector \
	MahonUnit \
	MihalasNieburIAFUnit \
	MihalasNieburSynapseIAFUnit \
	MotoneuronUnit \
	PoissonIAFUnit \
	RabinovichWinnerlessUnit \
	SpineIAFUnit \
	SwitchInput \
	ToneUnit \
	TraubIAFUnit \
	WaveDriverUnit \
	ZhengSORNExcUnit \
	ZhengSORNInhUnit \
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
	GTCU_LN_Input \
	LinskerInfomaxUnit_THinput \
	LinskerInfomaxUnit_LNinput \
	NormalizedThalamicInput \
#}}}

#{{{VARIABLE
VARIABLE_MODULES += BoutonIAFUnitDataCollector \
	CleftAstrocyteIAFUnitDataCollector \
	FileDriverUnitDataCollector \
	GatedThalamicUnitDataCollector \
	GatedThalamoCorticalUnitDataCollector \
	GoodwinDataCollector \
	IzhikUnitDataCollector \
	LeakyIAFUnitDataCollector \
	LifeDataCollector \
	LinskerInfomaxUnitDataCollector \
	MahonUnitDataCollector \
	MihalasNieburIAFUnitDataCollector \
	MihalasNieburSynapseIAFUnitDataCollector \
	MotoneuronUnitDataCollector \
	PoissonIAFUnitDataCollector \
	RabinovichWinnerlessUnitDataCollector \
	SpineIAFUnitDataCollector \
	ToneUnitDataCollector \
	TraubIAFUnitDataCollector \
	WaveDriverUnitDataCollector \
	ZhengSORNUnitDataCollector \
	LFPDataAnalyzer \
#}}}

