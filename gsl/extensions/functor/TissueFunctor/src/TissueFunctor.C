// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "TissueFunctor.h"
#include "CG_TissueFunctorBase.h"
#include "LensContext.h"
#include "LayerDefinitionContext.h"
#include "Simulation.h"
#include "Grid.h"
#include "NodeSet.h"
#include "Node.h"
#include "GridLayerDescriptor.h"
#include "NodeDescriptor.h"
#include "ParameterSet.h"
#include "DataItem.h"
#include "DoubleDataItem.h"
#include "FloatDataItem.h"
#include "FloatArrayDataItem.h"
#include "StructDataItem.h"
#include "StringDataItem.h"
#include "DataItemArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "ConstantDataItem.h"
#include "Struct.h"
#include "NodeAccessor.h"
#include "TissueElement.h"
#include "Connector.h"
#include "VectorOstream.h"
#include "BGCartesianPartitioner.h"
#include "CG_CompartmentDimension.h"
#include "CG_BranchData.h"
#include "Granule.h"

#ifdef HAVE_MPI
#include "../../../../../nti/MaxComputeOrder.h"

#include "../../../../../nti/SegmentForceAggregator.h"
#include "../../../../../nti/AllInSegmentSpace.h"
#include "../../../../../nti/TouchDetectTissueSlicer.h"
#include "../../../../../nti/FrontSegmentSpace.h"
#include "../../../../../nti/FrontLimitedSegmentSpace.h"
#include "../../../../../nti/SegmentKeySegmentSpace.h"
#include "../../../../../nti/ANDSegmentSpace.h"
#include "../../../../../nti/ORSegmentSpace.h"
#include "../../../../../nti/NOTSegmentSpace.h"
#include "../../../../../nti/NeuroDevTissueSlicer.h"
#include "../../../../../nti/SynapseTouchSpace.h"
#include "../../../../../nti/AllInTouchSpace.h"
#include "../../../../../nti/Director.h"
#include "../../../../../nti/SegmentForceDetector.h"
#include "../../../../../nti/Communicator.h"
#include "../../../../../nti/Params.h"
#include "../../../../../nti/TissueGrowthSimulator.hpp"

#include "../../../../../nti/LENSTissueSlicer.h"
#include "../../../../../nti/TouchDetector.h"
#include "../../../../../nti/ORTouchSpace.h"
#include "../../../../../nti/ComputeBranch.h"
#include "../../../../../nti/TouchVector.h"
#include "../../../../../nti/TouchAggregator.h"

#include "../../../../../nti/VolumeDecomposition.h"
#include "../../../../../nti/CountableModel.h"

#include "../../../../../nti/Neurogenesis.h"
#include "../../../../../nti/NeurogenParams.h"
#include "../../../../../nti/BoundingSurfaceMesh.h"
#include "../../../../../nti/CompositeSwc.h"

#define PAR_FILE_INDEX 8
#define N_BRANCH_TYPES 3
#define N_COMPARTMENTS(x) int(ceil(double(x)/double(_compartmentSize)))
#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))

//#define INFERIOR_OLIVE

#ifdef INFERIOR_OLIVE
#include "../../../../../nti/InferiorOliveGlomeruliDetector.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <list>
#include <assert.h>
#include <fstream>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <algorithm>

#ifdef USING_CVC
extern void cvc(int, float *, float *, int *, bool);
extern void set_cvc_config(char *);
#endif
#endif

#ifdef HAVE_MPI
TissueContext* TissueFunctor::_tissueContext=0;
int TissueFunctor::_instanceCounter=0;
#endif

#define TRAJECTORY_TYPE 3

TissueFunctor::TissueFunctor() 
  : CG_TissueFunctorBase(), _compartmentSize(0)
#ifdef HAVE_MPI
  , _size(0), _rank(0),
    _nbrGridNodes(0), _channelTypeCounter(0), _electricalSynapseTypeCounter(0), _chemicalSynapseTypeCounter(0),
    _compartmentVariableTypeCounter(0), _junctionTypeCounter(0),
    _preSynapticPointTypeCounter(0), _endPointTypeCounter(0), _junctionPointTypeCounter(0),
    _forwardSolvePointTypeCounter(0), _backwardSolvePointTypeCounter(0), _readFromFile(false)
#endif
{
#ifdef HAVE_MPI
  if (_instanceCounter==0) _tissueContext = new TissueContext();
  ++_instanceCounter;
#endif
}

TissueFunctor::TissueFunctor(TissueFunctor const & f) 
  : CG_TissueFunctorBase(f),
    _compartmentSize(f._compartmentSize)
#ifdef HAVE_MPI
  , _size(f._size), _rank(f._rank),
    _nbrGridNodes(f._nbrGridNodes), _compartmentVariableLayers(f._compartmentVariableLayers), 
    _junctionLayers(f._junctionLayers), _endPointLayers(f._endPointLayers),
    _junctionPointLayers(f._junctionPointLayers),
    _channelLayers(f._channelLayers),
    _electricalSynapseLayers(f._electricalSynapseLayers),
    _chemicalSynapseLayers(f._chemicalSynapseLayers),
    _preSynapticPointLayers(f._preSynapticPointLayers),
    _forwardSolvePointLayers(f._forwardSolvePointLayers),
    _backwardSolvePointLayers(f._backwardSolvePointLayers),
    _indexBranchMap(f._indexBranchMap),
    _branchIndexMap(f._branchIndexMap),
    _indexJunctionMap(f._indexJunctionMap),
    _junctionIndexMap(f._junctionIndexMap),
    _branchForwardSolvePointIndexMap(f._branchForwardSolvePointIndexMap),
    _branchBackwardSolvePointIndexMap(f._branchBackwardSolvePointIndexMap),
    _capsuleCptPointIndexMap(f._capsuleCptPointIndexMap),
    _capsuleJctPointIndexMap(f._capsuleJctPointIndexMap),
    _channelBranchIndices1(f._channelBranchIndices1),
    _channelBranchIndices2(f._channelBranchIndices2),
    _channelJunctionIndices1(f._channelJunctionIndices1),
    _channelJunctionIndices2(f._channelJunctionIndices2),
    _channelTypeCounter(f._channelTypeCounter),
    _electricalSynapseTypeCounter(f._electricalSynapseTypeCounter), 
    _chemicalSynapseTypeCounter(f._chemicalSynapseTypeCounter),
    _compartmentVariableTypeCounter(f._compartmentVariableTypeCounter),
    _junctionTypeCounter(f._junctionTypeCounter),
    _preSynapticPointTypeCounter(f._preSynapticPointTypeCounter),
    _endPointTypeCounter(f._endPointTypeCounter),
    _junctionPointTypeCounter(f._junctionPointTypeCounter),
    _forwardSolvePointTypeCounter(f._forwardSolvePointTypeCounter),
    _backwardSolvePointTypeCounter(f._backwardSolvePointTypeCounter),
    _tissueParams(f._tissueParams),
    _synapseGeneratorMap(f._synapseGeneratorMap),
    _compartmentVariableTypes(f._compartmentVariableTypes),
    _electricalSynapseTypesMap(f._electricalSynapseTypesMap),
    _chemicalSynapseTypesMap(f._chemicalSynapseTypesMap),
    _compartmentVariableTypesMap(f._compartmentVariableTypesMap),
    _junctionTypesMap(f._junctionTypesMap),
    _channelTypesMap(f._channelTypesMap),
    _preSynapticPointTypesMap(f._preSynapticPointTypesMap),
    _endPointTypesMap(f._endPointTypesMap),
    _junctionPointTypesMap(f._junctionPointTypesMap),
    _forwardSolvePointTypesMap(f._forwardSolvePointTypesMap),
    _backwardSolvePointTypesMap(f._backwardSolvePointTypesMap),
    _readFromFile(f._readFromFile),
    _segmentDescriptor(f._segmentDescriptor)
#endif
{
  if (f._layoutFunctor.get()) f._layoutFunctor->duplicate(_layoutFunctor);
  if (f._nodeInitFunctor.get()) f._nodeInitFunctor->duplicate(_nodeInitFunctor);
  if (f._connectorFunctor.get()) f._connectorFunctor->duplicate(_connectorFunctor);
  if (f._probeFunctor.get()) f._probeFunctor->duplicate(_probeFunctor);
  if (f._MGSifyFunctor.get()) f._MGSifyFunctor->duplicate(_MGSifyFunctor);
  if (f._params.get()) f._params->duplicate(_params);
  for (int i=0; i<2; ++i) {
    _generatedChemicalSynapses[i]=f._generatedChemicalSynapses[i];
    _nonGeneratedMixedChemicalSynapses[i]=f._nonGeneratedMixedChemicalSynapses[i];
    _generatedElectricalSynapses[i]=f._generatedElectricalSynapses[i];
  }
  ++_instanceCounter;
}

void TissueFunctor::userInitialize(LensContext* CG_c, String& commandLineArgs1, String& commandLineArgs2, 
				   String& compartmentParamFile, String& channelParamFile, String& synapseParamFile,
				   Functor*& layoutFunctor, Functor*& nodeInitFunctor, 
				   Functor*& connectorFunctor, Functor*& probeFunctor,
				   Functor*& MGSifyFunctor)
{
#ifdef HAVE_MPI
  _size=CG_c->sim->getNumProcesses();
  _rank=CG_c->sim->getRank();  
#endif
  layoutFunctor->duplicate(_layoutFunctor);
  nodeInitFunctor->duplicate(_nodeInitFunctor);
  connectorFunctor->duplicate(_connectorFunctor);
  probeFunctor->duplicate(_probeFunctor);
  MGSifyFunctor->duplicate(_MGSifyFunctor);

#ifdef HAVE_MPI
  String command="NULL ";
  command+=commandLineArgs1;
  if (_tissueContext->_commandLine.parse(command.c_str()) == false) {
    std::cerr<<"Error in simulation specification's commandLineArgs1 string argument, TissueFunctor:"<<std::endl;
    std::cerr<<commandLineArgs1<<std::endl;
    exit(0);
  }
#endif
  
  char paramFilename[256];
  strcpy(paramFilename, _tissueContext->_commandLine.getParamFileName().c_str());
  _tissueParams.readDevParams(paramFilename);
  _compartmentSize=_tissueContext->_commandLine.getCapsPerCpt();

  FILE* data=NULL;
  if (_tissueContext->_commandLine.getBinaryFileName()!="" && !_tissueContext->isInitialized() ) {
    if ( (data = fopen(_tissueContext->_commandLine.getBinaryFileName().c_str(), "rb") ) != NULL) {
#ifdef HAVE_MPI
      _readFromFile=true;
      _tissueContext->_decomposition = new VolumeDecomposition(_rank, data, _size, _tissueContext->_tissue, 
							       _tissueContext->_commandLine.getX(),
							       _tissueContext->_commandLine.getY(),
							       _tissueContext->_commandLine.getZ());
      _tissueContext->readFromFile(data, _size, _rank);
      _tissueContext->setUpCapsules(_tissueContext->_nCapsules, TissueContext::NOT_SET, _rank, MAX_COMPUTE_ORDER);
      _tissueContext->setInitialized();
      fclose(data);
#endif
    }
  }

  _tissueContext->seed(_rank);

  if (!_tissueContext->isInitialized()) {
    _tissueContext->_tissue = new Tissue(_size, _rank);
    neuroGen(&_tissueParams, CG_c);
    MPI_Barrier(MPI_COMM_WORLD);
    neuroDev(&_tissueParams, CG_c);
  }

#ifdef HAVE_MPI
  command="NULL ";
  command+=commandLineArgs2;
  if (_tissueContext->_commandLine.parse(command.c_str()) == false) {
    std::cerr<<"Error in simulation specification's commandLineArgs2 string argument, TissueFunctor:"<<std::endl;
    std::cerr<<commandLineArgs2<<std::endl;
    exit(0);
  }
#endif
  
  strcpy(paramFilename, _tissueContext->_commandLine.getParamFileName().c_str());
  _tissueParams.readDetParams(paramFilename);
  _tissueParams.readCptParams(compartmentParamFile.c_str());
  _tissueParams.readChanParams(channelParamFile.c_str());
  _tissueParams.readSynParams(synapseParamFile.c_str());

  if (!_tissueContext->isInitialized()) {
    touchDetect(&_tissueParams, CG_c);
    _tissueContext->setInitialized();
  }

  if ( (_tissueContext->_commandLine.getOutputFormat()=="b" || _tissueContext->_commandLine.getOutputFormat()=="bt") &&
       _tissueContext->_commandLine.getBinaryFileName()!="" && !_readFromFile && CG_c->sim->isSimulatePass() ) {
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    _tissueContext->writeToFile(_size, _rank);
#endif
  }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void TissueFunctor::neuroGen(Params* params, LensContext* CG_c)
{
#ifdef HAVE_MPI
  double start, now, then;
  start=then=MPI_Wtime();
  bool* somaGenerated=0;
  std::string baseParFileName="NULL", swcFileName="NULL";
  std::vector<double> complexities;
  int neuronBegin=0, neuronEnd=0;
  int nNeuronsGenerated=0;
  RNG rng;
  
  MPI_Barrier(MPI_COMM_WORLD);

  NeuroDevCommandLine& commandLine=_tissueContext->_commandLine;
  int nthreads=commandLine.getNumberOfThreads();
  char tissueFileName[256];
  int tissueFileNameLength=commandLine.getInputFileName().length();
  strcpy(tissueFileName, commandLine.getInputFileName().c_str());
  char* ext=&tissueFileName[tissueFileNameLength-3];
  if (strcmp(ext, "bin")==0) {
    if (_rank==0) std::cerr << "NeuroGen tissue file must be a text file." << std::endl;
    exit(-1);
  }
  if (_rank==0) {
    std::cout << "Tissue file name: "<< tissueFileName << std::endl << std::endl;
    std::cout << "Generating neurons...\n";
  }

  for (int branchType=0; branchType<N_BRANCH_TYPES; ++branchType) { // 0:axon, 1:basal, 2:apical
    std::string btype;
    if (branchType==0) btype="axons";
    else if (branchType==1) btype="denda";
    else if (branchType==2) btype="dendb";
    std::map<std::string, BoundingSurfaceMesh*> boundingSurfaceMap;
    bool stdout=false;
    bool fout=true;;
    double composite=0.0;
    std::string compositeSwcFileName;
    NeurogenParams** ng_params=0;
    char** fileNames=0;

    if (branchType==0) {
      // only need to do the following once per neuron
      double totalComplexity=0.0;
      std::ifstream tissueFile (tissueFileName);
      while ( tissueFile.good()) {
	std::string line;
	getline (tissueFile,line);
	if (line!="" && line.at(0)!='#')  {
	  std::string str = line;
	  std::stringstream strstr(str);
	  std::istream_iterator<std::string> it(strstr);
	  std::istream_iterator<std::string> end;
	  std::vector<std::string> results(it, end);
	  bool genParams=false;
	  if (results.size()>=PAR_FILE_INDEX+N_BRANCH_TYPES) {
	    // use first paramfile in tissue to seed RNG below
	    for (int bt=0; bt<N_BRANCH_TYPES; ++bt) {
	      if (results.at(PAR_FILE_INDEX + bt)!="NULL") {
		baseParFileName=results.at(PAR_FILE_INDEX + bt);
		genParams=true;
		break;
	      }
	    }
	    if (genParams) {
	      double complexity=0.0;
	      std::ifstream testFile(results.at(0).c_str());
	      if (!testFile) {
		if (results.size()>PAR_FILE_INDEX+N_BRANCH_TYPES) {
		  complexity=atof(results.at(PAR_FILE_INDEX+N_BRANCH_TYPES).c_str() );
		}
		else complexity=1.0;
	      }
	      else testFile.close();
	      totalComplexity+=complexity;
	      complexities.push_back(complexity);
	    }
	  }
	}
      }
      tissueFile.close();
      int nNeurons=complexities.size();
      int bufSize=(nNeurons>0) ? nNeurons : 1;
      somaGenerated=new bool[bufSize]; 
      double targetComplexity=totalComplexity/double(_size);
      double runningComplexity=0.0;
      int count=0, divisor=_size;
      bool assigned=false;
      for (int i=0; i<nNeurons; ++i) {
	somaGenerated[i]=false;
	if ((runningComplexity+=complexities[i])>=targetComplexity) {
	  --divisor;
	  neuronBegin=neuronEnd;
	  if (neuronBegin==i || complexities[i]==runningComplexity ||
	       runningComplexity-targetComplexity<targetComplexity-(runningComplexity-complexities[i]) ) {
	    totalComplexity-=runningComplexity;
	    targetComplexity=totalComplexity/divisor;
	    runningComplexity=0.0;
	    neuronEnd=i+1;
	  }
	  else {
	    totalComplexity-=(runningComplexity-complexities[i]);
	    targetComplexity=totalComplexity/divisor;
	    runningComplexity=complexities[i];
	    neuronEnd=i;
	  }
	  if (count==_rank) {
	    assigned=true;
	    break;
	  }
	  ++count;
	}
      }
      if (!assigned) neuronBegin=neuronEnd;
      NeurogenParams ng_params_p(baseParFileName, _rank);
      rng.reSeed(lrandom(ng_params_p._rng), _rank);
      nNeuronsGenerated=neuronEnd-neuronBegin;
    }
    int bufSize=(nNeuronsGenerated>0) ? nNeuronsGenerated : 1;
    ng_params = new NeurogenParams*[bufSize];
    fileNames = new char*[bufSize];
    for (int i=0; i<bufSize; ++i) ng_params[i]=0;
    
    int ln=strlen(tissueFileName);
    std::string statsFileName(tissueFileName);
    std::string parsFileName(tissueFileName);
    statsFileName.erase(ln-4, 4);
    parsFileName.erase(ln-4, 4);

    std::ostringstream statsFileNameStream, parsFileNameStream;
    statsFileNameStream<<statsFileName<<"."<<btype<<".out";
    statsFileName=statsFileNameStream.str();
    parsFileNameStream<<parsFileName<<"."<<btype<<".par";
    parsFileName=parsFileNameStream.str();

    if (composite>0) {
      compositeSwcFileName=tissueFileName;
      compositeSwcFileName.erase(ln-4, 4);
      std::ostringstream compositeSwcFileNameStream;
      compositeSwcFileNameStream<<compositeSwcFileName<<"."<<_rank<<".swc";
      compositeSwcFileName=compositeSwcFileNameStream.str();
    }
    
    int neuronID=0, idx=0;
    std::ifstream tissueFile (tissueFileName);
    if (tissueFile.is_open()) {
      while ( tissueFile.good() ) {
	std::string line;
	getline (tissueFile,line);
	if (line!="" && line.at(0)!='#')  {
	  if (neuronID>=neuronBegin && neuronID<neuronEnd) {
	    std::string str = line;
	    // construct a stream from the string
	    std::stringstream strstr(str);
	    
	    // use stream iterators to copy the stream to the vector as whitespace separated strings
	    std::istream_iterator<std::string> it(strstr);
	    std::istream_iterator<std::string> end;
	    std::vector<std::string> results(it, end);
	    if (results.size()>=PAR_FILE_INDEX+N_BRANCH_TYPES) {
	      fileNames[idx] = new char[results.at(0).length()+1];
	      strcpy(fileNames[idx], results.at(0).c_str());
	      if (complexities[idx]>0 && results.at(PAR_FILE_INDEX+branchType)!="NULL") {
		ng_params[idx] = new NeurogenParams(results.at(PAR_FILE_INDEX+branchType), _rank);
		ng_params[idx]->RandSeed=lrandom(rng);
		ng_params[idx]->_rng.reSeedShared(ng_params[idx]->RandSeed);
		ng_params[idx]->startX = atof(results.at(4).c_str());
		ng_params[idx]->startY = atof(results.at(5).c_str());
		ng_params[idx]->startZ = atof(results.at(6).c_str());
		std::map<std::string, BoundingSurfaceMesh*>::iterator miter = boundingSurfaceMap.find(ng_params[idx]->boundingSurface);
		if (miter==boundingSurfaceMap.end()) boundingSurfaceMap[ng_params[idx]->boundingSurface] = new BoundingSurfaceMesh(ng_params[idx]->boundingSurface);
	      }
	      ++idx;
	    }
	    else if (results.size()>0) {
	      std::cerr<<"Error in Tissue File at neuron "<<idx<<"."<<std::endl;
	      exit(-1);
	    }
	  }
	  neuronID++;
	}
      }
    }
    else {
      std::cerr<<"Cannot open tissue file!"<<tissueFileName<<std::endl;
      exit(0);
    }
    tissueFile.clear();
    tissueFile.close();
    
    Neurogenesis NG(_rank, _size, nthreads, statsFileName, parsFileName, stdout, fout, branchType+2, boundingSurfaceMap);
    NG.run(neuronBegin, nNeuronsGenerated, ng_params, fileNames, somaGenerated);

    if (composite>0 && _rank==0) CompositeSwc(tissueFileName, compositeSwcFileName.c_str(), composite, false);

    for (int nid=0; nid<nNeuronsGenerated; ++nid) {
      delete [] fileNames[nid];
      delete ng_params[nid];	
    }
    delete [] fileNames;
    delete [] ng_params;
    std::map<std::string, BoundingSurfaceMesh*>::iterator miter, mend = boundingSurfaceMap.end();
    for (miter=boundingSurfaceMap.begin(); miter!=mend; ++miter) delete miter->second;
  }
  delete [] somaGenerated;
  now=MPI_Wtime();
  if (_rank==0) printf("\nNeuron generation compute time : %lf\n\n",now-start);
#endif
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void TissueFunctor::neuroDev(Params* params, LensContext* CG_c)
{
  if (_rank==0) printf("Developing tissue...\n");
#ifdef HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD);

  NeuroDevCommandLine& commandLine=_tissueContext->_commandLine;

  bool clientConnect=commandLine.getClientConnect();
#ifdef USING_CVC
  if (clientConnect) set_cvc_config("./cvc.config");
#endif
  
  char inputFilename[256];
  int inputFilenameLength=commandLine.getInputFileName().length();
  strcpy(inputFilename, commandLine.getInputFileName().c_str());
  int nSlicers = commandLine.getNumberOfSlicers();
  if (nSlicers == 0 || nSlicers>_size) nSlicers = _size;
  int nSegmentForceDetectors = commandLine.getNumberOfDetectors();
  if (nSegmentForceDetectors == 0 || nSegmentForceDetectors>_size) nSegmentForceDetectors = _size;
  bool dumpResampledNeurons=( (_tissueContext->_commandLine.getOutputFormat()=="t" || _tissueContext->_commandLine.getOutputFormat()=="bt") ) ? true : false;
  _tissueContext->_neuronPartitioner = new NeuronPartitioner(_rank, inputFilename, commandLine.getResample(), dumpResampledNeurons, commandLine.getPointSpacing());
  char* ext=&inputFilename[inputFilenameLength-3];
  if (strcmp(ext, "bin")==0) _tissueContext->_neuronPartitioner->partitionBinaryNeurons(nSlicers, nSegmentForceDetectors, _tissueContext->_tissue);
  else _tissueContext->_neuronPartitioner->partitionTextNeurons(nSlicers, nSegmentForceDetectors, _tissueContext->_tissue);

  VolumeDecomposition* volumeDecomposition=0;
  int X=commandLine.getX();
  int Y=commandLine.getY();
  int Z=commandLine.getZ();  
  _tissueContext->_decomposition = volumeDecomposition = new VolumeDecomposition(_rank, NULL, _size, _tissueContext->_tissue, X, Y, Z);
   
#ifdef VERBOSE
  if (_rank==0) std::cout<<"Max Branch Order = "<<_tissueContext->_tissue->getMaxBranchOrder()<<std::endl;
#endif

  SegmentForceAggregator* segmentForceAggregator = new SegmentForceAggregator(_rank, nSlicers, _size, _tissueContext->_tissue);  


  AllInTouchSpace detectionTouchSpace;	// OBJECT CHOICE : PARAMETERIZABLE
  SegmentForceDetector *segmentForceDetector = new SegmentForceDetector(_rank, nSlicers, _size, commandLine.getNumberOfThreads(),
									&_tissueContext->_decomposition, &detectionTouchSpace, 
									_tissueContext->_neuronPartitioner, params);

  int maxIterations = commandLine.getMaxIterations();
  if (maxIterations<0) {
    std::cerr<<"max-iterations must be >= 0!"<<std::endl;
    MPI_Finalize();
    exit(0);
  }
   
  double Econ=commandLine.getEnergyCon();
  double dT=commandLine.getTimeStep();
  double E=0,dE=0,En=0;
 
  Communicator* communicator = new Communicator();   
  Director* director = new Director(communicator);
  TissueGrowthSimulator TissueSim(_size, _rank, _tissueContext->_tissue, director, segmentForceDetector, segmentForceAggregator, params, commandLine.getInitialFront());

  AllInSegmentSpace allInSegmentSpace;
 
  FrontSegmentSpace frontSegmentSpace(TissueSim);		        // OBJECT CHOICE : PARAMETERIZABLE
  FrontLimitedSegmentSpace frontLimitedSegmentSpace(TissueSim);		// OBJECT CHOICE : PARAMETERIZABLE

  std::vector<std::pair<std::string, unsigned int> > probeKey;
  probeKey.push_back(std::pair<std::string, unsigned int>(std::string("BRANCHTYPE"), TRAJECTORY_TYPE) );
  SegmentKeySegmentSpace gliaSegmentSpace(probeKey);
  NOTSegmentSpace notGliaSegmentSpace(&gliaSegmentSpace);

  ANDSegmentSpace coveredSegmentSpace(&frontLimitedSegmentSpace, &notGliaSegmentSpace);
  ANDSegmentSpace gliaOnFrontSegmentSpace(&frontSegmentSpace, &gliaSegmentSpace);
  ORSegmentSpace gliaOnFrontFrontLimitedSegmentSpace(&coveredSegmentSpace, &gliaOnFrontSegmentSpace);

  NeuroDevTissueSlicer* neuroDevTissueSlicer = new NeuroDevTissueSlicer(_rank, nSlicers, _size, 
									_tissueContext->_tissue, &_tissueContext->_decomposition,
									&frontSegmentSpace, params, segmentForceDetector->getEnergy());

#ifdef VERBOSE
  if (_rank==0) printf("Maximum Front level = %d\n",TissueSim.getMaxFrontNumber());
#endif
  bool attemptConnect=true;
  unsigned iteration=0;
  int nspheres=0;
  float *positions=0, *radii=0;
  int *types=0;
  double start, now, then;
  start=then=MPI_Wtime();
  director->addCommunicationCouple(neuroDevTissueSlicer, segmentForceDetector);
  volumeDecomposition->resetCriteria(&gliaOnFrontFrontLimitedSegmentSpace);
  neuroDevTissueSlicer->resetSegmentSpace(&coveredSegmentSpace);
  segmentForceDetector->updateCoveredSegments(true);
  director->iterate();
  segmentForceDetector->updateCoveredSegments(false);
  director->addCommunicationCouple(segmentForceDetector, segmentForceAggregator);
  bool grow = TissueSim.AdvanceFront() && maxIterations>0;
  neuroDevTissueSlicer->resetSegmentSpace(&frontSegmentSpace);    

  while (grow) {
#ifdef VERBOSE
    if (_rank==0) printf("Front level %d", TissueSim.getFrontNumber());
    if (!grow && _rank==0) printf(" <FINAL> ");
    if (_rank==0) printf("\n"); 
#endif
    En = 0;
    iteration = 0;
    do {
#ifdef USING_CVC
      if (clientConnect) {
	_tissueContext->_tissue->getVisualizationSpheres(vizSpace, nspheres, positions, radii, types);
	cvc(nspheres, positions, radii, types, attemptConnect);
	attemptConnect=false;
      }
#endif
      /* computeForces is inside this front simulation step, which is equivalent
	 to an entire step through the Director's CommunicationCouple list */
      TissueSim.FrontSimulationStep(iteration,dT,E);
      dE = E - En;
      En = E;
      now=MPI_Wtime();
      if (_rank==0 && iteration<maxIterations) std::cout<<"front = "
			    <<TissueSim.getFrontNumber()<<", begin = "
			    <<iteration<<", E = "
			    <<E<<", dE = "
			    <<dE<<", T = "
			    <<now<<", dT = "
			    <<now-then<<"."<<std::endl;
      then=now;
    } while( fabs(dE) > Econ && iteration < maxIterations);
    if (_rank==0) std::cout<<"front = "
			  <<TissueSim.getFrontNumber()<<", end = "
			  <<iteration<<", E = "
			  <<E<<", dE = "
			  <<dE<<"."<<std::endl;
#ifdef USING_CVC
    attemptConnect=true;
#endif
    director->clearCommunicationCouples();
    director->addCommunicationCouple(neuroDevTissueSlicer, segmentForceDetector);
    volumeDecomposition->resetCriteria(&gliaOnFrontFrontLimitedSegmentSpace);
    neuroDevTissueSlicer->resetSegmentSpace(&coveredSegmentSpace);
    segmentForceDetector->updateCoveredSegments(true);
    director->iterate();
    segmentForceDetector->updateCoveredSegments(false);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(neuroDevTissueSlicer, segmentForceDetector);
    director->addCommunicationCouple(segmentForceDetector, segmentForceAggregator);
    _tissueContext->_tissue->clearSegmentForces();     
    grow = TissueSim.AdvanceFront();
    neuroDevTissueSlicer->resetSegmentSpace(&frontSegmentSpace);    
  }
  volumeDecomposition->resetCriteria(&allInSegmentSpace);
  
  now=MPI_Wtime();
  if (_rank==0) printf("\nTissue development compute time : %lf\n\n",now-start);
  
  FILE* tissueOutFile=0;
  if (_tissueContext->_commandLine.getOutputFormat()=="t" || _tissueContext->_commandLine.getOutputFormat()=="bt") {
    std::string outExtension(".developed");
    if (maxIterations>0) {
      _tissueContext->_tissue->outputTextNeurons(outExtension, 0, 0);
    }
    if (commandLine.getOutputFileName()!="")  {
      int nextToWrite=0, written=0, segmentsWritten=0, globalOffset=0;
      while (nextToWrite<_size) {
	MPI_Allreduce((void*)&written, (void*)&nextToWrite, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);      
	MPI_Allreduce((void*)&segmentsWritten, (void*)&globalOffset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);      
	if (nextToWrite==_rank) {
	  if ( ( tissueOutFile = fopen(commandLine.getOutputFileName().c_str(), (_rank==0) ? "wt" : "at") ) == NULL) {
	    printf("Could not open the output file %s!\n", commandLine.getOutputFileName().c_str());
	    MPI_Finalize();
	    exit(0);
	  }
	  segmentsWritten=_tissueContext->_tissue->outputTextNeurons(outExtension, tissueOutFile, globalOffset);
	  fclose(tissueOutFile);
	  written=1;
	}
      }
    }
  }

  delete segmentForceAggregator;
  delete segmentForceDetector;
  delete communicator;
  delete director;
  delete neuroDevTissueSlicer;
  delete [] positions;
  delete [] radii;
  delete [] types;
#endif
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


void TissueFunctor::touchDetect(Params* params, LensContext* CG_c)
{
#ifdef HAVE_MPI
  double start, now, then;
  start=then=MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  NeuroDevCommandLine& commandLine=_tissueContext->_commandLine;
  int nSlicers = _tissueContext->_neuronPartitioner->getNumberOfSlicers();
  int nTouchDetectors = commandLine.getNumberOfDetectors();
  if (nTouchDetectors == 0) nTouchDetectors = _size;

  bool autapses=false;
  
  SynapseTouchSpace electricalSynapseTouchSpace(SynapseTouchSpace::ELECTRICAL,
						params,
						autapses);
  
  SynapseTouchSpace chemicalSynapseTouchSpace(SynapseTouchSpace::CHEMICAL,
					      params,
					      autapses);
  
  ORTouchSpace detectionTouchSpace(electricalSynapseTouchSpace, chemicalSynapseTouchSpace);

  TouchDetectTissueSlicer* touchDetectTissueSlicer = new TouchDetectTissueSlicer(_rank, nSlicers, nTouchDetectors, 
										 _tissueContext->_tissue, &_tissueContext->_decomposition,
										 _tissueContext, params, MAX_COMPUTE_ORDER);
  TouchSpace* touchCommunicateSpace = 0; // OBJECT CHOICE : PARAMETERIZABLE

  TouchDetector *touchDetector = new TouchDetector(_rank, nSlicers, nTouchDetectors, MAX_COMPUTE_ORDER,
						   commandLine.getNumberOfThreads(), commandLine.getAppositionSamplingRate(),
						   &_tissueContext->_decomposition, &detectionTouchSpace, 
						   touchCommunicateSpace, _tissueContext->_neuronPartitioner, _tissueContext, params);



#ifdef INFERIOR_OLIVE
  GlomeruliDetector* glomeruliDetector=new InferiorOliveGlomeruliDetector(_tissueContext);
#endif

  LENSTissueSlicer* lensTissueSlicer = new LENSTissueSlicer(_rank, nSlicers, nTouchDetectors, _tissueContext, params);
  TouchAggregator* touchAggregator = new TouchAggregator(_rank, nTouchDetectors, _tissueContext);

  Communicator* communicator = new Communicator();
  Director* director = new Director(communicator);

  if (_rank==0) {
    printf("Using %s decomposition.\n\n",commandLine.getDecomposition().c_str());
    printf("Detecting touches...\n\n");
  }

  if (commandLine.getDecomposition()=="volume" || commandLine.getDecomposition()=="cost-volume") {
 
    touchDetectTissueSlicer->sendLostDaughters(false);
#ifdef INFERIOR_OLIVE
    touchDetectTissueSlicer->addTolerance(glomeruliDetector->getGlomeruliSpacing());
#endif
    touchDetector->setPass(TissueContext::FIRST_PASS);
    touchDetector->unique(true);
    //touchDetector->unique(false);
    director->addCommunicationCouple(touchDetectTissueSlicer, touchDetector);
    director->iterate();
    touchDetector->setUpCapsules();

    if (commandLine.getDecomposition()=="cost-volume") {
      touchDetector->detectTouches();
#ifdef INFERIOR_OLIVE
      glomeruliDetector->findGlomeruli(touchDetector->getTouchVector());
#endif
      _tissueContext->rebalance(params, touchDetector->getTouchVector());
      director->clearCommunicationCouples();
      director->addCommunicationCouple(touchDetector, touchAggregator);
      director->iterate();
      touchDetector->resetTouchVector();
      Touch::compare c(0);
      _tissueContext->_touchVector.sort(c);
      director->clearCommunicationCouples();
      director->addCommunicationCouple(touchDetectTissueSlicer, touchDetector);
      touchDetectTissueSlicer->sendLostDaughters(true);
      director->iterate();
      _tissueContext->clearCapsuleMaps();
      touchDetector->setUpCapsules();
      _tissueContext->correctTouchKeys(_rank);
    }
    else {
      touchDetector->resetTouchVector();
      touchDetector->detectTouches();     
      touchDetectTissueSlicer->sendLostDaughters(true);
      director->iterate();
      touchDetector->setUpCapsules();
    }
    touchDetector->resetBufferSize(false);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(lensTissueSlicer, touchDetector);
    director->addCommunicationCouple(touchDetector, touchAggregator);
    director->iterate(); 

    delete touchAggregator;

    touchDetector->resetBufferSize(true);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(touchDetectTissueSlicer, touchDetector);
    director->iterate();

    delete touchDetectTissueSlicer;

    touchDetector->setUpCapsules();
    touchDetector->resetBufferSize(false);
    touchDetector->receiveAtBufferOffset(true);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(lensTissueSlicer, touchDetector);
    director->iterate();
    delete lensTissueSlicer;

    touchDetector->setPass(TissueContext::SECOND_PASS);
    touchDetector->setUpCapsules();

    _tissueContext->_origin = _tissueContext->_capsules;

    delete touchDetector;
    delete communicator;
    delete director;
  }

  else if (commandLine.getDecomposition()=="neuron") {
   
    touchDetectTissueSlicer->sendLostDaughters(false);
    touchDetectTissueSlicer->addCutPointJunctions(false);
    touchDetector->setPass(TissueContext::FIRST_PASS);
    touchDetector->unique(true);
    director->addCommunicationCouple(touchDetectTissueSlicer, touchDetector);
    director->iterate();
    touchDetector->setUpCapsules();

    touchDetector->resetTouchVector();
    touchDetector->detectTouches();

    Decomposition* volumeDecomposition = _tissueContext->_decomposition;
    _tissueContext->_decomposition=_tissueContext->_neuronPartitioner;
    touchDetector->resetBufferSize(false);
    director->iterate();

    touchDetector->resetBufferSize(true);
    _tissueContext->_decomposition=volumeDecomposition;
    director->iterate();
    touchDetector->setUpCapsules();
    int nVolCaps=_tissueContext->_nCapsules;

    _tissueContext->_decomposition=_tissueContext->_neuronPartitioner;
    touchDetector->resetBufferSize(false);
    touchDetector->receiveAtBufferOffset(true);
    director->iterate();

    director->clearCommunicationCouples();
    director->addCommunicationCouple(lensTissueSlicer, touchDetector);
    director->addCommunicationCouple(touchDetector, touchAggregator);
    director->iterate();

    _tissueContext->_origin = _tissueContext->_capsules;
    touchDetector->resetBufferSize(true);
    touchDetector->receiveAtBufferOffset(false);
    _tissueContext->_decomposition=volumeDecomposition;
    director->clearCommunicationCouples();
    director->addCommunicationCouple(touchDetectTissueSlicer, touchDetector);
    director->iterate();
    _tissueContext->clearCapsuleMaps();
    touchDetector->setUpCapsules();
    assert(nVolCaps==_tissueContext->_nCapsules);
    std::map<double, int> firstPassVolumeCapsuleMap, secondPassVolumeCapsuleMap;
    _tissueContext->getCapsuleMaps(firstPassVolumeCapsuleMap, secondPassVolumeCapsuleMap);
  
    _tissueContext->_decomposition=_tissueContext->_neuronPartitioner;
    touchDetector->resetBufferSize(false);
    touchDetector->receiveAtBufferOffset(true);
    director->iterate();
    _tissueContext->clearCapsuleMaps();
    touchDetector->setCapsuleOffset(nVolCaps);
    touchDetector->setUpCapsules();
    
    std::map<double, int> firstPassNeuronCapsuleMap, secondPassNeuronCapsuleMap;
    _tissueContext->getCapsuleMaps(firstPassNeuronCapsuleMap, secondPassNeuronCapsuleMap);

    _tissueContext->resetCapsuleMaps(firstPassVolumeCapsuleMap, secondPassVolumeCapsuleMap);
    touchDetector->setCapsuleOffset(-nVolCaps);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(lensTissueSlicer, touchDetector);
    director->iterate();

    _tissueContext->resetCapsuleMaps(firstPassNeuronCapsuleMap, secondPassNeuronCapsuleMap);
    touchDetector->setCapsuleOffset(nVolCaps);
    touchDetector->setPass(TissueContext::SECOND_PASS);
    touchDetector->setUpCapsules();

    delete touchAggregator;
    delete touchDetectTissueSlicer;
    delete lensTissueSlicer;
    delete touchDetector;
    delete communicator;
    delete director;
    delete volumeDecomposition;
  }
  else {
    std::cerr<<"Unrecognized decomposition : "<<commandLine.getDecomposition()<<std::endl;
    exit(0);
  }

  Touch::compare c(0);
  _tissueContext->_touchVector.sort(c);
  now=MPI_Wtime();
  if (_rank==0) printf("Touch detection compute time : %lf\n\n",now-start);
#endif
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int TissueFunctor::compartmentalize(LensContext* lc, NDPairList* params, 
				     std::string& nodeCategory, std::string& nodeType, 
				     int nodeIndex, int densityIndex)
{
  int rval=1;
  if (nodeCategory=="CompartmentVariables" || nodeCategory=="BranchChannels" || nodeCategory=="JunctionChannels") {
    std::vector<int> size;
    if (nodeCategory=="CompartmentVariables" || nodeCategory=="BranchChannels") {
      ComputeBranch* branch=0;
      if (nodeCategory=="CompartmentVariables") branch=findBranch(nodeIndex, densityIndex, nodeType);
      else {
	std::pair<int, int>& channelBranchIndexPair=_channelBranchIndices1[_channelLayers.size()-1][densityIndex][0];
	branch=findBranch(nodeIndex, channelBranchIndexPair.first, _compartmentVariableTypes[channelBranchIndexPair.second]);
      }
      assert(branch);
      int ncaps=branch->_nCapsules;
      int ncomps=N_COMPARTMENTS(ncaps);
      size.push_back(ncomps);
      rval=ncomps;
      assert(branch->_parent || branch->_daughters.size()>0);

      if (nodeCategory=="CompartmentVariables") {
	DataItemArrayDataItem* dimArray=new DataItemArrayDataItem(size);
	ConstantType* ct=lc->sim->getConstantType("CompartmentDimension");
	std::auto_ptr<DataItem> aptr_cst;
	std::vector<CG_CompartmentDimension*>& dimensions = _tissueContext->_branchDimensionsMap[branch];
	if (dimensions.size()>0) assert(dimensions.size()==ncomps);
	else {
	  for (int i=0, j=ncaps-( (ncaps%_compartmentSize==0)?_compartmentSize:ncaps%_compartmentSize); 
	       i<ncomps; 
	       ++i, j-=_compartmentSize) {
	    // compartment indexing is distal to proximal, while capsule indexing is proximal to distal
	    Capsule* begCap=&branch->_capsules[j];
	    Capsule* endCap=&branch->_capsules[(i==0)?ncaps-1:j+_compartmentSize-1];
	    double radius=0;
	    for (Capsule* capPtr=begCap; capPtr<=endCap; ++capPtr) radius+=capPtr->getRadius();
	    radius/=((endCap-begCap)+1);

	    StructDataItem* dimsDI=getDimension(lc, begCap->getBeginCoordinates(), endCap->getEndCoordinates(), radius, begCap->getDist2Soma());
	    std::auto_ptr<DataItem> dimsDI_ap(dimsDI);
	    NDPair* ndp=new NDPair("dimension", dimsDI_ap);

	    NDPairList dimParams;
	    dimParams.push_back(ndp);
	    ct->getInstance(aptr_cst, dimParams, lc);
	    ConstantDataItem* cdi=dynamic_cast<ConstantDataItem*>(aptr_cst.get());
	    std::auto_ptr<Constant> aptr_dim;
	    cdi->getConstant()->duplicate(aptr_dim);
	    Constant* dim=aptr_dim.release();
	    dimensions.push_back(dynamic_cast<CG_CompartmentDimension*>(dim));	  
	  }
	}
      }
    }
    else size.push_back(1);

    const std::vector<DataItem*>* cpt=extractCompartmentalization(params);

    std::vector<DataItem*>::const_iterator cptiter, cptend=cpt->end();
    NDPairList::iterator ndpiter, ndpend=params->end();
    for (cptiter=cpt->begin(); cptiter!=cptend; ++cptiter) {
      bool foundNDP=false;
      for (ndpiter=params->begin(); ndpiter!=ndpend; ++ndpiter) {
	if ((*ndpiter)->getName()==(*cptiter)->getString()) {
	  foundNDP=true;
	  ArrayDataItem* arrayDI = dynamic_cast<ArrayDataItem*>((*ndpiter)->getDataItem());
	  if (arrayDI == 0) {
	    std::cerr<<"TissueFunctor: "<<*(*cptiter)
		     <<" comparmentalization can only be applied to an array parameter!"<<std::endl;
	    exit(-1);
	  }
	  arrayDI->setDimensions(size);
	  break;
	}
      }
      if (!foundNDP) {
	ArrayDataItem* arrayDI=new FloatArrayDataItem(size);
	std::auto_ptr<DataItem> arrayDI_ap(arrayDI);
	NDPair* ndp=new NDPair((*cptiter)->getString(), arrayDI_ap);
	params->push_back(ndp);
      }
    }
  }
  else if (nodeCategory=="Junctions") {
    Capsule* junctionCapsule=findJunction(nodeIndex, densityIndex, nodeType);
    std::map<Capsule*, CG_CompartmentDimension*>::iterator miter=_tissueContext->_junctionDimensionMap.find(junctionCapsule);
    if (miter==_tissueContext->_junctionDimensionMap.end()) {
      ConstantType* ct=lc->sim->getConstantType("CompartmentDimension");
      std::auto_ptr<DataItem> aptr_cst;
      StructDataItem* dimsDI=getDimension(lc, junctionCapsule->getEndCoordinates(), junctionCapsule->getRadius(), junctionCapsule->getDist2Soma());
      std::auto_ptr<DataItem> dimsDI_ap(dimsDI);
      NDPair* ndp=new NDPair("dimension", dimsDI_ap);
      NDPairList dimParams;
      dimParams.push_back(ndp);
      ct->getInstance(aptr_cst, dimParams, lc);
      ConstantDataItem* cdi=dynamic_cast<ConstantDataItem*>(aptr_cst.get());
      std::auto_ptr<Constant> aptr_dim;
      cdi->getConstant()->duplicate(aptr_dim);
      Constant* dim=aptr_dim.release();
      _tissueContext->_junctionDimensionMap[junctionCapsule]=dynamic_cast<CG_CompartmentDimension*>(dim);      
    }
  }
  return rval;
}

std::vector<DataItem*> const * TissueFunctor::extractCompartmentalization(NDPairList* params)
{
  const std::vector<DataItem*>* cpt;
  NDPairList::iterator ndpiter, ndpend=params->end();
  for (ndpiter=params->begin(); ndpiter!=ndpend; ++ndpiter) {
    if ((*ndpiter)->getName()=="compartmentalize") {
      DataItemArrayDataItem* cptDI = dynamic_cast<DataItemArrayDataItem*>((*ndpiter)->getDataItem());
      if (cptDI == 0) {
	std::cerr<<"TissueFunctor: compartmentalization parameter is not a list of parameter names!"<<std::endl;
	exit(-1);
      }
      cpt=cptDI->getDataItemVector();    
      params->erase(ndpiter);
      break;
    }
  }
  return cpt;
}

StructDataItem* TissueFunctor::getDimension(LensContext* lc, double* cds, double radius, double dist2soma)
{
  DoubleDataItem* xddi=new DoubleDataItem(cds[0]);
  std::auto_ptr<DataItem> xddi_ap(xddi);
  NDPair* x=new NDPair("x", xddi_ap);
 
  DoubleDataItem* yddi=new DoubleDataItem(cds[1]);
  std::auto_ptr<DataItem> yddi_ap(yddi);
  NDPair* y=new NDPair("y", yddi_ap);

  DoubleDataItem* zddi=new DoubleDataItem(cds[2]);
  std::auto_ptr<DataItem> zddi_ap(zddi);
  NDPair* z=new NDPair("z", zddi_ap);

  DoubleDataItem* rddi=new DoubleDataItem(radius);
  std::auto_ptr<DataItem> rddi_ap(rddi);
  NDPair* r=new NDPair("r", rddi_ap);

  DoubleDataItem* d2sddi=new DoubleDataItem(dist2soma);
  std::auto_ptr<DataItem> d2sddi_ap(d2sddi);
  NDPair* d2s=new NDPair("dist2soma", d2sddi_ap);

  NDPairList dimList;
  dimList.push_back(x);
  dimList.push_back(y);
  dimList.push_back(z);
  dimList.push_back(r);
  dimList.push_back(d2s);

  StructType* st=lc->sim->getStructType("DimensionStruct");
  std::auto_ptr<Struct> dims;
  st->getStruct(dims);
  dims->initialize(dimList);
  StructDataItem* dimsDI=new StructDataItem(dims);
  return dimsDI;
}

StructDataItem* TissueFunctor::getDimension(LensContext* lc, double* cds1, double* cds2, double radius, double dist2soma)
{
  double center[3];
  double dsqrd=0;
  for (int i=0; i<3; ++i) {
    center[i]=(cds1[i]+cds2[i])/2.0;
    double d=(cds1[i]-cds2[i]);
    dsqrd+=d*d;
  }
  dist2soma+=0.5*sqrt(dsqrd);
  return getDimension(lc, center, radius, dist2soma);
}

void TissueFunctor::getNodekind(const NDPairList* ndpl, std::vector<std::string>& nodekind)
{
  nodekind.clear();
  assert(ndpl);
  NDPairList::const_iterator ndpiter, ndpend=ndpl->end();
  for (ndpiter=ndpl->begin(); ndpiter!=ndpend; ++ndpiter) {
    if ((*ndpiter)->getName()=="nodekind") {
      StringDataItem* nkDI = dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
      if (nkDI == 0) {
	std::cerr<<"TissueFunctor: nodekind parameter is not a string!"<<std::endl;
	exit(-1);
      }
      std::string kind=nkDI->getString();
      char* ckind=new char[kind.size()+1];
      strcpy(ckind, kind.c_str());
      char* p = strtok(ckind, "][");
      while (p!=0) {
	nodekind.push_back(std::string(p));
	p = strtok(0, "][");
      }
      break;
    }
  }
}

ComputeBranch* TissueFunctor::findBranch(int nodeIndex, int densityIndex, std::string const & nodeType)
{
  ComputeBranch* rval=0;
  std::map<std::string, std::map<int, std::map<int, ComputeBranch*> > >::iterator mapiter1=
    _indexBranchMap.find(nodeType);
  if (mapiter1==_indexBranchMap.end()) {
    std::cerr<<"Tissue Functor::findBranch, branch node type "<<nodeType<<" not found in Branch Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  std::map<int, std::map<int, ComputeBranch*> >::iterator mapiter2=mapiter1->second.find(nodeIndex);
  if (mapiter2==mapiter1->second.end()) {
    std::cerr<<"Tissue Functor::findBranch, branch index not found in Branch Index Map! rank="<<_rank
	     <<", nodeIndex (failed)="<<nodeIndex<<", densityIndex="<<densityIndex<<std::endl;
    exit(0);
  }
  std::map<int, ComputeBranch*>::iterator mapiter3=mapiter2->second.find(densityIndex);
  if (mapiter3==mapiter2->second.end()) {
    std::cerr<<"Tissue Functor::findBranch, branch density index not found in Branch Map! rank="<<_rank
	     <<", nodeIndex="<<nodeIndex<<", densityIndex="<<densityIndex<<std::endl;
    exit(0);
  }
  rval=mapiter3->second;
  return rval;
}

std::vector<int>& TissueFunctor::findBranchIndices(ComputeBranch* b, std::string const & nodeType)
{
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >::iterator mapiter1=
    _branchIndexMap.find(nodeType);
  if (mapiter1==_branchIndexMap.end()) {
    std::cerr<<"Tissue Functor::findBranchIndices, branch node type "<<nodeType<<" not found in Branch Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  std::map<ComputeBranch*, std::vector<int> >::iterator mapiter2=mapiter1->second.find(b);
  if (mapiter2==mapiter1->second.end()) {
    std::cerr<<"Tissue Functor::findBranchIndices, branch indices not found in Branch Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  return mapiter2->second;
}

Capsule* TissueFunctor::findJunction(int nodeIndex, int densityIndex, std::string const & nodeType)
{
  std::map<std::string, std::map<int, std::map<int, Capsule*> > >::iterator mapiter1=
    _indexJunctionMap.find(nodeType);
  if (mapiter1==_indexJunctionMap.end()) {
    std::cerr<<"Tissue Functor::findJunction, junction node type "<<nodeType<<" not found in Junction Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  std::map<int, std::map<int, Capsule*> >::iterator mapiter2=
    mapiter1->second.find(nodeIndex);
  if (mapiter2==mapiter1->second.end()) {
    std::cerr<<"Tissue Functor::findJunction, junction index not found in Junction Map! rank="<<_rank
	     <<", nodeIndex="<<nodeIndex<<", densityIndex="<<densityIndex<<std::endl;
    exit(0);
  }
  std::map<int, Capsule*>::iterator mapiter3=
    mapiter2->second.find(densityIndex);
  if (mapiter3==mapiter2->second.end()) {
    std::cerr<<"Tissue Functor::findJunction, junction density index not found in Junction Map! rank="<<_rank
	     <<", nodeIndex="<<nodeIndex<<", densityIndex="<<densityIndex<<std::endl;
    exit(0);
  }
  return mapiter3->second;
}

std::vector<int>& TissueFunctor::findJunctionIndices(Capsule* c, std::string const & nodeType)
{  
  std::map<std::string, std::map<Capsule*, std::vector<int> > >::iterator mapiter1=_junctionIndexMap.find(nodeType);
  if (mapiter1==_junctionIndexMap.end()) {
    std::cerr<<"Tissue Functor::findJunctionIndices, junction type not found in Junction Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  std::map<Capsule*, std::vector<int> >::iterator mapiter2=mapiter1->second.find(c);
  if (mapiter2==mapiter1->second.end()) {
    std::cerr<<"Tissue Functor::findJunction, junction not found in Junction Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  return mapiter2->second;
}

std::vector<int>& TissueFunctor::findForwardSolvePointIndices(ComputeBranch* b, std::string& nodeType)
{
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >::iterator mapiter1=
    _branchForwardSolvePointIndexMap.find(nodeType);
  if (mapiter1==_branchForwardSolvePointIndexMap.end()) {
    std::cerr<<"Tissue Functor: forward solve point node type "<<nodeType<<" not found in Forward Solve Point Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  std::map<ComputeBranch*, std::vector<int> >::iterator mapiter2=mapiter1->second.find(b);
  if (mapiter2==mapiter1->second.end()) {
    std::cerr<<"Tissue Functor: forward solve point index not found in Forward Solve Point Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  return mapiter2->second;
}

std::vector<int>& TissueFunctor::findBackwardSolvePointIndices(ComputeBranch* b, std::string& nodeType)
{
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >::iterator mapiter1=
    _branchBackwardSolvePointIndexMap.find(nodeType);
  if (mapiter1==_branchBackwardSolvePointIndexMap.end()) {
    std::cerr<<"Tissue Functor: backward solve point node type "<<nodeType<<" not found in Branch Backward Solve Point Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  std::map<ComputeBranch*, std::vector<int> >::iterator mapiter2=mapiter1->second.find(b);
  if (mapiter2==mapiter1->second.end()) {
    std::cerr<<"Tissue Functor: backward solve point index not found in Branch Backward Solve Point Index Map! rank="<<_rank<<std::endl;
    exit(0);
  }
  return mapiter2->second;
}

void TissueFunctor::connect(Simulation* sim, Connector* connector, NodeDescriptor* from, NodeDescriptor* to, NDPairList& ndpl)
{
  std::auto_ptr<ParameterSet> inAttrPSet, outAttrPSet;
  to->getGridLayerDescriptor()->getNodeType()->getInAttrParameterSet(inAttrPSet);
  from->getGridLayerDescriptor()->getNodeType()->getOutAttrParameterSet(outAttrPSet);
  inAttrPSet->set(ndpl);
  connector->nodeToNode(from, outAttrPSet.get(), to, inAttrPSet.get(), sim);
}

std::auto_ptr<Functor> TissueFunctor::userExecute(LensContext* CG_c, String& tissueElement, NDPairList*& params) 
{
  params->duplicate(_params);
  std::auto_ptr<Functor> rval; 

  if (tissueElement=="Layout") {
    TissueElement* element=dynamic_cast<TissueElement*>(_layoutFunctor.get());
    if (element==0) {
      std::cerr<<"Functor passed to TissueFunctor as argument 4 is not a TissueElement!"<<std::endl;
      exit(-1);
    }
    element->setTissueFunctor(this);
    _layoutFunctor->duplicate(rval);
  }
  else if (tissueElement=="NodeInit") {
    TissueElement* element=dynamic_cast<TissueElement*>(_nodeInitFunctor.get());
    if (element==0) {
      std::cerr<<"Functor passed to TissueFunctor as argument 5 is not a TissueElement!"<<std::endl;
      exit(-1);
    }
    element->setTissueFunctor(this);
    _nodeInitFunctor->duplicate(rval);
  }
  else if (tissueElement=="Connector") {
    TissueElement* element=dynamic_cast<TissueElement*>(_connectorFunctor.get());
    if (element==0) {
      std::cerr<<"Functor passed to TissueFunctor as argument 6 is not a TissueElement!"<<std::endl;
      exit(-1);
    }
    element->setTissueFunctor(this);
    _connectorFunctor->duplicate(rval);
  }
  else if (tissueElement=="Probe") {
    TissueElement* element=dynamic_cast<TissueElement*>(_probeFunctor.get());
    if (element==0) {
      std::cerr<<"Functor passed to TissueFunctor as argument 7 is not a TissueElement!"<<std::endl;
      exit(-1);
    }
    element->setTissueFunctor(this);
    _probeFunctor->duplicate(rval);
  }
  else if (tissueElement=="MGSify") {
    TissueElement* element=dynamic_cast<TissueElement*>(_MGSifyFunctor.get());
    if (element==0) {
      std::cerr<<"Functor passed to TissueFunctor as argument 8 is not a TissueElement!"<<std::endl;
      exit(-1);
    }
    element->setTissueFunctor(this);
    _MGSifyFunctor->duplicate(rval);
  }
  else if (tissueElement=="Connect") {
    doConnector(CG_c);
  }
  else if (tissueElement=="Tissue->MGS") {
    doMGSify(CG_c);
  }
  else {
    std::cerr<<"Unrecognized tissue element specifier: "<<tissueElement<<std::endl;
    exit(0);
  }
  return rval;
}

ShallowArray< int > TissueFunctor::doLayout(LensContext* lc)
{
  assert(_params.get());

  std::vector<std::string> nodekind;
  getNodekind(_params.get(), nodekind);
  assert(nodekind.size()>0);
  std::string& nodeCategory=nodekind[0];
  std::string nodeType="";
  int nodeComputeOrder=-1;
  if (nodekind.size()>1) nodeType=nodekind[1];
  if (nodekind.size()>2) nodeComputeOrder=atoi(nodekind[2].c_str());
  if (nodeCategory!="CompartmentVariables" && nodeCategory!="Junctions" 
      && nodeCategory!="EndPoints" && nodeCategory!="JunctionPoints"
      && nodeCategory!="ForwardSolvePoints" && nodeCategory!="BackwardSolvePoints"
      && nodeCategory!="Channels" && nodeCategory!="ElectricalSynapses"
      && nodeCategory!="ChemicalSynapses" && nodeCategory!="PreSynapticPoints"
      ) {
    std::cerr<<"Unrecognized nodeCategory parameter on Layer : "<<nodeCategory<<std::endl;
    exit(0);
  }

  if (nodeCategory=="Channels") {
    _channelBranchIndices1.push_back(std::vector<std::vector<std::pair<int, int> > >() );
    _channelJunctionIndices1.push_back(std::vector<std::vector<std::pair<int, int> > >() );
    _channelBranchIndices2.push_back(std::vector<std::vector<std::pair<int, int> > >() );
    _channelJunctionIndices2.push_back(std::vector<std::vector<std::pair<int, int> > >() );
    assert(_channelTypesMap.find(nodeType)==_channelTypesMap.end());
    _channelTypesMap[nodeType]=_channelTypeCounter;
  }

  if (nodeCategory=="CompartmentVariables") {
    assert(_compartmentVariableTypesMap.find(nodeType)==_compartmentVariableTypesMap.end());
    assert(_compartmentVariableTypesMap.size()==_compartmentVariableTypeCounter);
    _compartmentVariableTypesMap[nodeType]=_compartmentVariableTypeCounter;
    _compartmentVariableTypes.push_back(nodeType);
  }

  if (nodeCategory=="Junctions") {
    assert(_junctionTypesMap.find(nodeType)==_junctionTypesMap.end());
    _junctionTypesMap[nodeType]=_junctionTypeCounter;
  }

  if (nodeCategory=="EndPoints") {
    assert(_endPointTypesMap.find(nodeType)==_endPointTypesMap.end());
    _endPointTypesMap[nodeType]=_endPointTypeCounter;
  }

  if (nodeCategory=="JunctionPoints") {
    assert(_junctionPointTypesMap.find(nodeType)==_junctionPointTypesMap.end());
    _junctionPointTypesMap[nodeType]=_junctionPointTypeCounter;
  }

  if (nodeCategory=="ForwardSolvePoints") {
    assert(_forwardSolvePointTypesMap.find(nodeComputeOrder)==_forwardSolvePointTypesMap.end() ||
	   _forwardSolvePointTypesMap[nodeComputeOrder].find(nodeType)==_forwardSolvePointTypesMap[nodeComputeOrder].end());
    _forwardSolvePointTypesMap[nodeComputeOrder][nodeType]=_forwardSolvePointTypeCounter;
  }

  if (nodeCategory=="BackwardSolvePoints") {
    assert(_backwardSolvePointTypesMap.find(nodeComputeOrder)==_backwardSolvePointTypesMap.end() ||
	   _backwardSolvePointTypesMap[nodeComputeOrder].find(nodeType)==_backwardSolvePointTypesMap[nodeComputeOrder].end());
    _backwardSolvePointTypesMap[nodeComputeOrder][nodeType]=_backwardSolvePointTypeCounter;
  }

  bool electrical=(nodeCategory=="ElectricalSynapses" && _tissueParams.electricalSynapses());
  bool chemical=( nodeCategory=="ChemicalSynapses" && _tissueParams.chemicalSynapses());
  bool point=( nodeCategory=="PreSynapticPoints" && _tissueParams.chemicalSynapses());

  ShallowArray<int> rval; 
  Grid* grid=lc->layerContext->grid;
  if (_nbrGridNodes==0) _nbrGridNodes=grid->getNbrGridNodes();
  else if (_nbrGridNodes!=grid->getNbrGridNodes()) {
    std::cerr<<"Error, number of grid nodes has changed! "
	     <<_nbrGridNodes<<"!="<<grid->getNbrGridNodes()<<std::endl;
    assert(0);
  }
  rval.assign(_nbrGridNodes,0);

  int counter=0;
  if (electrical) {
    _electricalSynapseTypesMap[nodeType] = counter = _electricalSynapseTypeCounter;
  }
  else if (chemical) {
    _chemicalSynapseTypesMap[nodeType] = counter = _chemicalSynapseTypeCounter;
  }
  else if (point) {
    assert(_preSynapticPointTypesMap.find(nodeType)==_preSynapticPointTypesMap.end());
    _preSynapticPointTypesMap[nodeType]=counter=_preSynapticPointTypeCounter;
  }

  if (electrical || chemical || point) {
    for (int direction=0; direction<=1; ++direction) {
      
      TouchVector::TouchIterator titer=_tissueContext->_touchVector.begin(), 
	tend=_tissueContext->_touchVector.end();
      for (; titer!=tend; ++titer) {
	if (!_tissueContext->isLensTouch(*titer, _rank)) continue;
	double key1, key2;
	if (direction==0) {
	  key1=titer->getKey1();
	  key2=titer->getKey2();
	}
	else {
	  key1=titer->getKey2();
	  key2=titer->getKey1();
	}
	Capsule* preCapsule=&_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key1)];
	Capsule* postCapsule=&_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key2)];
	ComputeBranch* postBranch=postCapsule->getBranch();
	assert(postBranch);

	unsigned int indexPre, indexPost;
	bool preJunction=false;
	bool postJunction=false;

	if (_segmentDescriptor.getFlag(key1) && 
	    _tissueContext->isTouchToEnd(*preCapsule,*titer) ) {
	  // pre component is LENS junction
	  if (point && _capsuleJctPointIndexMap[nodeType].find(preCapsule)!=_capsuleJctPointIndexMap[nodeType].end()) continue;
	  preJunction=true;
	  indexPre=_tissueContext->getRankOfEndPoint(preCapsule->getBranch());
	}
	else {
	  // pre component is LENS branch
	  if (point && _capsuleCptPointIndexMap[nodeType].find(preCapsule)!=_capsuleCptPointIndexMap[nodeType].end()) continue;
	  preJunction=false;
	  indexPre=_tissueContext->getRankOfBeginPoint(preCapsule->getBranch());
	}

	std::vector<double> probabilities;

	if (point) {
	  std::list<std::string>& synapseTypes=_tissueParams.getPreSynapticPointSynapseTypes(nodeType);
	  std::list<std::string>::iterator synIter, synEnd=synapseTypes.end();
	  for (synIter=synapseTypes.begin(); synIter!=synEnd; ++synIter) {
	    if (isPointRequired(titer, direction, *synIter) ) {
	      probabilities.push_back(1.0);
	      break;
	    }
	    if (probabilities.size()>0) break;
	  }
	}
	else if (electrical) getElectricalSynapseProbabilities(probabilities, titer, direction, nodeType);
	else if (chemical) getChemicalSynapseProbabilities(probabilities, titer, direction, nodeType);
	else assert(0);

	for (int i=0; i<probabilities.size(); ++i) {
	  if (probabilities[i]>0 ) {
	    if ( _tissueContext->isTouchToEnd(*postCapsule,*titer) &&
		 _segmentDescriptor.getFlag(postCapsule->getKey())) {
	      // post component is LENS junction
	      postJunction=true;
	      indexPost=_tissueContext->getRankOfEndPoint(postBranch);
	      Sphere postEndSphere;
	      postCapsule->getEndSphere(postEndSphere);
	      // assert(indexPost==_tissueContext->_decomposition->getRank(postEndSphere));
	    }
	    else {
	      // post component is LENS branch
	      postJunction=false;
	      indexPost=_tissueContext->getRankOfBeginPoint(postBranch);
	      // assert(indexPost==_tissueContext->_decomposition->getRank(postCapsule->getSphere()));
	    }
	    //assert(indexPre==_rank || indexPost==_rank);
	    if (indexPre==_rank || indexPost==_rank) {
	      if (point) {
		if (preJunction) _capsuleJctPointIndexMap[nodeType][preCapsule]=rval[indexPre];
		else _capsuleCptPointIndexMap[nodeType][preCapsule]=rval[indexPre];
		rval[indexPre]++;
	      }
	      else {
		if (electrical) {
		  if (probabilities[i]>=drandom(findSynapseGenerator(indexPre, indexPost) ) ) {
		    rval[indexPre]++;
		    rval[indexPost]++;
		    setGenerated(_generatedElectricalSynapses[direction], titer, counter, i);
		  }
		}
		else if (chemical) {
		  if (probabilities[i]>=drandom(findSynapseGenerator(indexPre, indexPost) ) ) {
		    rval[indexPost]++;
		    setGenerated(_generatedChemicalSynapses[direction], titer, counter, i);
		  }
		  else setNonGenerated(titer, direction, nodeType, i);
		}
	      }
	    }
	  }
	}
      }
    }
  }


  std::map<unsigned int, std::vector<ComputeBranch*> >::iterator 
    mapIter, mapEnd=_tissueContext->_neurons.end();
  for (mapIter=_tissueContext->_neurons.begin(); mapIter!=mapEnd; ++mapIter) {
    std::vector<ComputeBranch*>& branches=mapIter->second;
    std::vector<ComputeBranch*>::iterator iter, end=branches.end();
    for (iter=branches.begin(); iter!=end; ++iter) {
      Capsule* branchCapsules=(*iter)->_capsules;
      int nCapsules=(*iter)->_nCapsules;
      unsigned int index, indexJct;
      double key=branchCapsules[0].getKey();
      if (nodeCategory=="Channels" || _tissueParams.isCompartmentVariableTarget(key, nodeType) ) {
	unsigned int computeOrder=_segmentDescriptor.getComputeOrder(key);
	unsigned int branchOrder=_segmentDescriptor.getBranchOrder(key);

	index=_tissueContext->getRankOfBeginPoint(*iter);
	indexJct=_tissueContext->getRankOfEndPoint(*iter);
	bool channelTarget=false;
	if (nodeCategory=="Channels") channelTarget=isChannelTarget(key, nodeType);
	if ( branchOrder!=0 &&
	     ( nodeCategory=="CompartmentVariables"
	       || (nodeCategory=="EndPoints" && (*iter)->_parent && computeOrder==0 )
	       || (nodeCategory=="ForwardSolvePoints" && (*iter)->_parent && computeOrder==nodeComputeOrder)
	       || (channelTarget && index==_rank) ) ) {
	  if (nodeCategory=="CompartmentVariables") {
	    _indexBranchMap[nodeType][index][rval[index]]=(*iter);
	    std::vector<int> indices;
	    indices.push_back(index);
	    indices.push_back(rval[index]);
	    _branchIndexMap[nodeType][(*iter)]=indices;
	    rval[index]++;
	  }
	  else if (nodeCategory=="EndPoints") rval[index]++;
	  else if (nodeCategory=="ForwardSolvePoints") {
	    std::vector<int> indices;
	    indices.push_back(index);
	    indices.push_back(rval[index]);
	    _branchForwardSolvePointIndexMap[nodeType][*iter]=indices;
	    rval[index]++;
	  }
	  else if (channelTarget) {
	    std::list<Params::ChannelTarget> * targets=_tissueParams.getChannelTargets(key);
	    if (targets) {	
	      std::list<Params::ChannelTarget>::iterator iiter=targets->begin(), iend=targets->end();
	      for (; iiter!=iend; ++iiter) {
		if ( iiter->_type==nodeType ) {
		  rval[index]++;
		  std::vector<std::pair<int, int> > targetVector;
		  std::list<std::string>::iterator viter, vend=iiter->_target1.end();
		  assert(iiter->_target1.size()>0);
		  for (viter=iiter->_target1.begin(); viter!=vend; ++viter) {
		    std::vector<int>& branchIndices=findBranchIndices(*iter, *viter);
		    assert(branchIndices[0]==_rank);
		    targetVector.push_back(std::pair<int, int>(branchIndices[1], _compartmentVariableTypesMap[*viter]) );
		  }
		  _channelBranchIndices1[_channelTypeCounter].push_back(targetVector);

		  targetVector.clear();
		  vend=iiter->_target2.end();
		  assert(iiter->_target2.size()>0);
		  for (viter=iiter->_target2.begin(); viter!=vend; ++viter) {
		    std::vector<int>& branchIndices=findBranchIndices(*iter, *viter);
		    assert(branchIndices[0]==_rank);
		    targetVector.push_back(std::pair<int, int>(branchIndices[1], _compartmentVariableTypesMap[*viter]) );
		  }
		  _channelBranchIndices2[_channelTypeCounter].push_back(targetVector);
		}
	      }
	    }
	  }
	}

	if ( nodeCategory!="CompartmentVariables" && nodeCategory!="ForwardSolvePoints") {
	  if ((*iter)->_daughters.size()>0) {
	    if (computeOrder==MAX_COMPUTE_ORDER) {
	      assert(_segmentDescriptor.getFlag((*iter)->lastCapsule().getKey()));
	      if (nodeCategory=="EndPoints" && branchOrder!=0 ) rval[index]++;
	      else {
		if (nodeCategory=="Junctions") {
		  _indexJunctionMap[nodeType][indexJct][rval[indexJct]]=&((*iter)->lastCapsule());
		  std::vector<int> indices;
		  indices.push_back(indexJct);
		  indices.push_back(rval[indexJct]);
		  _junctionIndexMap[nodeType][&((*iter)->lastCapsule())]=indices;
		  rval[indexJct]++;
		}
		else if (nodeCategory=="JunctionPoints") rval[indexJct]++;
		else if (channelTarget && indexJct==_rank) {
		  std::list<Params::ChannelTarget> * targets=_tissueParams.getChannelTargets(key);
		  if (targets) {	
		    std::list<Params::ChannelTarget>::iterator iiter=targets->begin(), iend=targets->end();
		    for (; iiter!=iend; ++iiter) {
		      if ( iiter->_type==nodeType ) {
			rval[indexJct]++;
			std::vector<std::pair<int, int> > targetVector;
			std::list<std::string>::iterator viter, vend=iiter->_target1.end();
			assert(iiter->_target1.size()>0);
			for (viter=iiter->_target1.begin(); viter!=vend; ++viter) {
			  std::map<std::string, std::map<Capsule*,  std::vector<int> > >::iterator jmapiter1=_junctionIndexMap.find(*viter);
			  std::map<Capsule*, std::vector<int> >::iterator jmapiter2;
			  if ( jmapiter1!=_junctionIndexMap.end() &&
			       (jmapiter2=jmapiter1->second.find(&(*iter)->lastCapsule() ) )!=jmapiter1->second.end() ) {
			    std::vector<int>& junctionIndices=jmapiter2->second;
			    targetVector.push_back(std::pair<int, int>(junctionIndices[1], _compartmentVariableTypesMap[*viter]) );
			  }
			}
			_channelJunctionIndices1[_channelTypeCounter].push_back(targetVector);
			targetVector.clear();
			vend=iiter->_target2.end();
			assert(iiter->_target2.size()>0);
			for (viter=iiter->_target2.begin(); viter!=vend; ++viter) {
			  std::map<std::string, std::map<Capsule*,  std::vector<int> > >::iterator jmapiter1=_junctionIndexMap.find(*viter);
			  std::map<Capsule*, std::vector<int> >::iterator jmapiter2;
			  if ( jmapiter1!=_junctionIndexMap.end() &&
			       (jmapiter2=jmapiter1->second.find(&(*iter)->lastCapsule() ) )!=jmapiter1->second.end() ) {
			    std::vector<int>& junctionIndices=jmapiter2->second;
			    targetVector.push_back(std::pair<int, int>(junctionIndices[1], _compartmentVariableTypesMap[*viter]) );
			  }
			}
			_channelJunctionIndices2[_channelTypeCounter].push_back(targetVector);
		      }
		    }
		  }
		}
	      }
	    }
	    else if (nodeCategory=="BackwardSolvePoints" && computeOrder==nodeComputeOrder) {
	      std::vector<int> indices;
	      indices.push_back(index);
	      indices.push_back(rval[index]);
	      _branchBackwardSolvePointIndexMap[nodeType][*iter]=indices;
	      rval[index]++;
	    }
	  }
	  else if ( (nodeCategory=="Junctions" || nodeCategory=="JunctionPoints") && 
		    _segmentDescriptor.getFlag((*iter)->lastCapsule().getKey() ) ) {
	    if (nodeCategory=="Junctions") {
	      assert(indexJct!=_rank);
	      _indexJunctionMap[nodeType][indexJct][rval[indexJct]]=&((*iter)->lastCapsule());
	      std::vector<int> indices;
	      indices.push_back(indexJct);
	      indices.push_back(rval[indexJct]);
	      _junctionIndexMap[nodeType][&((*iter)->lastCapsule())]=indices;
	    }
	    rval[indexJct]++;
	  }
	}
      }
    }
  }
  
  if (nodeCategory=="Channels") ++_channelTypeCounter;
  if (nodeCategory=="ElectricalSynapses") ++_electricalSynapseTypeCounter;
  if (nodeCategory=="ChemicalSynapses") ++_chemicalSynapseTypeCounter;
  if (nodeCategory=="CompartmentVariables") ++_compartmentVariableTypeCounter;
  if (nodeCategory=="Junctions") ++_junctionTypeCounter;
  if (nodeCategory=="PreSynapticPoints") ++_preSynapticPointTypeCounter;
  if (nodeCategory=="EndPoints") ++_endPointTypeCounter;
  if (nodeCategory=="JunctionPoints") ++_junctionPointTypeCounter;  
  if (nodeCategory=="BackwardSolvePoints") ++_backwardSolvePointTypeCounter;
  if (nodeCategory=="ForwardSolvePoints") ++_forwardSolvePointTypeCounter;
  
  return rval;
}

void TissueFunctor::doNodeInit(LensContext* lc)
{
  assert(_params.get());
  std::auto_ptr<ParameterSet> initPset;
  std::vector<NodeDescriptor*>  nodes;
  std::vector<NodeDescriptor*>::iterator node, nodesEnd;
    
  NodeSet *nodeset = lc->layerContext->nodeset;
  std::vector<GridLayerDescriptor*> const & layers = nodeset->getLayers();
  assert(layers.size()==1);
  std::vector<GridLayerDescriptor*>::const_iterator gld = layers.begin();

  std::vector<std::string> nodekind;
  getNodekind(&(*gld)->getNDPList(), nodekind);
  assert(nodekind.size()>0);
  std::string& nodeCategory = nodekind[0];
  std::string& nodeType = nodekind[1];

  if (nodeCategory=="CompartmentVariables") {
    _compartmentVariableLayers.push_back(*gld);
  }
  else if (nodeCategory=="Junctions") {
    _junctionLayers.push_back(*gld);
  }
  else if (nodeCategory=="EndPoints") {
    _endPointLayers.push_back(*gld);
  }
  else if (nodeCategory=="ForwardSolvePoints") {
    _forwardSolvePointLayers.push_back(*gld);
  }
  else if (nodeCategory=="BackwardSolvePoints") {
    _backwardSolvePointLayers.push_back(*gld);
  }
  else if (nodeCategory=="JunctionPoints") {
    _junctionPointLayers.push_back(*gld);
  }
  else if (nodeCategory=="Channels") {
    _channelLayers.push_back(*gld);
  }
  else if (nodeCategory=="ElectricalSynapses") {
    _electricalSynapseLayers.push_back(*gld);
  }
  else if (nodeCategory=="ChemicalSynapses") {
    _chemicalSynapseLayers.push_back(*gld);
  }
  else if (nodeCategory=="PreSynapticPoints") {
    _preSynapticPointLayers.push_back(*gld);
  }
  else {
    std::cerr<<"Unrecognized nodeCategory parameter on NodeInit : "<<nodeCategory<<std::endl;
    exit(0);
  }
  
  nodes.clear();
  nodeset->getNodes(nodes, *gld);
  node = nodes.begin();
  nodesEnd = nodes.end();
  if ( ( (nodeCategory=="Channels" || 
	  nodeCategory=="ElectricalSynapses" || 
	  nodeCategory=="ChemicalSynapses" || 
	  nodeCategory=="PreSynapticalPoints")
	 && (_junctionLayers.size()==0 || _compartmentVariableLayers.size()==0) ) ||
       (_junctionLayers.size()>0 && _compartmentVariableLayers.size()==0) || 
       (_preSynapticPointLayers.size()>0 && _chemicalSynapseLayers.size()==0) ) {
    std::cerr<<"TissueFunctor:"
	     <<std::endl
	     <<"Layers (Branches, Junctions, Channels | Synapses, PreSynapticPoints) must be initialized in order."
	     <<std::endl;
    exit(0);
  }
  NDPairList emptyOutAttr;
  NDPairList dim2cpt;
  dim2cpt.push_back(new NDPair("identifier", "dimension"));
  NDPairList brd2cpt;
  brd2cpt.push_back(new NDPair("identifier", "branchData"));

  for (; node != nodesEnd; ++node) {
    if ((*node)->getNode()) {
      NDPairList paramsLocal=*(_params.get());
      (*gld)->getNodeType()->getInitializationParameterSet(initPset);
      ParameterSet* pset=initPset.get();
      int nodeIndex=(*node)->getNodeIndex();
      int densityIndex=(*node)->getDensityIndex();
      if (nodeCategory=="CompartmentVariables" || nodeCategory=="Junctions") {
	int size=compartmentalize(lc, &paramsLocal, nodeCategory, nodeType, 
			 nodeIndex, densityIndex);
	StructType* st=lc->sim->getStructType("BranchDataStruct");
	ConstantType* ct=lc->sim->getConstantType("BranchData");
	NDPairList branchDataStructParams;
	IntDataItem* sizeDI=new IntDataItem(size);
	std::auto_ptr<DataItem> sizeDI_ap(sizeDI);
	NDPair* ndp=new NDPair("size", sizeDI_ap);
	branchDataStructParams.push_back(ndp);
	if (nodeCategory=="CompartmentVariables") {
	  ComputeBranch* branch=findBranch(nodeIndex, densityIndex, nodeType);
	  CG_BranchData*& branchData = _tissueContext->_branchBranchDataMap[branch];
	  double key=branch->_capsules[0].getKey();
 	  getModelParams(Params::COMPARTMENT, paramsLocal, nodeType, key);
	  pset->set(paramsLocal);
	  (*node)->getNode()->initialize(pset);
	  DoubleDataItem* keyDI=new DoubleDataItem(branch->_capsules[0].getKey());
	  std::auto_ptr<DataItem> keyDI_ap(keyDI);
	  ndp=new NDPair("key", keyDI_ap);
	  branchDataStructParams.push_back(ndp);
	  std::auto_ptr<DataItem> aptr_st;
	  st->getInstance(aptr_st, branchDataStructParams, lc);
	  ndp=new NDPair("branchData", aptr_st);
	  NDPairList branchDataParams;
	  branchDataParams.push_back(ndp);
	  std::auto_ptr<DataItem> aptr_cst;
	  ct->getInstance(aptr_cst, branchDataParams, lc);
	  ConstantDataItem* cdi=dynamic_cast<ConstantDataItem*>(aptr_cst.get());
	  std::auto_ptr<Constant> aptr_brd;
	  cdi->getConstant()->duplicate(aptr_brd);
	  Constant* brd=aptr_brd.release();
	  branchData=(dynamic_cast<CG_BranchData*>(brd));
	  assert(branchData);
	  _lensConnector.constantToNode(branchData, *node, &emptyOutAttr, &brd2cpt);

	  std::map<ComputeBranch*, std::vector<CG_CompartmentDimension*> >::iterator miter=_tissueContext->_branchDimensionsMap.find(branch);
	  assert(miter!=_tissueContext->_branchDimensionsMap.end());
	  std::vector<CG_CompartmentDimension*>& dimensions = miter->second;
	  std::vector<CG_CompartmentDimension*>::iterator diter, dend=dimensions.end();
	  for (diter=dimensions.begin(); diter!=dend; ++diter) _lensConnector.constantToNode(*diter, *node, &emptyOutAttr, &dim2cpt);
	}
	else {
	  Capsule* junctionCapsule=findJunction(nodeIndex, densityIndex, nodeType);	  
	  CG_BranchData*& branchData = _tissueContext->_junctionBranchDataMap[junctionCapsule];
	  double key=junctionCapsule->getKey();
 	  getModelParams(Params::COMPARTMENT, paramsLocal, nodeType, key);
	  pset->set(paramsLocal);
	  (*node)->getNode()->initialize(pset);
	  DoubleDataItem* keyDI=new DoubleDataItem(key);
	  std::auto_ptr<DataItem> keyDI_ap(keyDI);
	  NDPair* ndp=new NDPair("key", keyDI_ap);
	  branchDataStructParams.push_back(ndp);
	  std::auto_ptr<DataItem> aptr_st;
	  st->getInstance(aptr_st, branchDataStructParams, lc);
	  ndp=new NDPair("branchData", aptr_st);
	  NDPairList branchDataParams;
	  branchDataParams.push_back(ndp);
	  std::auto_ptr<DataItem> aptr_cst;
	  ct->getInstance(aptr_cst, branchDataParams, lc);
	  ConstantDataItem* cdi=dynamic_cast<ConstantDataItem*>(aptr_cst.get());
	  std::auto_ptr<Constant> aptr_brd;
	  cdi->getConstant()->duplicate(aptr_brd);
	  Constant* brd=aptr_brd.release();
	  branchData=(dynamic_cast<CG_BranchData*>(brd));	  
	  assert(branchData);
	  _lensConnector.constantToNode(branchData, *node, &emptyOutAttr, &brd2cpt);

	  std::map<Capsule*, CG_CompartmentDimension*>::iterator miter=_tissueContext->_junctionDimensionMap.find(junctionCapsule);
	  assert(miter!=_tissueContext->_junctionDimensionMap.end());
	  _lensConnector.constantToNode(miter->second, *node, &emptyOutAttr, &dim2cpt);
	}
      }
      else if (nodeCategory=="Channels") {
	assert(nodeIndex==_rank);
	std::string channelCategory;
	assert(_channelLayers.size()>0);
	int nChannelBranches=_channelBranchIndices1[_channelLayers.size()-1].size();
	double key;
	assert(_channelBranchIndices1[_channelLayers.size()-1].size()==_channelBranchIndices2[_channelLayers.size()-1].size());
	if (densityIndex<nChannelBranches) {
	  channelCategory="BranchChannels";
	  std::pair<int, int>& channelBranchIndexPair=_channelBranchIndices1[_channelLayers.size()-1][densityIndex][0];
	  key=findBranch(nodeIndex, channelBranchIndexPair.first, _compartmentVariableTypes[channelBranchIndexPair.second])->_capsules[0].getKey();
	}
	else {
	  channelCategory="JunctionChannels";
	  std::pair<int, int>& channelJunctionIndexPair=_channelJunctionIndices1[_channelLayers.size()-1][densityIndex-nChannelBranches][0];
	  key=findJunction(nodeIndex, channelJunctionIndexPair.first, _compartmentVariableTypes[channelJunctionIndexPair.second])->getKey();
	}

	std::list<std::pair<std::string, float> > channelParams;
	getModelParams(Params::CHANNEL, paramsLocal, nodeType, key);
	compartmentalize(lc, &paramsLocal, channelCategory, nodeType, nodeIndex, densityIndex);	

	pset->set(paramsLocal);
	(*node)->getNode()->initialize(pset);
      }
      else {
	pset->set(paramsLocal);
	(*node)->getNode()->initialize(pset);
      }
    }
  }
}

void TissueFunctor::doConnector(LensContext* lc)
{
  assert(_compartmentVariableLayers.size()==_compartmentVariableTypesMap.size());
  assert(_junctionLayers.size()==_junctionTypesMap.size());
  assert(_compartmentVariableTypesMap.size()==_junctionTypesMap.size());
  std::map<int, int> cptVarJctTypeMap;
  std::map<std::string, int>::iterator mapIter, mapEnd=_compartmentVariableTypesMap.end();
  for (mapIter=_compartmentVariableTypesMap.begin(); mapIter!=mapEnd; ++mapIter) {
    assert(_junctionTypesMap.find(mapIter->first)!=_junctionTypesMap.end());
    cptVarJctTypeMap[mapIter->second]=_junctionTypesMap[mapIter->first];
  }
  
  assert(_forwardSolvePointLayers.size()==_forwardSolvePointTypeCounter);
  assert(_backwardSolvePointLayers.size()==_backwardSolvePointTypeCounter);
  
  std::map<std::string, NDPairList> cpt2chan; 
  std::map<std::string, NDPairList> cpt2syn;
  std::map<std::string, NDPairList> chan2cpt;
  std::map<std::string, NDPairList> esyn2cpt;
  std::map<std::string, NDPairList> csyn2cpt;
  std::map<std::string, NDPairList> ic2syn;
  std::map<std::string, NDPairList> ic2chan;
  std::map<std::string, NDPairList> cnnxn2cnnxn;

  std::map<std::string, int>::iterator cptVarTypesIter, cptVarTypesEnd=_compartmentVariableTypesMap.end();
  for (cptVarTypesIter=_compartmentVariableTypesMap.begin(); cptVarTypesIter!=cptVarTypesEnd; ++cptVarTypesIter) {
    std::ostringstream os;

    NDPairList Mcpt2chan;
    os<<"compartment["<<cptVarTypesIter->first<<"]";
    Mcpt2chan.push_back(new NDPair("identifier", os.str()));
    cpt2chan[cptVarTypesIter->first]=Mcpt2chan;

    NDPairList Mcpt2syn;
    Mcpt2syn.push_back(new NDPair("identifier", os.str()));
    Mcpt2syn.push_back(new NDPair("idx", 0));
    cpt2syn[cptVarTypesIter->first]=Mcpt2syn;
    
    os.str("");;
    NDPairList Mchan2cpt;
    os<<"channels["<<cptVarTypesIter->first<<"]";
    Mchan2cpt.push_back(new NDPair("identifier", os.str()));
    chan2cpt[cptVarTypesIter->first]=Mchan2cpt;
    
    os.str("");
    NDPairList Mesyn2cpt;
    os<<"electricalSynapse["<<cptVarTypesIter->first<<"]";
    Mesyn2cpt.push_back(new NDPair("identifier", os.str()));
    Mesyn2cpt.push_back(new NDPair("idx", 0));
    esyn2cpt[cptVarTypesIter->first]=Mesyn2cpt;
    
    os.str("");
    NDPairList Mcsyn2cpt;
    os<<"chemicalSynapse["<<cptVarTypesIter->first<<"]";
    Mcsyn2cpt.push_back(new NDPair("identifier", os.str()));
    Mcsyn2cpt.push_back(new NDPair("idx", 0));
    csyn2cpt[cptVarTypesIter->first]=Mcsyn2cpt;
    
    os.str("");
    NDPairList Mic2syn;
    os<<"IC["<<cptVarTypesIter->first<<"]";
    Mic2syn.push_back(new NDPair("identifier", os.str()));
    Mic2syn.push_back(new NDPair("idx", 0));
    ic2syn[cptVarTypesIter->first]=Mic2syn;

    os.str("");
    NDPairList Mic2chan;
    os<<"IC["<<cptVarTypesIter->first<<"]";
    Mic2chan.push_back(new NDPair("identifier", os.str()));
    ic2chan[cptVarTypesIter->first]=Mic2chan;

    os.str("");
    NDPairList Mcnnxn2cnnxn;
    os<<"connexon["<<cptVarTypesIter->first<<"]";
    Mcnnxn2cnnxn.push_back(new NDPair("identifier", os.str()));
    cnnxn2cnnxn[cptVarTypesIter->first]=Mcnnxn2cnnxn;
  }

  NDPairList end2jct;
  end2jct.push_back(new NDPair("identifier", "endpoint"));
  
  NDPairList jctpt2dist;
  jctpt2dist.push_back(new NDPair("identifier", "proximalJunctionPoint"));
  
  NDPairList fwdpt2br;
  fwdpt2br.push_back(new NDPair("identifier", "forwardSolvePoint"));
  
  NDPairList jctpt2prox;
  jctpt2prox.push_back(new NDPair("identifier", "distalJunctionPoint"));
  
  NDPairList bwdpt2br;
  bwdpt2br.push_back(new NDPair("identifier", "backwardSolvePoint"));
  
  NDPairList prox2end;
  prox2end.push_back(new NDPair("identifier", "proximalEnd"));
  
  NDPairList dist2end;
  dist2end.push_back(new NDPair("identifier", "distalEnd"));
  
  NDPairList jct2jctpt;
  jct2jctpt.push_back(new NDPair("identifier", "junction"));

  NDPairList presynpt;
  presynpt.push_back(new NDPair("identifier", "preSynapticPoint"));
  
  NDPairList recp2recp;
  recp2recp.push_back(new NDPair("identifier", "receptor"));

  NDPairList br2fwdpt;
  NDPairList br2bwdpt;

  std::vector<NodeAccessor*> compartmentVariableAccessors;
  std::vector<GridLayerDescriptor*>::iterator layerIter, layerEnd=_compartmentVariableLayers.end();
  for (layerIter=_compartmentVariableLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    compartmentVariableAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  layerEnd=_junctionLayers.end();
  std::vector<NodeAccessor*> junctionAccessors;
  for (layerIter=_junctionLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    junctionAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  std::vector<NodeAccessor*> endPointAccessors;
  layerEnd=_endPointLayers.end();
  for (layerIter=_endPointLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    endPointAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  std::vector<NodeAccessor*> junctionPointAccessors;
  layerEnd=_junctionPointLayers.end();
  for (layerIter=_junctionPointLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    junctionPointAccessors.push_back((*layerIter)->getNodeAccessor());
  } 

  std::vector<NodeAccessor*> preSynapticPointAccessors;
  layerEnd=_preSynapticPointLayers.end();
  for (layerIter=_preSynapticPointLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    preSynapticPointAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  std::vector<NodeAccessor*> forwardSolvePointAccessors;
  layerEnd=_forwardSolvePointLayers.end();
  for (layerIter=_forwardSolvePointLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    forwardSolvePointAccessors.push_back((*layerIter)->getNodeAccessor());
  }
  std::vector<NodeAccessor*> backwardSolvePointAccessors;
  layerEnd=_backwardSolvePointLayers.end();
  for (layerIter=_backwardSolvePointLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    backwardSolvePointAccessors.push_back((*layerIter)->getNodeAccessor());
  }
  std::vector<NodeAccessor*> channelAccessors;
  layerEnd=_channelLayers.end();
  for (layerIter=_channelLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    channelAccessors.push_back((*layerIter)->getNodeAccessor());
  }
  std::vector<NodeAccessor*> electricalSynapseAccessors;
  layerEnd=_electricalSynapseLayers.end();
  for (layerIter=_electricalSynapseLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    electricalSynapseAccessors.push_back((*layerIter)->getNodeAccessor());
  }
  std::vector<NodeAccessor*> chemicalSynapseAccessors;
  layerEnd=_chemicalSynapseLayers.end();
  for (layerIter=_chemicalSynapseLayers.begin(); layerIter!=layerEnd; ++layerIter) {
    chemicalSynapseAccessors.push_back((*layerIter)->getNodeAccessor());
  }
  
  Simulation* sim=lc->sim;
  Connector* connector;
  
  if (sim->isGranuleMapperPass()) {
    connector=&_noConnector;
  } else if (sim->isCostAggregationPass()) {
    connector=&_granuleConnector;
  } else if (sim->isSimulatePass()) {
    connector=&_lensConnector;
  } else {
    std::cerr<<"Error, TissueFunctor : no connection context set!"<<std::endl;
    exit(0);
  }

  for (int i=0; i<_nbrGridNodes; ++i) {
    std::map<std::string, int>::iterator cptVarTypesIter, cptVarTypesEnd=_compartmentVariableTypesMap.end();
    for (cptVarTypesIter=_compartmentVariableTypesMap.begin(); cptVarTypesIter!=cptVarTypesEnd; ++cptVarTypesIter) {
      std::string cptVarType=cptVarTypesIter->first;
      int cptVarTypeIdx=cptVarTypesIter->second;
      int branchDensity=_compartmentVariableLayers[cptVarTypeIdx]->getDensity(i);       

      std::vector<int> endPointCounters;
      endPointCounters.resize(_endPointTypeCounter,0);
      for (int j=0; j<branchDensity; ++j) { // FIX
	ComputeBranch* br=findBranch(i, j, cptVarType);
	if (br) {
	  double key=br->_capsules[0].getKey();
	  int computeOrder=_segmentDescriptor.getComputeOrder(key);
	  assert(i==_tissueContext->getRankOfBeginPoint(br));
	  int endPointType=_endPointTypesMap[cptVarType];
	  int junctionPointType=_junctionPointTypesMap[cptVarType];

	  if (br->_daughters.size()>0) {
	    NodeDescriptor* compartmentVariable=compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(i, j);
	    assert(i==sim->getGranule(*compartmentVariable)->getPartitionId());
	    if (computeOrder==MAX_COMPUTE_ORDER) {
	      NodeDescriptor* endPoint=endPointAccessors[endPointType]->getNodeDescriptor(i, endPointCounters[endPointType]);
	      ++endPointCounters[endPointType];
	      connect(sim, connector, compartmentVariable, endPoint, dist2end);
	      Capsule* c = &br->lastCapsule();
	      std::vector<int>& junctionIndices=findJunctionIndices(c, cptVarType);
	      NodeDescriptor* junction=junctionAccessors[cptVarJctTypeMap[cptVarTypeIdx]]->
		getNodeDescriptor(junctionIndices[0], junctionIndices[1]);	
	      connect(sim, connector, endPoint, junction, end2jct);
	      NodeDescriptor* junctionPoint=junctionPointAccessors[junctionPointType]->
		getNodeDescriptor(junctionIndices[0], junctionIndices[1]);
	      connect(sim, connector, junction, junctionPoint, jct2jctpt);
	      connect(sim, connector, junctionPoint, compartmentVariable, jctpt2prox);
	    }
	    else {
	      std::vector<int>& backwardSolvePointIndices=findBackwardSolvePointIndices(br, cptVarType);
	      assert(i==backwardSolvePointIndices[0]);
	      NodeDescriptor* backwardSolvePoint=
		backwardSolvePointAccessors[_backwardSolvePointTypesMap[computeOrder][cptVarType]]->
		getNodeDescriptor(i, backwardSolvePointIndices[1]);	
	      connect(sim, connector, compartmentVariable, backwardSolvePoint, br2bwdpt);
	    }
	  }

	  if (br->_parent) {
	    NodeDescriptor* compartmentVariable=compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(i, j);
	    assert(i==sim->getGranule(*compartmentVariable)->getPartitionId());
	    if (computeOrder==0) {
	      NodeDescriptor* endPoint=endPointAccessors[endPointType]->getNodeDescriptor(i, endPointCounters[endPointType]);
	      ++endPointCounters[endPointType];
	      connect(sim, connector, compartmentVariable, endPoint, prox2end);
	      Capsule* c = &br->_parent->lastCapsule();
	      std::vector<int>& junctionIndices=findJunctionIndices(c, cptVarType);
	      NodeDescriptor* junction=junctionAccessors[cptVarJctTypeMap[cptVarTypeIdx]]->
		getNodeDescriptor(junctionIndices[0], junctionIndices[1]);
	      connect(sim, connector, endPoint, junction, end2jct);
	      NodeDescriptor* junctionPoint=junctionPointAccessors[junctionPointType]->
		getNodeDescriptor(junctionIndices[0], junctionIndices[1]);
	      connect(sim, connector, junction, junctionPoint, jct2jctpt);
	      connect(sim, connector, junctionPoint, compartmentVariable, jctpt2dist);
	    }
	    else {
	      std::vector<int>& forwardSolvePointIndices=findForwardSolvePointIndices(br, cptVarType);
	      assert(i==forwardSolvePointIndices[0]);
	      NodeDescriptor* forwardSolvePoint=
		forwardSolvePointAccessors[_forwardSolvePointTypesMap[computeOrder][cptVarType]]->
		getNodeDescriptor(i, forwardSolvePointIndices[1]);
	      connect(sim, connector, compartmentVariable, forwardSolvePoint, br2fwdpt);
	    }
	  }
	}
      }
    }
  }

  int i=_rank;
  assert(_channelTypeCounter==_channelLayers.size());
  for (int ctype=0; ctype<_channelTypeCounter; ++ctype) {
    int channelDensity=_channelLayers[ctype]->getDensity(i);
    std::vector<std::vector<std::pair<int, int> > >::iterator iter=_channelBranchIndices1[ctype].begin(), end=_channelBranchIndices1[ctype].end();
    for (int j=0; iter!=end; ++iter, ++j) {
      NodeDescriptor *channel=channelAccessors[ctype]->getNodeDescriptor(i,j);
      NodeDescriptor *compartmentVariable=0;
      std::vector<std::pair<int, int> >& channelBranchIndexPairs=(*iter);	
      std::vector<std::pair<int, int> >::iterator ctiter=channelBranchIndexPairs.begin(), ctend=channelBranchIndexPairs.end();
      for(; ctiter!=ctend; ++ctiter) {
	assert(ctiter->second<_compartmentVariableTypesMap.size());
	compartmentVariable=compartmentVariableAccessors[ctiter->second]->getNodeDescriptor(i, ctiter->first);
	assert(compartmentVariable);
	assert (sim->getGranule(*compartmentVariable)->getPartitionId()==_rank);
	connect(sim, connector, compartmentVariable, channel, ic2chan[_compartmentVariableTypes[ctiter->second]]);
	connect(sim, connector, compartmentVariable, channel, cpt2chan[_compartmentVariableTypes[ctiter->second]]);
      }
    }
    iter=_channelBranchIndices2[ctype].begin(), end=_channelBranchIndices2[ctype].end();
    for (int j=0; iter!=end; ++iter, ++j) {
      NodeDescriptor *channel=channelAccessors[ctype]->getNodeDescriptor(i,j);
      NodeDescriptor *compartmentVariable=0;
      std::vector<std::pair<int, int> >& channelBranchIndexPairs=(*iter);	
      std::vector<std::pair<int, int> >::iterator ctiter=channelBranchIndexPairs.begin(), ctend=channelBranchIndexPairs.end();
      for(; ctiter!=ctend; ++ctiter) {
	assert(ctiter->second<_compartmentVariableTypesMap.size());
	compartmentVariable=compartmentVariableAccessors[ctiter->second]->getNodeDescriptor(i, ctiter->first);
	assert(compartmentVariable);
	assert (sim->getGranule(*compartmentVariable)->getPartitionId()==_rank);
	connect(sim, connector, compartmentVariable, channel, ic2chan[_compartmentVariableTypes[ctiter->second]]);
	connect(sim, connector, channel, compartmentVariable, chan2cpt[_compartmentVariableTypes[ctiter->second]]);	    
      }
    }

    iter=_channelJunctionIndices1[ctype].begin(), end=_channelJunctionIndices1[ctype].end();
    for (int j=_channelBranchIndices1[ctype].size(); iter!=end; ++iter, ++j) {
      NodeDescriptor *channel=channelAccessors[ctype]->getNodeDescriptor(i,j);
      NodeDescriptor *compartmentVariable=0;
      std::vector<std::pair<int, int> >& channelJunctionIndexPairs=(*iter);	
      std::vector<std::pair<int, int> >::iterator ctiter=channelJunctionIndexPairs.begin(), ctend=channelJunctionIndexPairs.end();
      for(; ctiter!=ctend; ++ctiter) {
	assert(ctiter->second<_compartmentVariableTypesMap.size());
	compartmentVariable=junctionAccessors[ctiter->second]->getNodeDescriptor(i, ctiter->first);
	assert(compartmentVariable);
	assert (sim->getGranule(*compartmentVariable)->getPartitionId()==_rank);
	connect(sim, connector, compartmentVariable, channel, ic2chan[_compartmentVariableTypes[ctiter->second]]);
	connect(sim, connector, compartmentVariable, channel, cpt2chan[_compartmentVariableTypes[ctiter->second]]);
      }
    }
    iter=_channelJunctionIndices2[ctype].begin(), end=_channelJunctionIndices2[ctype].end();
    for (int j=_channelBranchIndices2[ctype].size(); iter!=end; ++iter, ++j) {
      NodeDescriptor *channel=channelAccessors[ctype]->getNodeDescriptor(i,j);
      NodeDescriptor *compartmentVariable=0;
      std::vector<std::pair<int, int> >& channelJunctionIndexPairs=(*iter);	
      std::vector<std::pair<int, int> >::iterator ctiter=channelJunctionIndexPairs.begin(), ctend=channelJunctionIndexPairs.end();
      for(; ctiter!=ctend; ++ctiter) {
	assert(ctiter->second<_compartmentVariableTypesMap.size());
	compartmentVariable=junctionAccessors[ctiter->second]->getNodeDescriptor(i, ctiter->first);
	assert(compartmentVariable);
	assert (sim->getGranule(*compartmentVariable)->getPartitionId()==_rank);
	connect(sim, connector, compartmentVariable, channel, ic2chan[_compartmentVariableTypes[ctiter->second]]);
	connect(sim, connector, channel, compartmentVariable, chan2cpt[_compartmentVariableTypes[ctiter->second]]);	    
      }
    }
  }

  for (int i=0; i<_nbrGridNodes; ++i) {
    std::map<std::string, int>::iterator cptVarTypesIter, cptVarTypesEnd=_compartmentVariableTypesMap.end();
    for (cptVarTypesIter=_compartmentVariableTypesMap.begin(); cptVarTypesIter!=cptVarTypesEnd; ++cptVarTypesIter) {
      std::string cptVarType=cptVarTypesIter->first;
      int cptVarTypeIdx=cptVarTypesIter->second;
      int branchDensity=_compartmentVariableLayers[cptVarTypeIdx]->getDensity(i);
      for (int j=0; j<branchDensity; ++j) {  // FIX
	ComputeBranch* br=findBranch(i, j, cptVarType);
	double key=br->_capsules[0].getKey();
	int computeOrder=_segmentDescriptor.getComputeOrder(key);

	if (br->_daughters.size()>0 && computeOrder!=MAX_COMPUTE_ORDER) {
	  NodeDescriptor* compartmentVariable=compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(i, j);
	  std::list<ComputeBranch*>::iterator diter, dend=br->_daughters.end();
	  for (diter=br->_daughters.begin(); diter!=dend; ++diter) {
	    assert(_tissueContext->getPass((*diter)->_capsules->getKey())==TissueContext::FIRST_PASS);
	    std::vector<int>& forwardSolvePointIndices=findForwardSolvePointIndices(*diter, cptVarType);
	    // This is hard! If you want to change it, you better be sure...
	    NodeDescriptor* forwardSolvePoint=
	      forwardSolvePointAccessors[_forwardSolvePointTypesMap[computeOrder+1][cptVarType]]->
	      getNodeDescriptor(forwardSolvePointIndices[0], forwardSolvePointIndices[1]);
	    connect(sim, connector, forwardSolvePoint, compartmentVariable, fwdpt2br);
	  }
	}
	if (br->_parent && computeOrder!=0) {
	  assert(_tissueContext->getPass(br->_parent->_capsules->getKey())==TissueContext::FIRST_PASS);
	  NodeDescriptor* compartmentVariable=compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(i, j);
	  assert(computeOrder==_segmentDescriptor.getComputeOrder(br->_parent->_capsules[0].getKey())+1);
	  std::vector<int>& backwardSolvePointIndices=findBackwardSolvePointIndices(br->_parent, cptVarType);
	  // This is hard! If you want to change it, you better be sure...
	  NodeDescriptor* backwardSolvePoint=
	    backwardSolvePointAccessors[_backwardSolvePointTypesMap[computeOrder-1][cptVarType]]->
	    getNodeDescriptor(backwardSolvePointIndices[0], backwardSolvePointIndices[1]);
	  connect(sim, connector, backwardSolvePoint, compartmentVariable, bwdpt2br);
	}
      }
    }
  }

  std::vector<std::map<int, int> > electricalSynapseCounters, chemicalSynapseCounters;
  electricalSynapseCounters.resize(_electricalSynapseTypeCounter);
  chemicalSynapseCounters.resize(_chemicalSynapseTypeCounter );
  

  for (int direction=0; direction<=1; ++direction) {

    TouchVector::TouchIterator titer=_tissueContext->_touchVector.begin(), 
      tend=_tissueContext->_touchVector.end();

    for (; titer!=tend; ++titer) {
      if (!_tissueContext->isLensTouch(*titer, _rank)) continue;
      double key1, key2;
      if (direction==0) {
	key1=titer->getKey1();
	key2=titer->getKey2();
      }
      else {
	key1=titer->getKey2();
	key2=titer->getKey1();
      }

      Capsule* preCapsule=&_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key1)];
      Capsule* postCapsule=&_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key2)];
      bool preJunction, postJunction;
      unsigned int indexPre, indexPost;

      if (_segmentDescriptor.getFlag(key1) && 
	  _tissueContext->isTouchToEnd(*preCapsule, *titer) ) {
	preJunction=true;
	indexPre=_tissueContext->getRankOfEndPoint(preCapsule->getBranch());
      }
      else {
	preJunction=false;
	indexPre=_tissueContext->getRankOfBeginPoint(preCapsule->getBranch());
      }
      if (_segmentDescriptor.getFlag(key2) &&
	  _tissueContext->isTouchToEnd(*postCapsule, *titer) ) {	  
	postJunction=true;
	indexPost=_tissueContext->getRankOfEndPoint(postCapsule->getBranch());
      }
      else {
	postJunction=false;
	indexPost=_tissueContext->getRankOfBeginPoint(postCapsule->getBranch());
      }


      std::list<Params::ElectricalSynapseTarget>* esynTargets = _tissueParams.getElectricalSynapseTargets(key1, key2);
      if (esynTargets) {
	std::list<Params::ElectricalSynapseTarget>::iterator esiter, esend=esynTargets->end();
	std::vector<int> typeCounter;
	typeCounter.resize(_electricalSynapseTypesMap.size(),0);
	for (esiter=esynTargets->begin(); esiter!=esend; ++esiter) {
	  int synapseType=_electricalSynapseTypesMap[esiter->_type];
	  if (isGenerated(_generatedElectricalSynapses[direction], titer, synapseType, typeCounter[synapseType]) ) {
	    std::map<int, int>& ecounts=electricalSynapseCounters[synapseType];
	    int preDI=getCountAndIncrement(ecounts, indexPre);
	    int postDI=getCountAndIncrement(ecounts, indexPost);
	    std::list<std::string>::iterator etiter=esiter->_target.begin(), etend=esiter->_target.end();
	    for (; etiter!=etend; ++etiter) {
	      NodeDescriptor *preCpt=0;
	      int preIdx=0;
	      if (preJunction) {
		std::map<std::string, std::map<Capsule*, std::vector<int> > >::iterator jmapiter1=_junctionIndexMap.find(*etiter);
		assert (jmapiter1!=_junctionIndexMap.end());
		std::map<Capsule*, std::vector<int> >::iterator jmapiter2=jmapiter1->second.find(preCapsule);
		assert(jmapiter2!=jmapiter1->second.end());
		std::vector<int>& junctionIndices=jmapiter2->second;
		preCpt=junctionAccessors[cptVarJctTypeMap[_compartmentVariableTypesMap[*etiter]]]->getNodeDescriptor(junctionIndices[0], junctionIndices[1]);	
	      }
	      else {
		std::vector<int>& branchIndices=findBranchIndices(preCapsule->getBranch(), *etiter);
		preCpt = compartmentVariableAccessors[_compartmentVariableTypesMap[*etiter]]->getNodeDescriptor(branchIndices[0], branchIndices[1]);
		preIdx=N_COMPARTMENTS(preCapsule->getBranch()->_nCapsules)-((preCapsule-preCapsule->getBranch()->_capsules)/_compartmentSize)-1;
	      }

	      bool electrical, chemical, generated;
	      
	      NodeDescriptor *postCpt=0;
	      int postIdx=0;
	      if (postJunction) {
		std::map<std::string, std::map<Capsule*, std::vector<int> > >::iterator jmapiter1=_junctionIndexMap.find(*etiter);
		assert(jmapiter1!=_junctionIndexMap.end());
		std::map<Capsule*, std::vector<int> >::iterator jmapiter2=jmapiter1->second.find(postCapsule);
		assert(jmapiter2!=jmapiter1->second.end());
		std::vector<int>& junctionIndices=jmapiter2->second;
		postCpt=junctionAccessors[cptVarJctTypeMap[_compartmentVariableTypesMap[*etiter]]]->getNodeDescriptor(junctionIndices[0], junctionIndices[1]);
	      }
	      else {
		std::vector<int>& branchIndices=findBranchIndices(postCapsule->getBranch(), *etiter);
		postCpt=compartmentVariableAccessors[_compartmentVariableTypesMap[*etiter]]->getNodeDescriptor(branchIndices[0], branchIndices[1]);
		postIdx=N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules)-((postCapsule-postCapsule->getBranch()->_capsules)/_compartmentSize)-1;
	      }

	      NodeDescriptor* preConnexon=
		electricalSynapseAccessors[synapseType]->
		getNodeDescriptor(indexPre, preDI);

	      NodeDescriptor* postConnexon=
		electricalSynapseAccessors[synapseType]->
		getNodeDescriptor(indexPost, postDI);
	      
	      NDPairList Mcpt2syn=cpt2syn[*etiter];
	      NDPairList Mesyn2cpt=esyn2cpt[*etiter];
	      NDPairList Mcnnxn2cnnxn=cnnxn2cnnxn[*etiter];

	      Mesyn2cpt.replace("idx", preIdx);
	      connect(sim, connector, preConnexon, preCpt, Mesyn2cpt);
	      Mcpt2syn.replace("idx", preIdx);
	      connect(sim, connector, preCpt, preConnexon, Mcpt2syn);
	      Mesyn2cpt.replace("idx", postIdx);
	      connect(sim, connector, postConnexon, postCpt, Mesyn2cpt);
	      Mcpt2syn.replace("idx", postIdx);
	      connect(sim, connector, postCpt, postConnexon, Mcpt2syn);
	      connect(sim, connector, preConnexon, postConnexon, Mcnnxn2cnnxn);
	      connect(sim, connector, postConnexon, preConnexon, Mcnnxn2cnnxn);
	    }
	  }	
	  typeCounter[synapseType]++;
	}
      }

      std::vector<NodeDescriptor*> preSynPoints;
      preSynPoints.resize(_chemicalSynapseTypeCounter, 0);
      std::list<Params::ChemicalSynapseTarget>* csynTargets = _tissueParams.getChemicalSynapseTargets(key1, key2);
      if (csynTargets) {
	std::list<Params::ChemicalSynapseTarget>::iterator csiter, csend=csynTargets->end();
	std::vector<int> typeCounter;
	typeCounter.resize(_chemicalSynapseTypesMap.size(),0);
	for (csiter=csynTargets->begin(); csiter!=csend; ++csiter) {
	  std::map<std::string, std::pair<std::list<std::string>, std::list<std::string> > >::iterator 
	    targetsIter, targetsEnd=csiter->_targets.end();
	  std::vector<NodeDescriptor*> mixedSynapse;
	  for (targetsIter=csiter->_targets.begin(); targetsIter!=targetsEnd; ++targetsIter) {
	    std::map<std::string, int>::iterator miter=_chemicalSynapseTypesMap.find(targetsIter->first);
	    assert(miter!=_chemicalSynapseTypesMap.end());
	    int synapseType=miter->second;
	    if (isGenerated(_generatedChemicalSynapses[direction], titer, synapseType, typeCounter[synapseType]) ) {    
	      std::map<int, int>& ccounts=chemicalSynapseCounters[synapseType];
	      NodeDescriptor* receptor=chemicalSynapseAccessors[synapseType]->getNodeDescriptor(indexPost, getCountAndIncrement(ccounts, indexPost));
	      mixedSynapse.push_back(receptor);
	      // Pre
	      std::list<std::string>::iterator ctiter=targetsIter->second.first.begin(), ctend=targetsIter->second.first.end();
	      for (; ctiter!=ctend; ++ctiter) {
		NodeDescriptor *preCpt=0;			
		int preIdx=0;
		if (preJunction) {
		  std::map<std::string, std::map<Capsule*, std::vector<int> > >::iterator jmapiter1=_junctionIndexMap.find(*ctiter);
		  assert(jmapiter1!=_junctionIndexMap.end());
		  std::map<Capsule*, std::vector<int> >::iterator jmapiter2=jmapiter1->second.find(preCapsule);
		  assert(jmapiter2!=jmapiter1->second.end() );
		  std::vector<int>& junctionIndices=jmapiter2->second;
		  preCpt=junctionAccessors[cptVarJctTypeMap[_compartmentVariableTypesMap[*ctiter]]]->getNodeDescriptor(junctionIndices[0], junctionIndices[1]);	
		}
		else {
		  std::vector<int>& branchIndices=findBranchIndices(preCapsule->getBranch(), *ctiter);
		  preCpt=compartmentVariableAccessors[_compartmentVariableTypesMap[*ctiter]]->getNodeDescriptor(branchIndices[0], branchIndices[1]);
		  preIdx=N_COMPARTMENTS(preCapsule->getBranch()->_nCapsules)-((preCapsule-preCapsule->getBranch()->_capsules)/_compartmentSize)-1;
		}
		NodeAccessor* preSynapticPointAccessor=0;
		std::string preSynapticPointType=_tissueParams.getPreSynapticPointTarget(targetsIter->first);
		std::map<std::string, int>::iterator tmapiter=_preSynapticPointTypesMap.find(preSynapticPointType);
		assert(tmapiter!=_preSynapticPointTypesMap.end());
		unsigned int preSynPointType=(tmapiter->second);
		if (preSynPoints[preSynPointType]==0) {
		  assert(preSynapticPointAccessors.size()>preSynPointType);
		  preSynapticPointAccessor=preSynapticPointAccessors[preSynPointType];
		  if (preJunction) {
		    preSynPoints[preSynPointType]=preSynapticPointAccessor->
		      getNodeDescriptor(indexPre, _capsuleJctPointIndexMap[preSynapticPointType][preCapsule]);
		  }
		  else {
		    preSynPoints[preSynPointType]=preSynapticPointAccessor->
		      getNodeDescriptor(indexPre, _capsuleCptPointIndexMap[preSynapticPointType][preCapsule]);
		  }
		}
		NDPairList Mcpt2syn=cpt2syn[*ctiter];
		Mcpt2syn.replace("idx", preIdx);
		connect(sim, connector, preCpt, preSynPoints[preSynPointType], Mcpt2syn);
		connect(sim, connector, preSynPoints[preSynPointType], receptor, presynpt);
	      }
	      
	      // Post
	      ctiter=targetsIter->second.second.begin(), ctend=targetsIter->second.second.end();
	      for (; ctiter!=ctend; ++ctiter) {
		NodeDescriptor *postCpt=0;
		int postIdx=0;
		if (postJunction) {
		  std::map<std::string, std::map<Capsule*, std::vector<int> > >::iterator jmapiter1=_junctionIndexMap.find(*ctiter);
		  assert(jmapiter1!=_junctionIndexMap.end());
		  std::map<Capsule*, std::vector<int> >::iterator jmapiter2=jmapiter1->second.find(postCapsule);
		  assert(jmapiter2!=jmapiter1->second.end());
		  std::vector<int>& junctionIndices=jmapiter2->second;
		  postCpt=junctionAccessors[cptVarJctTypeMap[_compartmentVariableTypesMap[*ctiter]]]->getNodeDescriptor(junctionIndices[0], junctionIndices[1]);
		}
		else {
		  std::vector<int>& branchIndices=findBranchIndices(postCapsule->getBranch(), *ctiter);
		  postCpt=compartmentVariableAccessors[_compartmentVariableTypesMap[*ctiter]]->getNodeDescriptor(branchIndices[0], branchIndices[1]);
		  postIdx=N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules)-((postCapsule-postCapsule->getBranch()->_capsules)/_compartmentSize)-1;
		}
		NDPairList Mcsyn2cpt=csyn2cpt[*ctiter];
		Mcsyn2cpt.replace("idx", postIdx);
		connect(sim, connector, receptor, postCpt, Mcsyn2cpt);
		
		NDPairList Mcpt2syn=cpt2syn[*ctiter];
		Mcpt2syn.replace("idx", postIdx);
		NDPairList Mic2syn=ic2syn[*ctiter];
		Mic2syn.replace("idx", postIdx);
		connect(sim, connector, postCpt, receptor, Mcpt2syn);
		connect(sim, connector, postCpt, receptor, Mic2syn);
	      }
	    }
	    typeCounter[synapseType]++;
	  }
	  for (int i=0; i<mixedSynapse.size(); ++i) {
	    for (int j=0; j<mixedSynapse.size(); ++j) {
	      if (i!=j) connect(sim, connector, mixedSynapse[i], mixedSynapse[j], recp2recp);
	    }
	  }
	}
      }
    }
  }
  
  if (sim->isSimulatePass()) {
    CountableModel* countableModel=0;
    
    std::vector<GridLayerDescriptor*>::iterator layerIter, layerEnd;
    std::list<CountableModel*> models;
    
    layerEnd=_compartmentVariableLayers.end();
    for (layerIter=_compartmentVariableLayers.begin(); layerIter!=layerEnd; ++layerIter) {
      countableModel=dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel) {
	if(find(models.begin(), models.end(), countableModel)==models.end()) {
	  countableModel->count();     
	  models.push_back(countableModel);
	}
      }
    }
    models.clear();

    layerEnd=_junctionLayers.end();
    for (layerIter=_junctionLayers.begin(); layerIter!=layerEnd; ++layerIter) {
      countableModel=dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel) {
	if(find(models.begin(), models.end(), countableModel)==models.end()) {
	  countableModel->count();     
	  models.push_back(countableModel);
	}
      }
    }
    models.clear();
    
    layerEnd=_channelLayers.end();
    for (layerIter=_channelLayers.begin(); layerIter!=layerEnd; ++layerIter) {
      countableModel=dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel) {
	if(find(models.begin(), models.end(), countableModel)==models.end()) {
	  countableModel->count();     
	  models.push_back(countableModel);
	}
      }
    }
    models.clear();

    layerEnd=_electricalSynapseLayers.end();
    for (layerIter=_electricalSynapseLayers.begin(); layerIter!=layerEnd; ++layerIter) {
      countableModel=dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel) {
	if(find(models.begin(), models.end(), countableModel)==models.end()) {
	  countableModel->count();     
	  models.push_back(countableModel);
	}
      }
    }
    models.clear();

    layerEnd=_chemicalSynapseLayers.end();
    for (layerIter=_chemicalSynapseLayers.begin(); layerIter!=layerEnd; ++layerIter) {
      countableModel=dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel) {
	if(find(models.begin(), models.end(), countableModel)==models.end()) {
	  countableModel->count();     
	  models.push_back(countableModel);
	}
      }
    }
  }
}

void TissueFunctor::doProbe(LensContext* lc, std::auto_ptr<NodeSet>& rval)
{
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  NDPairList::iterator ndpiter=_params->end(), ndpend_reverse=_params->begin();
  --ndpiter;
  --ndpend_reverse;
  
  if ((*ndpiter)->getName()!="CATEGORY") {
    std::cerr<<"First parameter of TissueProbe must be CATEGORY!"<<std::endl;
    exit(0);
  }
  StringDataItem* categoryDI = dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
  if (categoryDI==0) {
    std::cerr<<"CATEGORY parameter of TissueProbe must be a string!"<<std::endl;
    exit(0);
  }  
  std::string category=categoryDI->getString();
  if (category!="BRANCH" && category!="JUNCTION" && category!="CHANNEL" && category!="SYNAPSE") {
    std::cerr<<"Unrecognized CATEGORY during TissueProbe : "<<category<<" !"<<std::endl;
    exit(0);
  }
  
  --ndpiter;
  
  if ((*ndpiter)->getName()!="TYPE") {
    std::cerr<<"Second parameter of TissueProbe must be TYPE!"<<std::endl;
    exit(0);
  }
  StringDataItem* typeDI = dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
  if (typeDI==0) {
    std::cerr<<"TYPE parameter of TissueProbe must be a string!"<<std::endl;
    exit(0);
  }  
  std::string type=typeDI->getString();
  
  int typeIdx;
  std::map<std::string, int>::iterator typeIter;
  if (category=="BRANCH") {
    typeIter=_compartmentVariableTypesMap.find(type);
    if (typeIter==_compartmentVariableTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  if (category=="JUNCTION") {
    typeIter=_junctionTypesMap.find(type);
    if (typeIter==_junctionTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  if (category=="CHANNEL") {
    typeIter=_channelTypesMap.find(type);
    if (typeIter==_channelTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  if (category=="CHANNEL") {
    typeIter=_channelTypesMap.find(type);
    if (typeIter==_channelTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  bool esyn=false;
  if (category=="SYNAPSE") {
    typeIter=_chemicalSynapseTypesMap.find(type);
    if (typeIter==_chemicalSynapseTypesMap.end()) {
      typeIter=_electricalSynapseTypesMap.find(type);
      if (typeIter==_electricalSynapseTypesMap.end()) {
	std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
	esyn=true;
	exit(0);
      }
    }
    typeIdx=typeIter->second;
  }

  unsigned int* ids=new unsigned int[_params->size()-2];
  unsigned int idx=-1;

  --ndpiter;

  unsigned long long mask=0;
  double targetKey=0;

  if (category=="BRANCH" || category=="JUNCTION" || category=="CHANNEL") {
    for (; ndpiter!=ndpend_reverse; --ndpiter) {
      NumericDataItem* ndi=dynamic_cast<NumericDataItem*>((*ndpiter)->getDataItem());
      if (ndi==0) {
	std::cerr<<"TissueProbe parameter specification must comprise unsigned integers!"<<std::endl;
	exit(0);
      }
      maskVector.push_back(_segmentDescriptor.getSegmentKeyData((*ndpiter)->getName()));
      ids[++idx]=ndi->getUnsignedInt();
    }

    mask=_segmentDescriptor.getMask(maskVector);
    targetKey=_segmentDescriptor.getSegmentKey(maskVector, ids);
    delete ids;
  }
  std::vector<NodeDescriptor*> nodeDescriptors;
  GridLayerDescriptor* layer=0;
  std::map<ComputeBranch*, std::vector<int> >* indexMap;
  if (category=="BRANCH") {
    layer=_compartmentVariableLayers[typeIdx];
    assert(layer);
    std::map<ComputeBranch*, std::vector<int> >::iterator mapiter, mapend=_branchIndexMap[type].end();
    for (mapiter=_branchIndexMap[type].begin(); mapiter!=mapend; ++mapiter) {
      double key=mapiter->first->_capsules->getKey();
      if ( (mapiter->second)[0]==_rank && _segmentDescriptor.getSegmentKey(key, mask)==targetKey)
	nodeDescriptors.push_back(layer->getNodeAccessor()->
				  getNodeDescriptor((mapiter->second)[0], (mapiter->second)[1]));
    }
  }
  if (category=="JUNCTION") {
    layer=_junctionLayers[typeIdx];
    assert(layer);
    std::map<Capsule*, std::vector<int> >::iterator mapiter, mapend=_junctionIndexMap[type].end();
    for (mapiter=_junctionIndexMap[type].begin(); mapiter!=mapend; ++mapiter) {
      double key=mapiter->first->getKey();
      if ( (mapiter->second)[0]==_rank && _segmentDescriptor.getSegmentKey(key, mask)==targetKey)
	nodeDescriptors.push_back(layer->getNodeAccessor()->
				  getNodeDescriptor((mapiter->second)[0], (mapiter->second)[1]));
    }
  }
  if (category=="CHANNEL") {
    layer=_channelLayers[typeIdx];
    assert(layer);
    int density=layer->getDensity(_rank);
    int nChannelBranches=_channelBranchIndices1[typeIdx].size(); 
    double key;
    for (int i=0; i<density; ++i) {  // FIX
      if (i<nChannelBranches) {
	std::pair<int, int>& channelBranchIndexPair=_channelBranchIndices1[typeIdx][i][0];
	key=findBranch(_rank, channelBranchIndexPair.first, _compartmentVariableTypes[channelBranchIndexPair.second])->_capsules[0].getKey();
      }
      else {
	std::pair<int, int>& channelJunctionIndexPair=_channelJunctionIndices1[typeIdx][i-nChannelBranches][0];
	key=findJunction(_rank, channelJunctionIndexPair.first, _compartmentVariableTypes[channelJunctionIndexPair.second])->getKey();
      }
      if (_segmentDescriptor.getSegmentKey(key, mask)==targetKey)
	nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
    }
  }
  if (category=="SYNAPSE") {
    layer = esyn ? _electricalSynapseLayers[typeIdx] : _chemicalSynapseLayers[typeIdx];
    assert(layer);
    int density=layer->getDensity(_rank);
    for (int i=0; i<density; ++i) { // FIX
      nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
    }
  }
  
  NodeSet* ns=0;
  if (nodeDescriptors.size()>0) {
    ns=new NodeSet( (*nodeDescriptors.begin())->getGridLayerDescriptor()->getGrid(), nodeDescriptors);
  }
  else {
    ns=new NodeSet(layer->getGrid());
    ns->empty();
  }
  rval.reset(ns);
}

void TissueFunctor::doMGSify(LensContext* lc)
{
  std::vector<std::vector<GridLayerDescriptor*> > layers;
  layers.push_back(_compartmentVariableLayers);
  layers.push_back(_junctionLayers);
  layers.push_back(_endPointLayers);
  layers.push_back(_junctionPointLayers);
  layers.push_back(_channelLayers);
  layers.push_back(_electricalSynapseLayers);
  layers.push_back(_chemicalSynapseLayers);
  layers.push_back(_preSynapticPointLayers);
  layers.push_back(_forwardSolvePointLayers);
  layers.push_back(_backwardSolvePointLayers);
    
  std::vector<std::vector<GridLayerDescriptor*> >::iterator lliter, llend=layers.end();
  unsigned* mgsrval = new unsigned(_nbrGridNodes);
  for (lliter=layers.begin(); lliter!=llend; ++lliter) {
    std::vector<GridLayerDescriptor*>::iterator liter, lend=lliter->end();
    for (liter=lliter->begin(); liter!=lend; ++liter) {
      unsigned n = (*liter)->getDensity(_rank);
      MPI_Allgather(&n, 1, MPI_UNSIGNED, mgsrval, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
      (*liter)->replaceDensityVector(mgsrval, _nbrGridNodes);
    }
  }
  delete [] mgsrval;
}

void TissueFunctor::getModelParams(Params::ModelType modelType, NDPairList& paramsLocal, std::string& nodeType, double key)
{
  std::list<std::pair<std::string, float> > compartmentParams;
  _tissueParams.getModelParams(modelType, nodeType, key, compartmentParams);
  std::list<std::pair<std::string, float> >::iterator cpiter=compartmentParams.begin(), cpend=compartmentParams.end();
  for (; cpiter!=cpend; ++cpiter) {
    FloatDataItem* paramDI=new FloatDataItem(cpiter->second);
    std::auto_ptr<DataItem> paramDI_ap(paramDI);
    NDPair* ndp=new NDPair(cpiter->first, paramDI_ap);
    paramsLocal.push_back(ndp);
  }
  
  std::list<std::pair<std::string, std::vector<float> > > compartmentArrayParams;
  _tissueParams.getModelArrayParams(modelType, nodeType, key, compartmentArrayParams);
  std::list<std::pair<std::string, std::vector<float> > >::iterator capiter=compartmentArrayParams.begin(), capend=compartmentArrayParams.end();
  for (; capiter!=capend; ++capiter) {
    ShallowArray<float> farr;
    std::vector<float>::iterator viter=capiter->second.begin(), vend=capiter->second.end();
    for (; viter!=vend; ++viter) farr.push_back(*viter);
    FloatArrayDataItem* paramDI=new FloatArrayDataItem(farr);
    std::auto_ptr<DataItem> paramDI_ap(paramDI);
    if (!paramsLocal.replace(capiter->first, paramDI_ap) ) {
      NDPair* ndp=new NDPair(capiter->first, paramDI_ap);
      paramsLocal.push_back(ndp);
    }
  }
}

bool TissueFunctor::isChannelTarget(double key, std::string nodeType)
{
  bool rval=false;
  std::list<Params::ChannelTarget> * channelTypes=_tissueParams.getChannelTargets(key);
  if (channelTypes) {
    std::list<Params::ChannelTarget>::iterator iiter=channelTypes->begin(),
      iend=channelTypes->end();
    for (; iiter!=iend; ++iiter) {
      if (iiter->_type==nodeType) {
	rval=true;
	break;
      }
    }
  }
  return rval;
}

 void TissueFunctor::getElectricalSynapseProbabilities(std::vector<double>& probabilities, TouchVector::TouchIterator & titer, int direction, std::string nodeType)
{
  double key1 = (direction==0) ? titer->getKey1() : titer->getKey2();
  double key2 = (direction==0) ? titer->getKey2() : titer->getKey1();
  if (_tissueParams.electricalSynapses()  && (key1<key2 || !_tissueParams.symmetricElectricalSynapseTargets(key1, key2) ) ) {
    std::list<Params::ElectricalSynapseTarget> * synapseTypes=_tissueParams.getElectricalSynapseTargets(key1, key2);
    if (synapseTypes) {
      std::list<Params::ElectricalSynapseTarget>::iterator iiter=synapseTypes->begin(),
	iend=synapseTypes->end();
      for (; iiter!=iend; ++iiter) {
	if (iiter->_type==nodeType)
	  probabilities.push_back(iiter->_parameter);
      }
    }
  }
}

void TissueFunctor::getChemicalSynapseProbabilities(std::vector<double>& probabilities, TouchVector::TouchIterator & titer, int direction, std::string nodeType)
{
  double key1 = (direction==0) ? titer->getKey1() : titer->getKey2();
  double key2 = (direction==0) ? titer->getKey2() : titer->getKey1();
  if (_tissueParams.chemicalSynapses()) {
    std::list<Params::ChemicalSynapseTarget> * synapseTypes=_tissueParams.getChemicalSynapseTargets(key1, key2);
    if (synapseTypes) {
      std::vector<int> typeCounter;
      typeCounter.resize(_chemicalSynapseTypesMap.size(),0);
      std::list<Params::ChemicalSynapseTarget>::iterator iiter=synapseTypes->begin(),
	iend=synapseTypes->end();
      for (; iiter!=iend; ++iiter) {
	bool generated=false;
	bool nonGenerated=false;
	bool hit=false;
	bool mixedSynapse=(iiter->_targets.size()>1) ? true : false;
	std::map<std::string, std::pair<std::list<std::string>, std::list<std::string> > >::iterator 
	  targetsIter, targetsEnd=iiter->_targets.end();
	int d=0;
	for (targetsIter=iiter->_targets.begin(); targetsIter!=targetsEnd; ++targetsIter, ++d) {
	  std::map<std::string, int>::iterator miter=_chemicalSynapseTypesMap.find(targetsIter->first);
	  if (miter!=_chemicalSynapseTypesMap.end()) {
	    int type=miter->second;
	    assert(type<typeCounter.size());
	    if (mixedSynapse && !nonGenerated && !generated) nonGenerated=isNonGenerated(titer, direction, targetsIter->first, typeCounter[type]);
	    if (!generated && !nonGenerated) generated=isGenerated(_generatedChemicalSynapses[direction], titer, type, typeCounter[type]);
	    if (targetsIter->first==nodeType) hit=true;
	    typeCounter[type]++;
	  }
	}
	if (hit) {
	  if (generated) probabilities.push_back(1.0);
	  else if (nonGenerated) probabilities.push_back(0.0);
	  else probabilities.push_back(iiter->_parameter);
	}
      }
    }
  }
}

bool TissueFunctor::isPointRequired(TouchVector::TouchIterator & titer, int direction, std::string nodeType)
{
  bool rval=false;
  double key1 = (direction==0) ? titer->getKey1() : titer->getKey2();
  double key2 = (direction==0) ? titer->getKey2() : titer->getKey1();
  if (_tissueParams.chemicalSynapses()) {
    std::list<Params::ChemicalSynapseTarget> * synapseTypes=_tissueParams.getChemicalSynapseTargets(key1, key2);
    if (synapseTypes) {
      std::list<Params::ChemicalSynapseTarget>::iterator iiter=synapseTypes->begin(),
	iend=synapseTypes->end();
      for (; iiter!=iend && !rval; ++iiter) {
	std::map<std::string, std::pair<std::list<std::string>, std::list<std::string> > >::iterator 
	  targetsIter, targetsEnd=iiter->_targets.end();
	for (targetsIter=iiter->_targets.begin(); targetsIter!=targetsEnd && !rval; ++targetsIter) {
	  if (targetsIter->first==nodeType) rval=true;
	}
      }
    }
  }
  return rval;
}

void TissueFunctor::setGenerated(std::map<Touch*, std::list<std::pair<int, int> > >& smap, 
				 TouchVector::TouchIterator & titer, 
				 int type, int order)
{
  std::map<Touch*, std::list<std::pair<int, int> > >::iterator miter=smap.find(&(*titer));
  if (miter==smap.end()) {
    smap[&(*titer)]=std::list<std::pair<int, int> >();
    miter=smap.find(&(*titer));
  }
  miter->second.push_back(std::pair<int, int>(type, order) );
}

std::list<Params::ChemicalSynapseTarget>::iterator TissueFunctor::getChemicalSynapseTargetFromOrder(TouchVector::TouchIterator & titer, 
												    int direction, std::string type, int order)
{
  std::list<Params::ChemicalSynapseTarget>::iterator rval, rend;
  double key1 = (direction==0) ? titer->getKey1() : titer->getKey2();
  double key2 = (direction==0) ? titer->getKey2() : titer->getKey1();
  assert(_tissueParams.chemicalSynapses());
  std::list<Params::ChemicalSynapseTarget> * synapseTypes=_tissueParams.getChemicalSynapseTargets(key1, key2);
  assert(synapseTypes);
  rval=synapseTypes->begin();
  rend=synapseTypes->end();
  while (order>=0 && rval!=rend) {
    std::map<std::string, std::pair<std::list<std::string>, std::list<std::string> > >::iterator 
      targetsIter, targetsEnd=rval->_targets.end();
    int j=0;
    for (targetsIter=rval->_targets.begin(); targetsIter!=targetsEnd; ++targetsIter) {
      if (targetsIter->first==type) --order;
    }
    if (order>=0) ++rval;
  }
  assert(rval!=rend);
  return rval;
}

bool TissueFunctor::isGenerated(std::map<Touch*, std::list<std::pair<int, int> > >& smap, 
			        TouchVector::TouchIterator & titer,
				int type, int order)
{
  bool rval=false;
  std::map<Touch*, std::list<std::pair<int, int> > >::iterator miter=smap.find(&(*titer));
  if (miter!=smap.end()) {
    if (type<0) rval=true;
    else {
      std::list<std::pair<int, int> >& l=miter->second;
      std::list<std::pair<int, int> >::iterator liter, lend=l.end();
      for (liter=l.begin(); liter!=lend; ++liter) {
	if (liter->first==type && liter->second==order) {
	  rval=true;
	  break;
	}
      }
    }
  }
  return rval;
}

void TissueFunctor::setNonGenerated(TouchVector::TouchIterator & titer,
				    int direction, std::string type, int order)
{
  std::list<Params::ChemicalSynapseTarget>::iterator iiter=getChemicalSynapseTargetFromOrder(titer, direction, type, order);
  if (iiter->_targets.size()>1) {
    _nonGeneratedMixedChemicalSynapses[direction][&(*titer)].push_back(iiter);
  }
}

bool TissueFunctor::isNonGenerated(TouchVector::TouchIterator & titer, 
				   int direction, std::string nodeType, int order)
{
  bool rval=false;
  std::list<Params::ChemicalSynapseTarget>::iterator iiter=getChemicalSynapseTargetFromOrder(titer, direction, nodeType, order);
  if (iiter->_targets.size()>1) {
    std::map<Touch*, std::list<std::list<Params::ChemicalSynapseTarget>::iterator> >::iterator miter = _nonGeneratedMixedChemicalSynapses[direction].find(&(*titer));
    if (miter!=_nonGeneratedMixedChemicalSynapses[direction].end() ) {
      if (find(miter->second.begin(), miter->second.end(), iiter)!=miter->second.end()) rval=true;
    }
  }
  return rval;
}

RNG& TissueFunctor::findSynapseGenerator(int preRank, int postRank)
{
  if (preRank==postRank) return _tissueContext->_localSynapseGenerator;
  int rank1=MIN(preRank,postRank);
  int rank2=MAX(preRank,postRank);

  std::map<int, std::map<int, RNG> >::iterator miter = _synapseGeneratorMap.find(rank1);
  if (miter==_synapseGeneratorMap.end()) {
    RNG rng;
    resetSynapseGenerator(rng, rank1, rank2);
    std::map<int, RNG> smap;
    smap[rank2]=rng;
    _synapseGeneratorMap[rank1]=smap;
    return _synapseGeneratorMap[rank1][rank2];
  }
  else {
    std::map<int, RNG>::iterator miter2 = miter->second.find(rank2);
    if (miter2==miter->second.end()) {
      RNG rng;
      resetSynapseGenerator(rng, rank1, rank2);
      miter->second[rank2]=rng;
      return miter->second[rank2];
    }
    else {
      return miter2->second;
    }
  }
}

void TissueFunctor::resetSynapseGenerator(RNG& rng, int rank1, int rank2)
{
  rng.reSeedShared(_tissueContext->_boundarySynapseGeneratorSeed);
  for (int i=0; i<rank1; ++i) {
    long nextSeed=lrandom(rng);
    rng.reSeedShared(nextSeed);
  }
  rng.reSeedShared(lrandom(rng)+rank2);
}

TissueFunctor::~TissueFunctor() 
{
#ifdef HAVE_MPI
  --_instanceCounter;
  if (_instanceCounter==0) {
    std::map<ComputeBranch*, std::vector<CG_CompartmentDimension*> >::iterator miter, mend=_tissueContext->_branchDimensionsMap.end();
    for (miter=_tissueContext->_branchDimensionsMap.begin(); miter!=mend; ++miter) {
      std::vector<CG_CompartmentDimension*>::iterator viter, vend=miter->second.end();
      for (viter=miter->second.begin(); viter!=vend; ++viter) {
	delete *viter;
      }
    }
    std::map<Capsule*, CG_CompartmentDimension*>::iterator miter2, mend2=_tissueContext->_junctionDimensionMap.end();
    for (miter2=_tissueContext->_junctionDimensionMap.begin(); miter2!=mend2; ++miter2) {
      delete miter2->second;
    }
    std::map<ComputeBranch*, CG_BranchData*>::iterator miter3, mend3=_tissueContext->_branchBranchDataMap.end();
    for (miter3=_tissueContext->_branchBranchDataMap.begin(); miter3!=mend3; ++miter3) {
      delete miter3->second;
    }
    std::map<Capsule*, CG_BranchData*>::iterator miter4, mend4=_tissueContext->_junctionBranchDataMap.end();
    for (miter4=_tissueContext->_junctionBranchDataMap.begin(); miter4!=mend4; ++miter4) {
      delete miter4->second;
    }
    delete _tissueContext;
  }
#endif
}

void TissueFunctor::duplicate(std::auto_ptr<TissueFunctor>& dup) const
{
  dup.reset(new TissueFunctor(*this));
}

void TissueFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
  dup.reset(new TissueFunctor(*this));
}

void TissueFunctor::duplicate(std::auto_ptr<CG_TissueFunctorBase>& dup) const
{
  dup.reset(new TissueFunctor(*this));
}


int TissueFunctor::getCountAndIncrement(std::map<int, int>& cmap, int index)
{
  int rval=0;
  std::map<int, int>::iterator miter=cmap.find(index);
  if (miter==cmap.end()) cmap[index]=1;
  else {
    rval=miter->second;
    ++miter->second;
  }
  return rval;
}
