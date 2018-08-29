// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "StringUtils.h"
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
#include "IntDataItem.h"
#include "IntArrayDataItem.h"
#include "DoubleDataItem.h"
#include "DoubleArrayDataItem.h"
#include "StringDataItem.h"
#include "StringArrayDataItem.h"
#include <mpi.h>
#include <typeinfo>

#include "MaxComputeOrder.h"
#include "NTSMacros.h"
#include "Coordinates.h"
#ifdef HAVE_MPI

#include "SegmentForceAggregator.h"
#include "AllInSegmentSpace.h"
#include "TouchDetectTissueSlicer.h"
#include "FrontSegmentSpace.h"
#include "FrontLimitedSegmentSpace.h"
#include "SegmentKeySegmentSpace.h"
#include "ANDSegmentSpace.h"
#include "ORSegmentSpace.h"
#include "NOTSegmentSpace.h"
#include "NeuroDevTissueSlicer.h"
#include "SynapseTouchSpace.h"
#include "AllInTouchSpace.h"
#include "Director.h"
#include "SegmentForceDetector.h"
#include "Communicator.h"
#include "Params.h"
#include "TissueGrowthSimulator.hpp"

#include "LENSTissueSlicer.h"
#include "TouchDetector.h"
#include "ORTouchSpace.h"
#include "ComputeBranch.h"
#include "TouchVector.h"
#include "TouchAggregator.h"

#include "VolumeDecomposition.h"
#include "CountableModel.h"

#include "Neurogenesis.h"
#include "NeurogenParams.h"
#include "BoundingSurfaceMesh.h"
#include "CompositeSwc.h"
#include "Branch.h"
#include "Capsule.h"
#include "Touch.h"

#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// number of fields in tissue.txt file
#define PAR_FILE_INDEX 8
#define N_BRANCH_TYPES 3
/*
// REMARK: If upper ceiling is used, there is a chance that the last compartment
//     has only 1 capsule; while the other has 'ncaps'>1
// So the strategy:
//   1. use the floor instead of ceiling (but get 1 if the floor is 0)
//   2. distribute these remainder capsules to every compartments
//   from the distal-side
//   ncaps_branch = #caps on that branch
//   ncaps_cpt    = suggested #caps per compartment
//   The strategy make sure two compartments either having the same #capsules or
//   only 1 unit difference
#define N_COMPARTMENTS(ncaps_branch, ncaps_cpt)              \
  (int(floor(double(ncaps_branch) / double(ncaps_cpt))) > 0) \
      ? int(floor(double(ncaps_branch) / double(ncaps_cpt))) \
      : 1
                        */
//#define N_COMPARTMENTS(x) \
//  if (int(floor(double(x) / double(_compartmentSize))) > 0):int(floor(double(x) / double(_compartmentSize))):1
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
//#define INFERIOR_OLIVE

#ifdef INFERIOR_OLIVE
#include "InferiorOliveGlomeruliDetector.h"
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
#include <math.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <string>

#ifdef USING_CVC
extern void cvc(int, float*, float*, int*, bool);
extern void set_cvc_config(char*);
#endif
#endif

#ifdef HAVE_MPI
TissueContext* TissueFunctor::_tissueContext = 0;
int TissueFunctor::_instanceCounter = 0;
#endif

#define TRAJECTORY_TYPE 3

TissueFunctor::TissueFunctor()
    : CG_TissueFunctorBase(),
      _compartmentSize(0)
#ifdef HAVE_MPI
      ,
      _size(0),
      _rank(0),
      _nbrGridNodes(0),
      _channelTypeCounter(0),
      _electricalSynapseTypeCounter(0),
      _bidirectionalConnectionTypeCounter(0),
      _chemicalSynapseTypeCounter(0),
      _compartmentVariableTypeCounter(0),
      _junctionTypeCounter(0),
      _preSynapticPointTypeCounter(0),
      _synapticCleftTypeCounter(0),
      _endPointTypeCounter(0),
      _junctionPointTypeCounter(0),
      _forwardSolvePointTypeCounter(0),
      _backwardSolvePointTypeCounter(0),
      _readFromFile(false)
#endif
{
#ifdef HAVE_MPI
  if (_instanceCounter == 0) _tissueContext = new TissueContext();
  ++_instanceCounter;
#endif
}

TissueFunctor::TissueFunctor(TissueFunctor const& f)
    : CG_TissueFunctorBase(f),
      _compartmentSize(f._compartmentSize)
#ifdef HAVE_MPI
      ,
      _size(f._size),
      _rank(f._rank),
      _nbrGridNodes(f._nbrGridNodes),
      _compartmentVariableLayers(f._compartmentVariableLayers),
      _junctionLayers(f._junctionLayers),
      _endPointLayers(f._endPointLayers),
      _junctionPointLayers(f._junctionPointLayers),
      _channelLayers(f._channelLayers),
      _electricalSynapseLayers(f._electricalSynapseLayers),
      _bidirectionalConnectionLayers(f._bidirectionalConnectionLayers),
      _chemicalSynapseLayers(f._chemicalSynapseLayers),
      _preSynapticPointLayers(f._preSynapticPointLayers),
      _synapticCleftLayers(f._synapticCleftLayers),
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
#ifdef MICRODOMAIN_CALCIUM
      _microdomainOnBranch(f._microdomainOnBranch),
      _microdomainOnJunction(f._microdomainOnJunction),
#endif
      _channelTypeCounter(f._channelTypeCounter),
      _electricalSynapseTypeCounter(f._electricalSynapseTypeCounter),
      _bidirectionalConnectionTypeCounter(
          f._bidirectionalConnectionTypeCounter),
      _chemicalSynapseTypeCounter(f._chemicalSynapseTypeCounter),
      _compartmentVariableTypeCounter(f._compartmentVariableTypeCounter),
      _junctionTypeCounter(f._junctionTypeCounter),
      _preSynapticPointTypeCounter(f._preSynapticPointTypeCounter),
      _synapticCleftTypeCounter(f._synapticCleftTypeCounter),
      _endPointTypeCounter(f._endPointTypeCounter),
      _junctionPointTypeCounter(f._junctionPointTypeCounter),
      _forwardSolvePointTypeCounter(f._forwardSolvePointTypeCounter),
      _backwardSolvePointTypeCounter(f._backwardSolvePointTypeCounter),
      _tissueParams(f._tissueParams),
      _synapseGeneratorMap(f._synapseGeneratorMap),
      _synapseReceptorMaps(f._synapseReceptorMaps),
      _synapticCleftMaps(f._synapticCleftMaps),
      _probedLayoutsMap(f._probedLayoutsMap),
      _probedNodesMap(f._probedNodesMap),
      _compartmentVariableTypes(f._compartmentVariableTypes),
      _electricalSynapseTypesMap(f._electricalSynapseTypesMap),
      _bidirectionalConnectionTypesMap(f._bidirectionalConnectionTypesMap),
      _chemicalSynapseTypesMap(f._chemicalSynapseTypesMap),
      _compartmentVariableTypesMap(f._compartmentVariableTypesMap),
      _junctionTypesMap(f._junctionTypesMap),
      _channelTypesMap(f._channelTypesMap),
      _preSynapticPointTypesMap(f._preSynapticPointTypesMap),
      _synapticCleftTypesMap(f._synapticCleftTypesMap),
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
  if (f._connectorFunctor.get())
    f._connectorFunctor->duplicate(_connectorFunctor);
  if (f._probeFunctor.get()) f._probeFunctor->duplicate(_probeFunctor);
  if (f._MGSifyFunctor.get()) f._MGSifyFunctor->duplicate(_MGSifyFunctor);
  if (f._params.get()) f._params->duplicate(_params);
  _generatedChemicalSynapses = f._generatedChemicalSynapses;
  _nonGeneratedMixedChemicalSynapses = f._nonGeneratedMixedChemicalSynapses;
  _generatedSynapticClefts = f._generatedSynapticClefts;
  _generatedElectricalSynapses = f._generatedElectricalSynapses;
  _generatedBidirectionalConnections = f._generatedBidirectionalConnections;
  ++_instanceCounter;
}

// GOAL: this function automatically get called when
//   a TissueFunctor object is created
//   and perform different things
//     1.  build _tissueParams object
//     2.  detect touches
//     3.  generate spines
//  based on the inputs passed to the object
//!void TissueFunctor::userInitialize(LensContext* CG_c, String& commandLineArgs1, String& commandLineArgs2, 
//!				   String& compartmentParamFile, String& channelParamFile, String& synapseParamFile,
//!				   Functor*& layoutFunctor, Functor*& nodeInitFunctor, 
//!				   Functor*& connectorFunctor, Functor*& probeFunctor,
//!				   Functor*& MGSifyFunctor)
//!{
//!#ifdef HAVE_MPI
//!  _size = CG_c->sim->getNumProcesses();
//!  _rank = CG_c->sim->getRank();
//!#endif
//!  layoutFunctor->duplicate(_layoutFunctor);
//!  nodeInitFunctor->duplicate(_nodeInitFunctor);
//!  connectorFunctor->duplicate(_connectorFunctor);
//!  probeFunctor->duplicate(_probeFunctor);
//!  MGSifyFunctor->duplicate(_MGSifyFunctor);
//!
//!  // Validate inputs
//!  {
//!    TissueElement* element = dynamic_cast<TissueElement*>(_layoutFunctor.get());
//!    if (element == 0)
//!    {
//!      std::cerr << "Functor passed to TissueFunctor as argument 4 is not a "
//!                   "TissueElement!" << std::endl;
//!      exit(-1);
//!    }
//!  }
//!  {
//!    TissueElement* element =
//!        dynamic_cast<TissueElement*>(_nodeInitFunctor.get());
//!    if (element == 0)
//!    {
//!      std::cerr << "Functor passed to TissueFunctor as argument 5 is not a "
//!                   "TissueElement!" << std::endl;
//!      exit(-1);
//!    }
//!  }
//!  {
//!    TissueElement* element =
//!        dynamic_cast<TissueElement*>(_connectorFunctor.get());
//!    if (element == 0)
//!    {
//!      std::cerr << "Functor passed to TissueFunctor as argument 6 is not a "
//!                   "TissueElement!" << std::endl;
//!      exit(-1);
//!    }
//!  }
//!  {
//!    TissueElement* element = dynamic_cast<TissueElement*>(_probeFunctor.get());
//!    if (element == 0)
//!    {
//!      std::cerr << "Functor passed to TissueFunctor as argument 7 is not a "
//!                   "TissueElement!" << std::endl;
//!      exit(-1);
//!    }
//!  }
//!
//!#ifdef HAVE_MPI
//!  String command = "NULL ";
//!  command += commandLineArgs1;
//!  if (_tissueContext->_commandLine.parse(command.c_str()) == false)
//!  {
//!    std::cerr << "Error in simulation specification's commandLineArgs1 string "
//!                 "argument, TissueFunctor:" << std::endl;
//!    std::cerr << commandLineArgs1 << std::endl;
//!    exit(EXIT_FAILURE);
//!  }
//!#endif
//!
//!  std::string paramFilename(_tissueContext->_commandLine.getParamFileName());
//!  _tissueParams.readDevParams(paramFilename);
//!  _compartmentSize = _tissueContext->_commandLine.getCapsPerCpt();
//!
//!  // OPTION-1: re-read data structure from binary file
//!  FILE* data = NULL;
//!  if (_tissueContext->_commandLine.getBinaryFileName() != "" &&
//!      !_tissueContext->isInitialized())
//!  {
//!    if ((data = fopen(_tissueContext->_commandLine.getBinaryFileName().c_str(),
//!                      "rb")) != NULL)
//!    {
//!#ifdef HAVE_MPI
//!      _readFromFile = true;
//!      _tissueContext->_decomposition =
//!          new VolumeDecomposition(_rank, data, _size, _tissueContext->_tissue,
//!                                  _tissueContext->_commandLine.getX(),
//!                                  _tissueContext->_commandLine.getY(),
//!                                  _tissueContext->_commandLine.getZ());
//!      _tissueContext->readFromFile(data, _size, _rank);
//!      _tissueContext->setUpCapsules(_tissueContext->_nCapsules,
//!                                    TissueContext::NOT_SET, _rank,
//!                                    MAX_COMPUTE_ORDER);
//!      _tissueContext->setInitialized();
//!      fclose(data);
//!#endif
//!    }
//!  }
//!
//!  _tissueContext->seed(_rank);
//!
//!  // OPTION-2: regenerate data structure from .swc file
//!  if (!_tissueContext->isInitialized())
//!  {
//!    neuroGen(&_tissueParams, CG_c);
//!    MPI_Barrier(MPI_COMM_WORLD);
//!    neuroDev(&_tissueParams, CG_c);
//!  }
//!
//!#ifdef HAVE_MPI
//!  command = "NULL ";
//!  command += commandLineArgs2;
//!  if (_tissueContext->_commandLine.parse(command.c_str()) == false)
//!  {
//!    std::cerr << "Error in simulation specification's commandLineArgs2 string "
//!                 "argument, TissueFunctor:" << std::endl;
//!    std::cerr << commandLineArgs2 << std::endl;
//!    exit(EXIT_FAILURE);
//!  }
//!#endif
//!
//!  paramFilename = _tissueContext->_commandLine.getParamFileName();
//!  /*strcpy(paramFilename,
//!         _tissueContext->_commandLine.getParamFileName().c_str());*/
//!  _tissueParams.readDetParams(paramFilename);
//!  _tissueParams.readCptParams(compartmentParamFile.c_str());
//!  _tissueParams.readChanParams(channelParamFile.c_str());
//!  _tissueParams.readSynParams(synapseParamFile.c_str());
//!
//!  if (!_tissueContext->isInitialized())
//!  {
//!    touchDetect(&_tissueParams, CG_c);
//!    createSpines(&_tissueParams, CG_c);  // need to create spines here
//!    _tissueContext->setInitialized();
//!  }
//!
//!  if ((_tissueContext->_commandLine.getOutputFormat() == "b" ||
//!       _tissueContext->_commandLine.getOutputFormat() == "bt") &&
//!      _tissueContext->_commandLine.getBinaryFileName() != "" &&
//!      !_readFromFile && CG_c->sim->isSimulatePass())
//!  {
//!#ifdef HAVE_MPI
//!    MPI_Barrier(MPI_COMM_WORLD);
//!    _tissueContext->writeToFile(_size, _rank);
//!#endif
//!  }
//!}
void TissueFunctor::userInitialize(
    LensContext* CG_c, String& commandLineArgs1, String& commandLineArgs2,
    String& compartmentParamFile, String& channelParamFile,
    String& synapseParamFile, Functor*& layoutFunctor,
    Functor*& nodeInitFunctor, Functor*& connectorFunctor,
    Functor*& probeFunctor)
{
#ifdef HAVE_MPI
  _size = CG_c->sim->getNumProcesses();
  _rank = CG_c->sim->getRank();
#endif
  layoutFunctor->duplicate(_layoutFunctor);
  nodeInitFunctor->duplicate(_nodeInitFunctor);
  connectorFunctor->duplicate(_connectorFunctor);
  probeFunctor->duplicate(_probeFunctor);

  // Validate inputs
  {
    TissueElement* element = dynamic_cast<TissueElement*>(_layoutFunctor.get());
    if (element == 0)
    {
      std::cerr << "Functor passed to TissueFunctor as argument 4 is not a "
                   "TissueElement!" << std::endl;
      exit(-1);
    }
  }
  {
    TissueElement* element =
        dynamic_cast<TissueElement*>(_nodeInitFunctor.get());
    if (element == 0)
    {
      std::cerr << "Functor passed to TissueFunctor as argument 5 is not a "
                   "TissueElement!" << std::endl;
      exit(-1);
    }
  }
  {
    TissueElement* element =
        dynamic_cast<TissueElement*>(_connectorFunctor.get());
    if (element == 0)
    {
      std::cerr << "Functor passed to TissueFunctor as argument 6 is not a "
                   "TissueElement!" << std::endl;
      exit(-1);
    }
  }
  {
    TissueElement* element = dynamic_cast<TissueElement*>(_probeFunctor.get());
    if (element == 0)
    {
      std::cerr << "Functor passed to TissueFunctor as argument 7 is not a "
                   "TissueElement!" << std::endl;
      exit(-1);
    }
  }

#ifdef HAVE_MPI
  String command = "NULL ";
  command += commandLineArgs1;
  if (_tissueContext->_commandLine.parse(command.c_str()) == false)
  {
    std::cerr << "Error in simulation specification's commandLineArgs1 string "
                 "argument, TissueFunctor:" << std::endl;
    std::cerr << commandLineArgs1 << std::endl;
    exit(EXIT_FAILURE);
  }
#endif

  std::string paramFilename(_tissueContext->_commandLine.getParamFileName());
  _tissueParams.readDevParams(paramFilename);
  _compartmentSize = _tissueContext->_commandLine.getCapsPerCpt();

  // OPTION-1: re-read data structure from binary file
  FILE* data = NULL;
  if (_tissueContext->_commandLine.getBinaryFileName() != "" &&
      !_tissueContext->isInitialized())
  {
    if ((data = fopen(_tissueContext->_commandLine.getBinaryFileName().c_str(),
                      "rb")) != NULL)
    {
#ifdef HAVE_MPI
      _readFromFile = true;
      _tissueContext->_decomposition =
          new VolumeDecomposition(_rank, data, _size, _tissueContext->_tissue,
                                  _tissueContext->_commandLine.getX(),
                                  _tissueContext->_commandLine.getY(),
                                  _tissueContext->_commandLine.getZ());
      _tissueContext->readFromFile(data, _size, _rank);
      _tissueContext->setUpCapsules(_tissueContext->_nCapsules,
                                    TissueContext::NOT_SET, _rank,
                                    MAX_COMPUTE_ORDER);
      _tissueContext->setInitialized();
      fclose(data);
#endif
    }
  }

  _tissueContext->seed(_rank);

  // OPTION-2: regenerate data structure from .swc file
  if (!_tissueContext->isInitialized())
  {
    neuroGen(&_tissueParams, CG_c);
    MPI_Barrier(MPI_COMM_WORLD);
    neuroDev(&_tissueParams, CG_c);
  }

#ifdef HAVE_MPI
  command = "NULL ";
  command += commandLineArgs2;
  if (_tissueContext->_commandLine.parse(command.c_str()) == false)
  {
    std::cerr << "Error in simulation specification's commandLineArgs2 string "
                 "argument, TissueFunctor:" << std::endl;
    std::cerr << commandLineArgs2 << std::endl;
    exit(EXIT_FAILURE);
  }
#endif

  paramFilename = _tissueContext->_commandLine.getParamFileName();
  /*strcpy(paramFilename,
         _tissueContext->_commandLine.getParamFileName().c_str());*/
  _tissueParams.readDetParams(paramFilename);
  _tissueParams.readCptParams(compartmentParamFile.c_str());
  _tissueParams.readChanParams(channelParamFile.c_str());
  _tissueParams.readSynParams(synapseParamFile.c_str());

  if (!_tissueContext->isInitialized())
  {
    touchDetect(&_tissueParams, CG_c);
    createSpines(&_tissueParams, CG_c);  // need to create spines here
    _tissueContext->setInitialized();
  }

  if ((_tissueContext->_commandLine.getOutputFormat() == "b" ||
       _tissueContext->_commandLine.getOutputFormat() == "bt") &&
      _tissueContext->_commandLine.getBinaryFileName() != "" &&
      !_readFromFile && CG_c->sim->isSimulatePass())
  {
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    _tissueContext->writeToFile(_size, _rank);
#endif
  }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// GOAL: growth neurons that have either AXON_PAR, BASAL_PAR
//                                       APICAL_PAR
//                           that are not-NULL
void TissueFunctor::neuroGen(Params* params, LensContext* CG_c)
{
#ifdef HAVE_MPI
  double start, now, then;
  start = then = MPI_Wtime();
  bool* somaGenerated = 0;
  std::string baseParFileName = "NULL", swcFileName = "NULL";
  std::vector<double> complexities;
  int neuronBegin = 0, neuronEnd = 0;
  int nNeuronsGenerated = 0;
  RNG rng;

  MPI_Barrier(MPI_COMM_WORLD);

  NeuroDevCommandLine& commandLine = _tissueContext->_commandLine;
  int nthreads = commandLine.getNumberOfThreads();
  std::string tissueFileName(commandLine.getInputFileName());
  /*  char tissueFileName[256];
    int tissueFileNameLength = commandLine.getInputFileName().length();
    strcpy(tissueFileName, commandLine.getInputFileName().c_str());
    char* ext = &tissueFileName[tissueFileNameLength - 3];*/
  std::string ext = tissueFileName.substr(tissueFileName.length() - 3);
  if (ext == "bin")
  {
    // if (strcmp(ext, "bin") == 0) {
    if (_rank == 0)
      std::cerr << "NeuroGen tissue file must be a text file." << std::endl;
    exit(-1);
  }
  if (_rank == 0)
  {
    std::cout << "Tissue file name: " << tissueFileName << std::endl
              << std::endl;
    std::cout << "Generating neurons...\n";
  }

  for (int branchType = 0; branchType < N_BRANCH_TYPES; ++branchType)
  {  // 0:axon, 1:basal, 2:apical
    std::string btype;
    if (branchType == 0)
      btype = "axons";
    else if (branchType == 1)
      btype = "denda";
    else if (branchType == 2)
      btype = "dendb";
    std::map<std::string, BoundingSurfaceMesh*> boundingSurfaceMap;
    bool stdout = false;
    bool fout = true;

    double composite = 0.0;
    std::string compositeSwcFileName;
    NeurogenParams** ng_params = 0;
    // char** fileNames = 0;
    std::vector<std::string> fileNames;

    if (branchType == 0)
    {
      // only need to do the following once per neuron
      double totalComplexity = 0.0;
      std::ifstream tissueFile(tissueFileName.c_str());
      while (tissueFile.good())
      {
        std::string line;
        getline(tissueFile, line);
        if (line != "" && line.at(0) != '#')
        {
          std::string str = line;
          std::stringstream strstr(str);
          std::istream_iterator<std::string> it(strstr);
          std::istream_iterator<std::string> end;
          std::vector<std::string> results(it, end);
          bool genParams = false;
          if (results.size() >= PAR_FILE_INDEX + N_BRANCH_TYPES)
          {
            // use first paramfile in tissue to seed RNG below
            for (int bt = 0; bt < N_BRANCH_TYPES; ++bt)
            {
              if (results.at(PAR_FILE_INDEX + bt) != "NULL")
              {
                baseParFileName = results.at(PAR_FILE_INDEX + bt);
                genParams = true;
                break;
              }
            }
            if (genParams)
            {
              double complexity = 0.0;
              std::ifstream testFile(results.at(0).c_str());
              if (!testFile)
              {
                if (results.size() > PAR_FILE_INDEX + N_BRANCH_TYPES)
                {
                  complexity =
                      atof(results.at(PAR_FILE_INDEX + N_BRANCH_TYPES).c_str());
                }
                else
                  complexity = 1.0;
              }
              else
                testFile.close();
              totalComplexity += complexity;
              complexities.push_back(complexity);
            }
          }
        }
      }
      tissueFile.close();
      int nNeurons = complexities.size();
      int bufSize = (nNeurons > 0) ? nNeurons : 1;
      somaGenerated = new bool[bufSize];
      double targetComplexity = totalComplexity / double(_size);
      double runningComplexity = 0.0;
      int count = 0, divisor = _size;
      bool assigned = false;
      for (int i = 0; i < nNeurons; ++i)
      {
        somaGenerated[i] = false;
        if ((runningComplexity += complexities[i]) >= targetComplexity)
        {
          --divisor;
          neuronBegin = neuronEnd;
          if (neuronBegin == i || complexities[i] == runningComplexity ||
              runningComplexity - targetComplexity <
                  targetComplexity - (runningComplexity - complexities[i]))
          {
            totalComplexity -= runningComplexity;
            targetComplexity = totalComplexity / divisor;
            runningComplexity = 0.0;
            neuronEnd = i + 1;
          }
          else
          {
            totalComplexity -= (runningComplexity - complexities[i]);
            targetComplexity = totalComplexity / divisor;
            runningComplexity = complexities[i];
            neuronEnd = i;
          }
          if (count == _rank)
          {
            assigned = true;
            break;
          }
          ++count;
        }
      }
      if (!assigned) neuronBegin = neuronEnd;
      NeurogenParams ng_params_p(baseParFileName, _rank);
      rng.reSeed(lrandom(ng_params_p._rng), _rank);
      nNeuronsGenerated = neuronEnd - neuronBegin;
    }
    int bufSize = (nNeuronsGenerated > 0) ? nNeuronsGenerated : 1;
    ng_params = new NeurogenParams* [bufSize];
    // fileNames = new char* [bufSize];
    for (int i = 0; i < bufSize; ++i) ng_params[i] = 0;

    // int ln = strlen(tissueFileName);
    int ln = tissueFileName.length();
    std::string statsFileName(tissueFileName);
    std::string parsFileName(tissueFileName);
    statsFileName.erase(ln - 4, 4);
    parsFileName.erase(ln - 4, 4);

    struct stat st = {0};
    std::string statFolder = "./stats/";
    if (stat(statFolder.c_str(), &st) == -1)
    {
      mkdir (statFolder.c_str(), 0700);
    }
    std::ostringstream statsFileNameStream, parsFileNameStream;
    statsFileNameStream << statFolder << statsFileName << "." << btype << ".out";
    statsFileName = statsFileNameStream.str();
    parsFileNameStream << statFolder << parsFileName << "." << btype << ".par";
    parsFileName = parsFileNameStream.str();

    if (composite > 0)
    {
      compositeSwcFileName = tissueFileName;
      compositeSwcFileName.erase(ln - 4, 4);
      std::ostringstream compositeSwcFileNameStream;
      compositeSwcFileNameStream << compositeSwcFileName << "." << _rank
                                 << ".swc";
      compositeSwcFileName = compositeSwcFileNameStream.str();
    }

    int neuronID = 0, idx = 0;
    std::ifstream tissueFile(tissueFileName.c_str());
    if (tissueFile.is_open())
    {
      while (tissueFile.good())
      {
        std::string line;
        getline(tissueFile, line);
        if (line != "" && line.at(0) != '#')
        {
          if (neuronID >= neuronBegin && neuronID < neuronEnd)
          {
            std::string str = line;
            // construct a stream from the string
            std::stringstream strstr(str);

            // use stream iterators to copy the stream to the vector as
            // whitespace separated strings
            std::istream_iterator<std::string> it(strstr);
            std::istream_iterator<std::string> end;
            std::vector<std::string> results(it, end);
            if (results.size() >= PAR_FILE_INDEX + N_BRANCH_TYPES)
            {
              /*fileNames[idx] = new char[results.at(0).length() + 1];
              strcpy(fileNames[idx], results.at(0).c_str());*/
              fileNames.push_back(results.at(0));
              if (complexities[idx] > 0 &&
                  results.at(PAR_FILE_INDEX + branchType) != "NULL")
              {
                ng_params[idx] = new NeurogenParams(
                    results.at(PAR_FILE_INDEX + branchType), _rank);
                ng_params[idx]->RandSeed = lrandom(rng);
                ng_params[idx]->_rng.reSeedShared(ng_params[idx]->RandSeed);
                ng_params[idx]->startX = atof(results.at(4).c_str());
                ng_params[idx]->startY = atof(results.at(5).c_str());
                ng_params[idx]->startZ = atof(results.at(6).c_str());
                std::map<std::string, BoundingSurfaceMesh*>::iterator miter =
                    boundingSurfaceMap.find(ng_params[idx]->boundingSurface);
                if (miter == boundingSurfaceMap.end())
                  boundingSurfaceMap[ng_params[idx]->boundingSurface] =
                      new BoundingSurfaceMesh(ng_params[idx]->boundingSurface);
              }
              ++idx;
            }
            else if (results.size() > 0)
            {
              std::cerr << "Error in Tissue File at neuron " << idx << "."
                        << std::endl;
              exit(-1);
            }
          }
          neuronID++;
        }
      }
    }
    else
    {
      std::cerr << "Cannot open tissue file!" << tissueFileName << std::endl;
      exit(EXIT_FAILURE);
    }
    tissueFile.clear();
    tissueFile.close();

    Neurogenesis NG(_rank, _size, nthreads, statsFileName, parsFileName, stdout,
                    fout, branchType + 2, boundingSurfaceMap);
    NG.run(neuronBegin, nNeuronsGenerated, ng_params, fileNames, somaGenerated);

    if (composite > 0 && _rank == 0)
      CompositeSwc(tissueFileName.c_str(), compositeSwcFileName.c_str(),
                   composite, false);

    for (int nid = 0; nid < nNeuronsGenerated; ++nid)
    {
      // delete[] fileNames[nid];
      delete ng_params[nid];
    }
    // delete[] fileNames;
    delete[] ng_params;
    std::map<std::string, BoundingSurfaceMesh*>::iterator miter,
        mend = boundingSurfaceMap.end();
    for (miter = boundingSurfaceMap.begin(); miter != mend; ++miter)
      delete miter->second;
  }
  delete[] somaGenerated;
  now = MPI_Wtime();
  if (_rank == 0)
    printf("\nNeuron generation compute time : %lf\n\n", now - start);
#endif
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void TissueFunctor::neuroDev(Params* params, LensContext* CG_c)
{
  if (_rank == 0) printf("Developing tissue...\n");
#ifdef HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD);

  NeuroDevCommandLine& commandLine = _tissueContext->_commandLine;

  bool clientConnect = commandLine.getClientConnect();
#ifdef USING_CVC
  if (clientConnect) set_cvc_config("./cvc.config");
#endif

  // file containing lists of .swc files
  std::string inputFilename(commandLine.getInputFileName());  // e.g.
                                                              // tissues.txt

  _tissueContext->_tissue = new Tissue(_size, _rank);

  bool resample = commandLine.getResample();

  Communicator* communicator = new Communicator();
  Director* director = new Director(communicator);

  bool dumpResampledNeurons =
      ((_tissueContext->_commandLine.getOutputFormat() == "t" ||
        _tissueContext->_commandLine.getOutputFormat() == "bt"))
          ? true
          : false;

  int X = commandLine.getX();
  int Y = commandLine.getY();
  int Z = commandLine.getZ();

  int nSlicers = commandLine.getNumberOfSlicers();
  if (nSlicers == 0 || nSlicers > _size) nSlicers = _size;
  int nSegmentForceDetectors = commandLine.getNumberOfDetectors();
  if (nSegmentForceDetectors == 0 || nSegmentForceDetectors > _size)
    nSegmentForceDetectors = _size;

  _tissueContext->_neuronPartitioner = new NeuronPartitioner(
      _rank, inputFilename, resample, dumpResampledNeurons,
      commandLine.getPointSpacing());

  VolumeDecomposition* volumeDecomposition = 0;
  std::string ext = inputFilename.substr(inputFilename.length() - 3);
  if (ext == "bin")
  {
    // if (strcmp(ext, "bin") == 0) {
    _tissueContext->_neuronPartitioner->partitionBinaryNeurons(
        nSlicers, nSegmentForceDetectors, _tissueContext->_tissue);
  }
  else
  {
    _tissueContext->_neuronPartitioner->partitionTextNeurons(
        nSlicers, nSegmentForceDetectors, _tissueContext->_tissue);
  }
  _tissueContext->_decomposition = volumeDecomposition =
      new VolumeDecomposition(_rank, NULL, _size, _tissueContext->_tissue, X, Y,
                              Z);

#ifdef VERBOSE
  if (_rank == 0)
    std::cout << "Max Branch Order = "
              << _tissueContext->_tissue->getMaxBranchOrder() << std::endl;
#endif

  SegmentForceAggregator* segmentForceAggregator = new SegmentForceAggregator(
      _rank, nSlicers, _size, _tissueContext->_tissue);

  AllInTouchSpace detectionTouchSpace;  // OBJECT CHOICE : PARAMETERIZABLE
  SegmentForceDetector* segmentForceDetector = new SegmentForceDetector(
      _rank, nSlicers, _size, commandLine.getNumberOfThreads(),
      &_tissueContext->_decomposition, &detectionTouchSpace,
      _tissueContext->_neuronPartitioner, params);

  int maxIterations = commandLine.getMaxIterations();
  if (maxIterations < 0)
  {
    std::cerr << "max-iterations must be >= 0!" << std::endl;
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  double Econ = commandLine.getEnergyCon();
  double dT = commandLine.getTimeStep();
  double E = 0, dE = 0, En = 0;

  TissueGrowthSimulator TissueSim(
      _size, _rank, _tissueContext->_tissue, director, segmentForceDetector,
      segmentForceAggregator, params, commandLine.getInitialFront());

  AllInSegmentSpace allInSegmentSpace;

  FrontSegmentSpace frontSegmentSpace(
      TissueSim);  // OBJECT CHOICE : PARAMETERIZABLE
  FrontLimitedSegmentSpace frontLimitedSegmentSpace(
      TissueSim);  // OBJECT CHOICE : PARAMETERIZABLE

  std::vector<std::pair<std::string, unsigned int> > probeKey;
  probeKey.push_back(std::pair<std::string, unsigned int>(
      std::string("BRANCHTYPE"), TRAJECTORY_TYPE));
  SegmentKeySegmentSpace gliaSegmentSpace(probeKey);
  NOTSegmentSpace notGliaSegmentSpace(&gliaSegmentSpace);

  ANDSegmentSpace coveredSegmentSpace(&frontLimitedSegmentSpace,
                                      &notGliaSegmentSpace);
  ANDSegmentSpace gliaOnFrontSegmentSpace(&frontSegmentSpace,
                                          &gliaSegmentSpace);
  ORSegmentSpace gliaOnFrontFrontLimitedSegmentSpace(&coveredSegmentSpace,
                                                     &gliaOnFrontSegmentSpace);

  NeuroDevTissueSlicer* neuroDevTissueSlicer = new NeuroDevTissueSlicer(
      _rank, nSlicers, _size, _tissueContext->_tissue,
      &_tissueContext->_decomposition, &frontSegmentSpace, params,
      segmentForceDetector->getEnergy());

#ifdef VERBOSE
  if (_rank == 0)
    printf("Maximum Front level = %d\n", TissueSim.getMaxFrontNumber());
#endif
  bool attemptConnect = true;
  unsigned iteration = 0;
  int nspheres = 0;
  float* positions = 0, * radii = 0;
  int* types = 0;
  double start, now, then;
  start = then = MPI_Wtime();
  director->addCommunicationCouple(neuroDevTissueSlicer, segmentForceDetector);
  volumeDecomposition->resetCriteria(&gliaOnFrontFrontLimitedSegmentSpace);
  neuroDevTissueSlicer->resetSegmentSpace(&coveredSegmentSpace);
  segmentForceDetector->updateCoveredSegments(true);
  director->iterate();
  segmentForceDetector->updateCoveredSegments(false);
  director->addCommunicationCouple(segmentForceDetector,
                                   segmentForceAggregator);
  bool grow = TissueSim.AdvanceFront() && maxIterations > 0;
  neuroDevTissueSlicer->resetSegmentSpace(&frontSegmentSpace);

  while (grow)
  {
#ifdef VERBOSE
    if (_rank == 0) printf("Front level %d", TissueSim.getFrontNumber());
    if (!grow && _rank == 0) printf(" <FINAL> ");
    if (_rank == 0) printf("\n");
#endif
    En = 0;
    iteration = 0;
    do
    {
#ifdef USING_CVC
      if (clientConnect)
      {
        _tissueContext->_tissue->getVisualizationSpheres(
            vizSpace, nspheres, positions, radii, types);
        cvc(nspheres, positions, radii, types, attemptConnect);
        attemptConnect = false;
      }
#endif
      /* computeForces is inside this front simulation step, which is equivalent
         to an entire step through the Director's CommunicationCouple list */
      TissueSim.FrontSimulationStep(iteration, dT, E);
      dE = E - En;
      En = E;
      now = MPI_Wtime();
      if (_rank == 0 && iteration < maxIterations)
        std::cout << "front = " << TissueSim.getFrontNumber()
                  << ", begin = " << iteration << ", E = " << E
                  << ", dE = " << dE << ", T = " << now
                  << ", dT = " << now - then << "." << std::endl;
      then = now;
    } while (fabs(dE) > Econ && iteration < maxIterations);
    if (_rank == 0)
      std::cout << "front = " << TissueSim.getFrontNumber()
                << ", end = " << iteration << ", E = " << E << ", dE = " << dE
                << "." << std::endl;
#ifdef USING_CVC
    attemptConnect = true;
#endif
    director->clearCommunicationCouples();
    director->addCommunicationCouple(neuroDevTissueSlicer,
                                     segmentForceDetector);
    volumeDecomposition->resetCriteria(&gliaOnFrontFrontLimitedSegmentSpace);
    neuroDevTissueSlicer->resetSegmentSpace(&coveredSegmentSpace);
    segmentForceDetector->updateCoveredSegments(true);
    director->iterate();
    segmentForceDetector->updateCoveredSegments(false);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(neuroDevTissueSlicer,
                                     segmentForceDetector);
    director->addCommunicationCouple(segmentForceDetector,
                                     segmentForceAggregator);
    _tissueContext->_tissue->clearSegmentForces();
    grow = TissueSim.AdvanceFront();
    neuroDevTissueSlicer->resetSegmentSpace(&frontSegmentSpace);
  }
  volumeDecomposition->resetCriteria(&allInSegmentSpace);

  now = MPI_Wtime();
  if (_rank == 0)
    printf("\nTissue development compute time : %lf\n\n", now - start);

  // output the combined .swc file
  FILE* tissueOutFile = 0;
  if (_tissueContext->_commandLine.getOutputFormat() == "t" ||
      _tissueContext->_commandLine.getOutputFormat() == "bt")
  {
    std::string outExtension(".developed");
    if (maxIterations > 0)
    {
      _tissueContext->_tissue->outputTextNeurons(outExtension, 0, 0);
    }
    if (commandLine.getOutputFileName() != "")
    {
      int nextToWrite = 0, written = 0, segmentsWritten = 0, globalOffset = 0;
      while (nextToWrite < _size)
      {
        MPI_Allreduce((void*)&written, (void*)&nextToWrite, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce((void*)&segmentsWritten, (void*)&globalOffset, 1, MPI_INT,
                      MPI_SUM, MPI_COMM_WORLD);
        if (nextToWrite == _rank)
        {
          if ((tissueOutFile = fopen(commandLine.getOutputFileName().c_str(),
                                     (_rank == 0) ? "wt" : "at")) == NULL)
          {
            printf("Could not open the output file %s!\n",
                   commandLine.getOutputFileName().c_str());
            MPI_Finalize();
            exit(EXIT_FAILURE);
          }
          segmentsWritten = _tissueContext->_tissue->outputTextNeurons(
              outExtension, tissueOutFile, globalOffset);
          fclose(tissueOutFile);
          written = 1;
        }
      }
    }
  }

  delete segmentForceAggregator;
  delete segmentForceDetector;
  delete communicator;
  delete director;
  delete neuroDevTissueSlicer;
  delete[] positions;
  delete[] radii;
  delete[] types;
#endif
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// GOAL:
//   perform
//   1. generate Capsules from Segments structure
//   2. detect touches between any 2 capsules
void TissueFunctor::touchDetect(Params* params, LensContext* CG_c)
{
#ifdef IDEA1
  _tissueContext->_params = params;
#endif

#ifdef HAVE_MPI
  double start, now, then;
  start = then = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  NeuroDevCommandLine& commandLine = _tissueContext->_commandLine;

  int nSlicers = _tissueContext->_neuronPartitioner->getNumberOfSlicers();

  // number of processes to run touch-detection
  // can be the number of MPI processes or
  // the number as passed though the command-line (look Document for the right
  // option)
  int nTouchDetectors = commandLine.getNumberOfDetectors();
  if (nTouchDetectors == 0) nTouchDetectors = _size;

  bool autapses = false; // neuron that self-touch it

  SynapseTouchSpace electricalSynapseTouchSpace(SynapseTouchSpace::ELECTRICAL,
                                                params, autapses);

  SynapseTouchSpace chemicalSynapseTouchSpace(SynapseTouchSpace::CHEMICAL,
                                              params, autapses);

  ORTouchSpace detectionTouchSpace(electricalSynapseTouchSpace,
                                   chemicalSynapseTouchSpace);

  TouchDetectTissueSlicer* touchDetectTissueSlicer =
      new TouchDetectTissueSlicer(_rank, nSlicers, nTouchDetectors,
                                  _tissueContext->_tissue,
                                  &_tissueContext->_decomposition,
                                  _tissueContext, params, MAX_COMPUTE_ORDER);
  TouchSpace* touchCommunicateSpace = 0;  // OBJECT CHOICE : PARAMETERIZABLE

  TouchDetector* touchDetector = new TouchDetector(
      _rank, nSlicers, nTouchDetectors, MAX_COMPUTE_ORDER,
      commandLine.getNumberOfThreads(), commandLine.getAppositionSamplingRate(),
      &_tissueContext->_decomposition, &detectionTouchSpace,
      touchCommunicateSpace, _tissueContext->_neuronPartitioner, _tissueContext,
      params);

#ifdef INFERIOR_OLIVE
  GlomeruliDetector* glomeruliDetector =
      new InferiorOliveGlomeruliDetector(_tissueContext);
#endif

  LENSTissueSlicer* lensTissueSlicer = new LENSTissueSlicer(
      _rank, nSlicers, nTouchDetectors, _tissueContext, params);
  TouchAggregator* touchAggregator =
      new TouchAggregator(_rank, nTouchDetectors, _tissueContext);

  Communicator* communicator = new Communicator();
  Director* director = new Director(communicator);

  if (_rank == 0)
  {
    printf("Using %s decomposition.\n\n",
           commandLine.getDecomposition().c_str());
    printf("Detecting touches...\n\n");
  }

  if (commandLine.getDecomposition() == "volume" ||
      commandLine.getDecomposition() == "cost-volume")
  {
    touchDetectTissueSlicer->sendLostDaughters(false);
#ifdef INFERIOR_OLIVE
    touchDetectTissueSlicer->addTolerance(
        glomeruliDetector->getGlomeruliSpacing());
#endif
    ////TUAN TESTING
    //touchDetectTissueSlicer->addTolerance(
    //    20.0);
    ////END TUAN TESTING
    
    touchDetector->setPass(TissueContext::FIRST_PASS);
    touchDetector->unique(true);
    //touchDetector->unique(false);
#ifdef SYNAPSE_PARAMS_TOUCH_DETECT
    _tissueContext->_decomposition->resetCriteria(&detectionTouchSpace);
#endif
    director->addCommunicationCouple(touchDetectTissueSlicer, touchDetector);
    director->iterate();
    touchDetector->setUpCapsules();

    if (commandLine.getDecomposition() == "cost-volume")
    {
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
    else
    {
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
#ifdef IDEA1
    _tissueContext->makeProperComputeBranch();
#endif
    director->iterate();//exchange Touches across MPI processes

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
#ifdef IDEA1
    //_tissueContext->makeProperComputeBranch();
#endif
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
  else if (commandLine.getDecomposition() == "neuron")
  {
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
    _tissueContext->_decomposition = _tissueContext->_neuronPartitioner;
    touchDetector->resetBufferSize(false);
    director->iterate();

    touchDetector->resetBufferSize(true);
    _tissueContext->_decomposition = volumeDecomposition;
    director->iterate();
    touchDetector->setUpCapsules();
    int nVolCaps = _tissueContext->_nCapsules;

    _tissueContext->_decomposition = _tissueContext->_neuronPartitioner;
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
    _tissueContext->_decomposition = volumeDecomposition;
    director->clearCommunicationCouples();
    director->addCommunicationCouple(touchDetectTissueSlicer, touchDetector);
    director->iterate();
    _tissueContext->clearCapsuleMaps();
    touchDetector->setUpCapsules();
    assert(nVolCaps == _tissueContext->_nCapsules);
    std::map<double, int> firstPassVolumeCapsuleMap, secondPassVolumeCapsuleMap;
    _tissueContext->getCapsuleMaps(firstPassVolumeCapsuleMap,
                                   secondPassVolumeCapsuleMap);

    _tissueContext->_decomposition = _tissueContext->_neuronPartitioner;
    touchDetector->resetBufferSize(false);
    touchDetector->receiveAtBufferOffset(true);
    director->iterate();
    _tissueContext->clearCapsuleMaps();
    touchDetector->setCapsuleOffset(nVolCaps);
    touchDetector->setUpCapsules();

    std::map<double, int> firstPassNeuronCapsuleMap, secondPassNeuronCapsuleMap;
    _tissueContext->getCapsuleMaps(firstPassNeuronCapsuleMap,
                                   secondPassNeuronCapsuleMap);

    _tissueContext->resetCapsuleMaps(firstPassVolumeCapsuleMap,
                                     secondPassVolumeCapsuleMap);
    touchDetector->setCapsuleOffset(-nVolCaps);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(lensTissueSlicer, touchDetector);
    director->iterate();

    _tissueContext->resetCapsuleMaps(firstPassNeuronCapsuleMap,
                                     secondPassNeuronCapsuleMap);
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
  else
  {
    std::cerr << "Unrecognized decomposition : "
              << commandLine.getDecomposition() << std::endl;
    exit(EXIT_FAILURE);
  }

  Touch::compare c(0);
  _tissueContext->_touchVector.sort(c);
  now = MPI_Wtime();
  if (_rank == 0) printf("Touch detection compute time : %lf\n\n", now - start);
#endif
}

// GOAL:
//   create spines by
//   0. check for Touch(capA, capB)
//     if capA represents bouton, capB represent denshaft
//     then add new spine structure (2 compartment - head+neck)
//   1. generate new Capsules information
//     with keys for 2 new capsules, add them to the list of _capsules
//     modify Touch(capA,capB) --> Touch(capA,head)
//   2. adding new touches,
//       Touch(neck,capB)
void TissueFunctor::createSpines(Params* params, LensContext* CG_c)
{  // not implemented yet
   // loop throughs all  _tissueContext->_touchVector
   //
   /*	TouchVector::TouchIterator titer = _tissueContext->_touchVector.begin(),
                   tend = _tissueContext->_touchVector.end();
           for (; titer != tend; ++titer)
           {
           if (!_tissueContext->isLensTouch(*titer, _rank)) continue;
           key_size_t key1, key2;
             key1 = titer->getKey1();
             key2 = titer->getKey2();
           Capsule* preCapsule =
               &_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key1)];
           Capsule* postCapsule =
               &_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key2)];
           ComputeBranch* postBranch = postCapsule->getBranch();
                                   ComputeBranch* preBranch =
      preCapsule->getBranch();
           assert(postBranch);
         //check  which one is axon branch
           unsigned int indexPre, indexPost;
           bool preJunction = false;
           bool postJunction = false;
      //check if the touch (capsuleA, capsuleB) fit into the spine criteria
            //   preCapsule = capsuleA
            //if so, check the prob. for forming a  spine
            //
            //if so create the  spine
            // 1. create bouton? (maybe we don't need)
            // 2. create spine-head capsule  D
            // 3. create spine-neck capsule  E
            // 3.b connect spine-head to spine-neck as 2 adjacent components
      belonging
            //    to the same neuron of the shaft (Q: does using this same neuron
      index ok?)
            // 4. modify the existing touch
            //      --> (capsuleA, D)
            // 5. create new touch
            //      --> (E, capsuleB)
            // 6. assign key to each new capsule ?
           }
                                   */
}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// GOAL:
//   get call during NodeInit()
//   when nodeCategory == 'CompartmentVariables' or 'Junctions'
//                                or (in the case of 'Channels')
//                                 'JunctionChannels' or 'BranchChannels'
//
//     for a given 'nodeType' (e.g. 'Voltage'), at
//     a particular grid-node-index 'nodeIndex'
//     'densityIndex' = # of instances of that nodeType
//     find the associated ComputeBranch object
//    then
//    1. Create the CompartmentDimension array for that ComputeBranch
//          a ComputeBranch represents either a regular branch or an explicit
//          junction
//    2. Initialize the size of array of data members to the same size with that
//    above
// PARAMS:
//    nodeCategory = one of value above
//    nodeType    = any value defined for the Layer of the above nodeCategory
//    nodeIndex  = index in the grid (where it holds one or many instances
//                 of such NodeType's nodeCategory[nodeType])
//    densityIndex = index in the density vector, i.e. the exact instance to handle 
int TissueFunctor::compartmentalize(LensContext* lc, NDPairList* params,
                                    std::string& nodeCategory,
                                    std::string& nodeType, int nodeIndex,
                                    int densityIndex)
{
  int rval = 1;
  if (nodeCategory == "CompartmentVariables" ||
      nodeCategory == "BranchChannels" || nodeCategory == "JunctionChannels")
  {  // 1, regular branch or 2= 'channels' on regular branch, or 3 = 'channels'
     // on junction (i.e. a single-compartment branch)
    std::vector<int> size;  // holding # compartments at each branch
    // Does 2 things:
    //   1. get to know the 'size' (so that the data members as array
    //        can be initialized to that size)
    //   2. if nodeCategory=CompartmentVariables,
    //        also create CompartmentDimension(x,y,z,r,dist2soma,surface_area,
    //        volume)
    //            for every compartments on that ComputeBranch
    //            which is tracked by _tissueContext->_branchDimensionsMap
    if (nodeCategory == "CompartmentVariables" ||
        nodeCategory == "BranchChannels")
    {
      ComputeBranch* branch = 0;
      // just find the right ComputeBranch
      if (nodeCategory == "CompartmentVariables")
        branch = findBranch(nodeIndex, densityIndex, nodeType);
      else
      {
#ifdef MICRODOMAIN_CALCIUM
        std::tuple<int, int, std::string >& channelBranchIndexPair =
          _channelBranchIndices1[_channelLayers.size() - 1][densityIndex][0];
        branch = findBranch(
            nodeIndex, std::get<0>(channelBranchIndexPair),
            _compartmentVariableTypes[std::get<1>(channelBranchIndexPair)]);
#else
        std::pair<int, int>& channelBranchIndexPair =
          _channelBranchIndices1[_channelLayers.size() - 1][densityIndex][0];
        branch = findBranch(
            nodeIndex, channelBranchIndexPair.first,
            _compartmentVariableTypes[channelBranchIndexPair.second]);
#endif
      }
      assert(branch);

      //# capsules in that branch
      int ncaps = branch->_nCapsules;
      // Find: # compartments in the current branch
      // NOTE: cptsizes_in_branch holds the information about #capsules per cpt
      //   index=0 --> distal-end cpt
      //   index=ncpts-1 --> proximal-end cpt
      std::vector<int> cptsizes_in_branch;

#ifdef IDEA1
      int ncpts = _tissueContext->getNumCompartments(branch, cptsizes_in_branch);
#else
      //NUMCPT 1
      int ncpts = getNumCompartments(branch, cptsizes_in_branch);
#endif
      size.push_back(ncpts);
      // NOTE: The remainer capsules will be distributed every one to each
      // compartment from distal-end

      rval = ncpts;
      assert(branch->_parent || branch->_daughters.size() > 0);

      Capsule& firstcaps = branch->_capsules[0];
      // if (nodeCategory == "CompartmentVariables")
      if (nodeCategory == "CompartmentVariables" and
          // TUAN: this check 
          //       branch->_parent  (WRONG check)
          // is wrong, a branch starting of an MPI process
          // has no parent, but it is not a soma
          // TUAN: this is correct check
          _segmentDescriptor.getBranchType(firstcaps.getKey()) !=
          Branch::_SOMA)  // the branch is not a soma
      {
        DataItemArrayDataItem* dimArray = new DataItemArrayDataItem(size);
        ConstantType* ct = lc->sim->getConstantType("CompartmentDimension");
        std::auto_ptr<DataItem> aptr_cst;
        std::vector<CG_CompartmentDimension*>& dimensions =
          _tissueContext->_branchDimensionsMap[branch];
        if (dimensions.size() > 0)
          assert(dimensions.size() == ncpts);
        else  // no-data yet--> start putting information into
          // 'CompartmentDimension' vector
        {
          // revised the information of DimensionStruct
          //   to include (x,y,z, r, dist2soma, surface_area, volume)
          //   surface_area = sum(surface area of capsules)
          //       and takes into account
          //         (1) compartment next to soma (i.e. need to subtract the
          //         area of the capsule enclosed inside the soma sphere)
          //         (2) the compartment next to junction (i.e. half of the
          //         area of the capsule next to the junction)
          //
          ////TUAN :
          //# compartment should depend upon
          // if the branch faces
          //   1. explicit junction on proximal-side
          //       --> reserve some surface area
          //         (1) if parent is soma
          //         (2) if parent is cut/branch point
          //   3. implicit             proximal-side
          //        3.a implicit is just a cut point-->do nothing
          //        3.b implicit is a branching point--> reserve some surface
          //        area
          //   2. explicit junction on distal-side
          //       --> reserve some surface area
          //   4. implicit             distal-side
          //        4.a implicit cut point-->do nothing
          //        4.b implicit branching point -->
          //            (1) an additional CompartmentDimension is created as 1st
          //            cpt
          //            (2) reserve some surface area on the 2nd cpt
          // now
          std::vector<int>::const_iterator cibiter = cptsizes_in_branch.begin(),
            cibiend = cptsizes_in_branch.end();
          //NOTE: i = cpt index; j = capsule index
          for (int i = 0, j = ncaps - (*cibiter); i < ncpts, cibiter < cibiend;
              ++i, cibiter++, j -= (*cibiter))
          {  // compartment indexing is distal to proximal, while capsule
            // indexing is proximal to distal
            assert(j >= 0);
            Capsule* begCap = &branch->_capsules[j];
            Capsule* endCap = &branch->_capsules[j + (*cibiter) - 1];

            dyn_var_t radius = 0.0;
            dyn_var_t surface_area = 0.0;
            dyn_var_t volume = 0.0;
            dyn_var_t lost_distance_distal = 0.0;
            dyn_var_t lost_distance_proximal = 0.0;
#ifdef IDEA1
            float reserved4distend = branch->_numCapsulesEachSideForBranchPoint.second;
            float reserved4proxend = branch->_numCapsulesEachSideForBranchPoint.first;
            if (j == 0)
            {//proximal end
              //1. shift capPtr
              begCap += int(std::floor(reserved4proxend));
            }

            if (i == 0 && 
                branch->_daughters.size() >= 1)
            {//distal-end and the tree continue 
              //1. shift capPtr
              endCap -= int(std::floor(reserved4distend));
              //2. 
            }
#else
            if (j == 0 )
            {
              begCap = &branch->_capsules[j+getFirstIndexOfCapsuleSpanningSoma(branch)];
            }
#endif

            for (Capsule* capPtr = begCap; capPtr <= endCap; ++capPtr)
            {//find dimension information for this compartment by summing up the capsules
              radius += capPtr->getRadius();
              // A = pi/4 (D+d) * sqrt((D-d)^2 + 4h^2)
              //  D,d : diameters 2 ends
              //  h   : height
              // dyn_var_t d2 = d1 = 2* capPtr->getRadius();
              // dyn_var_t h = sqrt(SqDist(capPtr->getBeginCoordinates(),
              //                          capPtr->getEndCoordinates()));
              dyn_var_t h = capPtr->getLength();
              // surface_area += M_PI/4.0 * (d1+d2) sqrt(pow(d1-d2,2)+
              // 4*pow(h,2));
              // simplified form
              dyn_var_t r = capPtr->getRadius();
              // TUAN: TODO need to update to consider the case of
              // single-capsule single-compartment branch
              //  - we cannot put half to junction before and half to junction
              //  after --> no surface area left
              //  solution: put 1/4 to the junction before, 1/4 to junction
              //  after and half to the surface area
              surface_area += 2.0 * M_PI * r * h;
              volume += M_PI * r * r * h;

              dyn_var_t somaR = 0.0;          // soma radius (if present)
              //dyn_var_t lost_distance = 0.0;  // using lost_distance ensures no
                                              // negative surface_volume and
                                              // volume
              //lost_distance_distal = 0.0;
              //lost_distance_proximal = 0.0;
              if (j == 0 && capPtr == begCap)
              {  // check proximal-end
              //TUAN TODO NOW
                key_size_t key = capPtr->getKey();
                unsigned int computeOrder =
                  _segmentDescriptor.getComputeOrder(key);
                if (computeOrder == 0)
                {  // reserve some at proximal-end for the explicit junction
                  ComputeBranch* parentbranch = branch->_parent;
                  Capsule& firstcaps = parentbranch->_capsules[0];
#ifdef IDEA1
                    dyn_var_t frac = reserved4proxend - int(std::floor(reserved4proxend));
                    surface_area -= 2.0 * M_PI * r * h * frac;
                    volume -= M_PI * r * r * h * frac;
                    //lost_distance = h * frac;
                    lost_distance_proximal = h * frac;
#else
                  if (_segmentDescriptor.getBranchType(firstcaps.getKey()) ==
                      Branch::_SOMA)
                  {  // remove the part of the capsule covered inside soma sphere
                    //UPDATE: there is no longer need, as we already shift the capsule index
                    // as given above
                    // covered inside soma
                    //TUAN : disable it as we now ignore the physical location
                    //  of soma covering the dendrite
                    //  NOTE: we should not disable this as the soma typically large,
                    //  and if we add this it may hinder the proper signal propagation
                    //dyn_var_t h = capPtr->getLength();
                    //surface_area -= 2.0 * M_PI * r * h;
                    //volume -= M_PI * r * r * h;
                    //lost_distance = h ;
                    somaR = firstcaps.getRadius();
                    assert(capPtr->getDist2Soma()+capPtr->getLength()>somaR);
                    lost_distance_proximal = somaR - capPtr->getDist2Soma();
                    surface_area -= 2.0 * M_PI * r * lost_distance_proximal;
                    volume -= M_PI * r * r * lost_distance_proximal;
                    //if (h < somaR)
                    //{//TUAN DEBUG
                    //std::cerr << "First capsule to soma is too short\n";
                    //std::cerr << "length = " << h << "; while soma radius = " << somaR
                    //  << " from neuron index " <<
                    //  _segmentDescriptor.getNeuronIndex(key)
                    //  << std::endl;
                    //}
                    //assert(h > somaR);
                    assert(lost_distance_proximal >= 0.0);
                  }
                  else
                  {  // reserve proximal-end for the cut/branch explicit junction
                    dyn_var_t frac = getFractionCapsuleVolumeFromPost(branch);
                    lost_distance_proximal = h * frac;
                    surface_area -= 2.0 * M_PI * r * lost_distance_proximal;
                    volume -= M_PI * r * r * lost_distance_proximal;
                    //lost_distance = h * frac;
                    assert(lost_distance_proximal >= 0.0);
                  }
#endif
                }
                //assert(surface_area > 0);
                //assert(volume > 0);
              }
              if (i == 0 && capPtr == endCap)
              {  // check distal-end
              //TUAN TODO NOW
                key_size_t key = capPtr->getKey();
                unsigned int computeOrder =
                  _segmentDescriptor.getComputeOrder(key);
                if (computeOrder == MAX_COMPUTE_ORDER)
                {  // reserve some at distal-end for the explicit junction
                  if (branch->_daughters.size() >= 1)
                  {
#ifdef IDEA1
                    dyn_var_t frac = reserved4distend - int(std::floor(reserved4distend));
#else
                    dyn_var_t frac = getFractionCapsuleVolumeFromPre(branch);
#endif
                    //surface_area -= 2.0 * M_PI * r * (h - lost_distance) * frac;
                    //volume -= M_PI * r * r * (h - lost_distance) * frac;
                    //lost_distance_distal = (h-lost_distance_proximal) * frac;
                    lost_distance_distal = (h) * frac; //to make sure it consistent witht the amount transferred to the 'junction' 
                                            //as the junction has no idea about the fraction hold in proximal of this compartment
                    surface_area -= 2.0 * M_PI * r * lost_distance_distal;
                    volume -= M_PI * r * r * lost_distance_distal;
                  }
                }
#ifdef IDEA1
                 //NOTE: no compartment on implicit (slice-cut or branchpoint)
#else
                //else
                //{
                //  //TOFIX TUAN TODO: check if we create a compartment for the implicit 
                //  //branching point
                //  if (branch->_daughters.size() >= 1)
                //  {  // reserve some for the implicit branching junction
                //    dyn_var_t frac = getFractionCapsuleVolumeFromPre(branch);
                //    //surface_area -= 2.0 * M_PI * r * (h - lost_distance) * frac;
                //    //volume -= M_PI * r * r * (h - lost_distance) * frac;
                //    //lost_distance_distal = (h-lost_distance_proximal) * frac;
                //    lost_distance_distal = (h) * frac; //to make sure it consistent witht the amount transferred to the 'junction' 
                //                            //as the junction has no idea about the fraction hold in proximal of this compartment
                //    surface_area -= 2.0 * M_PI * r * lost_distance_distal;
                //    volume -= M_PI * r * r * lost_distance_distal;
                //  }
                //  else
                //  {  // do nothing for implicit cut junction or the terminal-end
                //    // capsule
                //  }
                //}
#endif
                //assert(surface_area > 0);
                //assert(volume > 0);
              }
            }
            assert(surface_area > 0);
            assert(volume > 0);
            radius /= ((endCap - begCap) + 1);  // still the average between the
                                                // first and last capsules
            //dyn_var_t dist2soma =
            //  begCap->getDist2Soma() +
            //  0.5 * (endCap->getDist2Soma() - begCap->getDist2Soma() -
            //      lost_distance);
            //      IMPORTANT: The first capsule stemming from soma
            //      then its 'dist2soma' is also the distance
            //      between its central point to the soma central point
            dyn_var_t length = endCap->getDist2Soma() + endCap->getLength() - 
                lost_distance_distal - (begCap->getDist2Soma() + lost_distance_proximal) ;
            dyn_var_t dist2soma = begCap->getDist2Soma() + lost_distance_proximal +
              0.5 * length;
            assert(dist2soma > 0.0);
            ComputeBranch* parentbranch = branch->_parent;
            Capsule& firstcaps = parentbranch->_capsules[0];
            if (_segmentDescriptor.getBranchType(firstcaps.getKey()) ==
                Branch::_SOMA)
            {
              assert(dist2soma > firstcaps.getRadius());
            }
#ifdef DEBUG_CPTS
            currentBranch = branch;
#endif
            // create DimensionStruct
            // //LOC 1
            // TRY to find the location of the center point
            //double center[3];
            StructDataItem* dimsDI = getDimension(
                lc, begCap->getBeginCoordinates(), endCap->getEndCoordinates(),
                radius, dist2soma, surface_area, volume, length);
            std::auto_ptr<DataItem> dimsDI_ap(dimsDI);
            NDPair* ndp = new NDPair("dimension", dimsDI_ap);

            NDPairList dimParams;
            dimParams.push_back(ndp);
            ct->getInstance(aptr_cst, dimParams, lc);
            ConstantDataItem* cdi =
              dynamic_cast<ConstantDataItem*>(aptr_cst.get());
            std::auto_ptr<Constant> aptr_dim;
            cdi->getConstant()->duplicate(aptr_dim);
            Constant* dim = aptr_dim.release();
            dimensions.push_back(dynamic_cast<CG_CompartmentDimension*>(dim));

            // currentcompartment_size = chemicalSynapseTouchSpace[i+1];
          }
        }
      }
    }
    else
    {  // only 1-compartment for JunctionChannels
      size.push_back(1);
    }

    const std::vector<DataItem*>* cpt = extractCompartmentalization(params);
    if (cpt == NULL)
    {
      std::cerr << "WARNING: Check if we need at least one argument (compartmentalize) at nodeType=" << nodeType << std::endl;
     // std::cerr << "ERROR: Expect at least one argument (compartmentalize) at nodeType=" << nodeType << std::endl;
     // assert(0);
    }
    else{
      // for parameters for the nodeType and are defined inside
      // 'compartmentalize='
      // arguments,
      std::vector<DataItem*>::const_iterator cptiter, cptend = cpt->end();
      //  ... make them an array of the same size as the #cpts in that
      //  ComputeBranch
      //   make the arrays of size = #-compartments in that ComputeBranch
      NDPairList::iterator ndpiter, ndpend = params->end();
      for (cptiter = cpt->begin(); cptiter != cptend; ++cptiter)
      {
        bool foundNDP = false;
        for (ndpiter = params->begin(); ndpiter != ndpend; ++ndpiter)
        {//for each data member
          std::string mydata = (*cptiter)->getString();//DEBUG purpose
          if ((*ndpiter)->getName() == (*cptiter)->getString())
          {//if the data member is part of 'compartmentalize' declaration
            //...adjust the size of the data member vector to the #cpts on that branch
            foundNDP = true;
            ArrayDataItem* arrayDI =
              dynamic_cast<ArrayDataItem*>((*ndpiter)->getDataItem());
            if (arrayDI == 0)
            {
              std::cerr << "TissueFunctor: " << *(*cptiter)
                << " comparmentalization can only be applied to an array "
                "parameter!" << std::endl;
              exit(-1);
            }
            arrayDI->setDimensions(size);
            break;
          }
        }
        if (!foundNDP)
        {
          ArrayDataItem* arrayDI = 0;
          std::string mystring = (*cptiter)->getString();
          std::vector<std::string> tokens;
          std::string delimiters = ":";
          StringUtils::Tokenize(mystring, tokens, delimiters);
          assert(tokens.size() == 1 || tokens.size() == 2);
          if (tokens.size() == 1 || tokens[0] == "float")
          {
            arrayDI = new FloatArrayDataItem(size);
          }
          else if (tokens[0] == "int")
          {
            arrayDI = new IntArrayDataItem(size);
          }
          else if (tokens[0] == "string")
          {
            arrayDI = new StringArrayDataItem(size);
          }
          assert(arrayDI);
          std::string varName = tokens[tokens.size() - 1];
          StringUtils::trim(varName);
          // std::cerr << "varName = " << varName << std::endl;

          std::auto_ptr<DataItem> arrayDI_ap(arrayDI);
          NDPair* ndp = new NDPair(varName, arrayDI_ap);
          params->push_back(ndp);
        }
      }
    }
  }
  else if (nodeCategory == "Junctions")
  {  // explicit junction (which can be
    //   1. soma
    //   2. explicit branching point
    //   3. or a slicing-cut junction
    //    (a cutpoint by 2 MPI processes if the previous ComputeBranch of
    //    computeOrder=MAX_COMPUTE_ORDER
    //   )
    Capsule* junctionCapsule = findJunction(nodeIndex, densityIndex, nodeType);
    std::map<Capsule*, CG_CompartmentDimension*>::iterator miter =
      _tissueContext->_junctionDimensionMap.find(junctionCapsule);
    // make sure the Layer of the given nodeType has not been declared
    // then
    // create the data structure for the explicit junction of the given nodeType
    // example of nodeType can be 'Voltage', 'Calcium', 'CalciumER'
    if (miter == _tissueContext->_junctionDimensionMap.end())
    {
      ConstantType* ct = lc->sim->getConstantType("CompartmentDimension");
      std::auto_ptr<DataItem> aptr_cst;
      ComputeBranch* branch_parent = junctionCapsule->getBranch();
      std::list<ComputeBranch*>::const_iterator
        iter = branch_parent->_daughters.begin(),
            iterend = branch_parent->_daughters.end();
      dyn_var_t h = junctionCapsule->getLength();
      dyn_var_t r = junctionCapsule->getRadius();
      dyn_var_t surface_area = 0.0;
      dyn_var_t volume = 0.0;
      dyn_var_t length = 0.0;
      // explicit junction can be
      //  1. soma junction
      //  2. slicing-cut junction (by the slicing plane split 2 MPI processes)
      //  3. branching-point junction
      if (_segmentDescriptor.getBranchType(junctionCapsule->getKey()) ==
          Branch::_SOMA)
      {  // soma explicit junction
        surface_area += 4.0 * M_PI * r * r;
        volume += 4.0 / 3.0 * M_PI * r * r * r;
        for (; iter != iterend; iter++)
        {  // subtract those covered by the steming axon/dendrite
          dyn_var_t r = (*iter)->_capsules[0].getRadius();
          surface_area -= M_PI * r * r;
        }
        if (surface_area <= 0)
        {
          std::cerr << "ERROR: The dendritic/axonal branch has size greater "
            "than the soma's radius" << std::endl;
          assert(surface_area > 0);
        }
        length = 2 * r;
  //#define TEST_IDEA_LONGER_SOMA
  //#ifdef TEST_IDEA_LONGER_SOMA
  //      length = 2.5 * r;
  //#endif
      }
      else
      {  // slicing-cut/branching-point explicit junction
        // 2.a first take proximal-side of junction
#ifdef IDEA1
        //which is the distal-end of the parent branch
        float reserved4distend = branch_parent->_numCapsulesEachSideForBranchPoint.second;
        Capsule* lastCapsule = &branch_parent->lastCapsule();
        Capsule* caps = lastCapsule;
        int ii = 1;
        for (; ii < reserved4distend; ii++)
        {
          dyn_var_t h = caps->getLength();
          dyn_var_t r = caps->getRadius();
          surface_area += 2.0 * M_PI * r * h;
          volume += M_PI * r * r * h;
          length += caps->getLength();
          caps = caps -1;
        }
        dyn_var_t h = caps->getLength();
        dyn_var_t r = caps->getRadius();
        dyn_var_t frac = reserved4distend-ii+1;
#else
        dyn_var_t frac = getFractionCapsuleVolumeFromPre(branch_parent);
#endif
        dyn_var_t part_reserved_on_parentBranch = h * frac;
        surface_area += 2.0 * M_PI * r * part_reserved_on_parentBranch;
        volume += M_PI * r * r * part_reserved_on_parentBranch;
        length += part_reserved_on_parentBranch;
        dyn_var_t sumlen = 0.0;
        //  2.b. then sum with the parts from the distal-side
        // NOTE: we have two scenarios:
        //    a. slicing-cut (1-child) branchpoint
        //    b. true-branching (many-children) branchpoint 
        //  but currently the two scenarios are treated the same
#ifdef IDEA1
        if (branch_parent->_daughters.size() >= 1)
        {// slicing cut point or branchpoint explicit junction
          for (; iter != iterend; iter++)
          {  
            ComputeBranch* childbranch = (*iter);
            Capsule* childCapsule = &childbranch->_capsules[0];
            //distal-side of branchpoint means the proximal-side of the child-branch
            float reserved4proxend = childbranch->_numCapsulesEachSideForBranchPoint.first;
            Capsule* caps = childCapsule;
            int ii = 1;
            for (ii = 1; ii < reserved4proxend; ii++)
            {
              dyn_var_t h = caps->getLength();
              dyn_var_t r = caps->getRadius();
              surface_area += 2.0 * M_PI * r * h;
              volume += M_PI * r * r * h;
              sumlen += caps->getLength();
              caps = caps +1;
            }
            dyn_var_t h = caps->getLength();
            dyn_var_t r = caps->getRadius();
            frac = reserved4proxend-ii+1;
            surface_area += 2.0 * M_PI * r * h * frac;
            volume += M_PI * r * r * h * frac;
            sumlen += frac * h;
          }

        }
#else
        //TUAN NOTE: using _daughters.size() == 1
        //is not a good criteria for slicing cut, as the user-defined branchType
        //can lead to a new branch, at which there is no Y-shape branching point
        //e.g. axon---AIS--axon
        //as for now, there is no different in treatment between slicing-cut cpt vs. branching cpt, so it is OK
        //to use this
        if (branch_parent->_daughters.size() == 1 
  //THINKING about this          
  //          and _segmentDescriptor.getComputeOrder(branch_parent->lastCapsule().getKey()) < MAX_COMPUTE_ORDER 
            )
        {  // slicing cut point explicit junction
          iter = branch_parent->_daughters.begin();
          ComputeBranch* childbranch = (*iter);
          Capsule* childCapsule = &childbranch->_capsules[0];
  //#ifdef IDEA1
  //        //distal-side of branchpoint means the proximal-side of the child-branch
  //        float reserved4proxend = childbranch->_numCapsulesEachSideForBranchPoint.first;
  //        Capsule* caps = childCapsule;
  //        int ii = 1;
  //        for (ii = 1; ii < reserved4proxend; ii++)
  //        {
  //          dyn_var_t h = caps->getLength();
  //          dyn_var_t r = caps->getRadius();
  //          surface_area += 2.0 * M_PI * r * h;
  //          volume += M_PI * r * r * h;
  //          sumlen += caps->getLength();
  //          caps = caps +1;
  //        }
  //        dyn_var_t h = caps->getLength();
  //        dyn_var_t r = caps->getRadius();
  //        frac = reserved4proxend-ii+1;
  //        surface_area += 2.0 * M_PI * r * h * frac;
  //        volume += M_PI * r * r * h * frac;
  //        sumlen += frac * h;
  //#else
          dyn_var_t h = childCapsule->getLength();
          dyn_var_t r = childCapsule->getRadius();
          frac = getFractionCapsuleVolumeFromPost(childbranch);
          dyn_var_t part_reserved_on_AChildBranch = h * frac;
          surface_area += 2.0 * M_PI * r * part_reserved_on_AChildBranch; 
          volume += M_PI * r * r * part_reserved_on_AChildBranch;
          sumlen += part_reserved_on_AChildBranch;
  //#endif
        }
        else
        {  // iterate all children branches: branching point explicit junction
          for (; iter != iterend; iter++)
          {  
  //#ifdef IDEA1
  //          ComputeBranch* childbranch = (*iter);
  //          Capsule* childCapsule = &childbranch->_capsules[0];
  //          //distal-side of branchpoint means the proximal-side of the child-branch
  //          float reserved4proxend = childbranch->_numCapsulesEachSideForBranchPoint.first;
  //          Capsule* caps = childCapsule;
  //          int ii = 1;
  //          for (ii = 1; ii < reserved4proxend; ii++)
  //          {
  //            dyn_var_t h = caps->getLength();
  //            dyn_var_t r = caps->getRadius();
  //            surface_area += 2.0 * M_PI * r * h;
  //            volume += M_PI * r * r * h;
  //            sumlen += caps->getLength();
  //            caps = caps +1;
  //          }
  //          dyn_var_t h = caps->getLength();
  //          dyn_var_t r = caps->getRadius();
  //          frac = reserved4proxend-ii+1;
  //          surface_area += 2.0 * M_PI * r * h * frac;
  //          volume += M_PI * r * r * h * frac;
  //          sumlen += frac * h;
  //#else
  //// take half surface area from the first capsule
            dyn_var_t h = (*iter)->_capsules[0].getLength();
            dyn_var_t r = (*iter)->_capsules[0].getRadius();
            frac = getFractionCapsuleVolumeFromPost((*iter));
            dyn_var_t part_reserved_on_AChildBranch = h * frac;
            surface_area += 2.0 * M_PI * r * part_reserved_on_AChildBranch;
            volume += M_PI * r * r * part_reserved_on_AChildBranch;
            sumlen += part_reserved_on_AChildBranch;
  //#endif
          }
        }
#endif
        length += (sumlen / branch_parent->_daughters.size());
      }
      // create DimensionStruct for explicit 'junction' single-compartment
      // branch
      dyn_var_t dist2soma ;
      if (_segmentDescriptor.getBranchType(junctionCapsule->getKey()) ==
          Branch::_SOMA)
      {
         dist2soma = 0.0;
      }
      else
      {
        dist2soma =
          junctionCapsule->getDist2Soma() + junctionCapsule->getLength();
        assert(dist2soma>0.0);
      }
#ifdef DEBUG_CPTS
      currentBranch = branch_parent;
#endif
      //LOC 2 - explicit junction
      StructDataItem* dimsDI =
        getDimension(lc, junctionCapsule->getEndCoordinates(),
            (dyn_var_t)junctionCapsule->getRadius(),
            (dyn_var_t)dist2soma, surface_area, volume, length);
      std::auto_ptr<DataItem> dimsDI_ap(dimsDI);
      NDPair* ndp = new NDPair("dimension", dimsDI_ap);
      NDPairList dimParams;
      dimParams.push_back(ndp);
      ct->getInstance(aptr_cst, dimParams, lc);
      ConstantDataItem* cdi = dynamic_cast<ConstantDataItem*>(aptr_cst.get());
      std::auto_ptr<Constant> aptr_dim;
      cdi->getConstant()->duplicate(aptr_dim);
      Constant* dim = aptr_dim.release();
      _tissueContext->_junctionDimensionMap[junctionCapsule] =
        dynamic_cast<CG_CompartmentDimension*>(dim);
    }
  }
  return rval;
}

// GOAL: During NodeInit statement
//   return the list of data members of type array
// PURPOSE: in the next step, they will be initialized
//        to a certain size, e.g. the size = the #-compartments per
//        ComputeBranch
// E.g.:
//			InitNodes ( .[].Layer(branches),
// tissueFunctor("NodeInit",
//<
// compartmentalize = {
//   "Vnew",
//   "Vcur",
//   "Aii",
//   "Aim",
//   "Aip",
//   "RHS",
//},
// Vnew={Vrest_value}
//>
//)
std::vector<DataItem*> const* TissueFunctor::extractCompartmentalization(
    NDPairList* params)
{
  const std::vector<DataItem*>* cpt = NULL;
  NDPairList::iterator ndpiter, ndpend = params->end();
  for (ndpiter = params->begin(); ndpiter != ndpend; ++ndpiter)
  {
    if ((*ndpiter)->getName() == "compartmentalize")
    {   
      DataItemArrayDataItem* cptDI =
          dynamic_cast<DataItemArrayDataItem*>((*ndpiter)->getDataItem());
      if (cptDI == 0)
      {
        std::cerr << "TissueFunctor: compartmentalization parameter is not a "
                     "list of parameter names!" << std::endl;
        exit(-1);
      }
      cpt = cptDI->getDataItemVector();
      params->erase(ndpiter);
      break;
    }
  }
  return cpt;
}

// GOAL: create a new DimensionStruct object
//   which holds information for 1 compartment
//   the information includes (x,y,z,r,dist2soma, surface_area, volume)
//  given
//    cds = (x,y,z) centroid point
//    radius = r (averaged radius over capsules of same cpt)
//    dist2soma = fiber-along distance
//            (use the distance from the first capsule in the compartment)
//    surface_area : sum from all capsules
//      NOTE: If a ComputeBranch has only 1 capsule, and MAX_COMPUTE_ORDER=0
//              then surface_area_lost2junction= 1/4 surface of that capsule
//              and will be used for both sides, i.e.
//              the surface_area = 1/2 of that capsule surface area
//    HISTORY:
//      1.1: use (x,y,z,r,dist2soma, surface_area, volume, length)
//      1.0: use (x,y,z,r,dist2soma)
StructDataItem* TissueFunctor::getDimension(LensContext* lc, double* cds,
                                            dyn_var_t radius,
                                            dyn_var_t dist2soma,
                                            dyn_var_t surface_area,
                                            dyn_var_t volume, dyn_var_t length)
{
  assert(radius > 0);
  assert(surface_area > 0);
  assert(dist2soma >= 0);
  assert(volume > 0);
  assert(length > 0);

  //TUAN TODO PLAN: Use FloatDataItem is better (for coordinate)
  DoubleDataItem* xddi = new DoubleDataItem(cds[0]);
  std::auto_ptr<DataItem> xddi_ap(xddi);
  NDPair* x = new NDPair("x", xddi_ap);

  DoubleDataItem* yddi = new DoubleDataItem(cds[1]);
  std::auto_ptr<DataItem> yddi_ap(yddi);
  NDPair* y = new NDPair("y", yddi_ap);

  DoubleDataItem* zddi = new DoubleDataItem(cds[2]);
  std::auto_ptr<DataItem> zddi_ap(zddi);
  NDPair* z = new NDPair("z", zddi_ap);

  DoubleDataItem* rddi = new DoubleDataItem(radius);
  std::auto_ptr<DataItem> rddi_ap(rddi);
  NDPair* r = new NDPair("r", rddi_ap);

  DoubleDataItem* d2sddi = new DoubleDataItem(dist2soma);
  std::auto_ptr<DataItem> d2sddi_ap(d2sddi);
  NDPair* d2s = new NDPair("dist2soma", d2sddi_ap);

  DoubleDataItem* areaddi = new DoubleDataItem(surface_area);
  std::auto_ptr<DataItem> areaddi_ap(areaddi);
  NDPair* area = new NDPair("surface_area", areaddi_ap);

  DoubleDataItem* volumeddi = new DoubleDataItem(volume);
  std::auto_ptr<DataItem> volumeddi_ap(volumeddi);
  NDPair* volumeptr = new NDPair("volume", volumeddi_ap);

  DoubleDataItem* lengthddi = new DoubleDataItem(length);
  std::auto_ptr<DataItem> lengthddi_ap(lengthddi);
  NDPair* lengthptr = new NDPair("length", lengthddi_ap);

  NDPairList dimList;
  dimList.push_back(x);
  dimList.push_back(y);
  dimList.push_back(z);

  dimList.push_back(r);
  dimList.push_back(d2s);
  dimList.push_back(area);
  dimList.push_back(volumeptr);

  dimList.push_back(lengthptr);
#ifdef DEBUG_CPTS
  std::cerr << "Dimension [(x,y,z,r),dist2soma, surface_area, volume, length] = "
            << "[ (" << cds[0] << "," << cds[1] << "," << cds[2] << ","
            << radius << ")," << dist2soma << "," << surface_area << ","
            << volume << "," << length << "]" << std::endl;
  //cpt_surfaceArea.push_back(surface_area);
  //cpt_volume.push_back(volume);
  // save the status for compartment in each branch
  Capsule& firstcaps = currentBranch->_capsules[0];
  int brType = _segmentDescriptor.getBranchType(firstcaps.getKey());
  if (brType ==  Branch::_AXON || 
      brType == Branch::_AIS ||
      brType == Branch::_AXONHILLLOCK)  // the branch is not a soma
  {
    cpt_surfaceArea.push_back(std::pair<int,float>(Branch::_AXON,surface_area));
    cpt_volume.push_back(std::pair<int,float>(Branch::_AXON,volume));
    cpt_length.push_back(std::pair<int,float>(Branch::_AXON,length));
  }else if (brType == Branch::_BASALDEN) 
  {
    cpt_surfaceArea.push_back(std::pair<int,float>(Branch::_BASALDEN,surface_area));
    cpt_volume.push_back(std::pair<int,float>(Branch::_BASALDEN,volume));
    cpt_length.push_back(std::pair<int,float>(Branch::_BASALDEN,length));
  }
  else if (brType == Branch::_APICALDEN ||
      brType == Branch::_TUFTEDDEN)
  {
    cpt_surfaceArea.push_back(std::pair<int,float>(Branch::_APICALDEN,surface_area));
    cpt_volume.push_back(std::pair<int,float>(Branch::_APICALDEN,volume));
    cpt_length.push_back(std::pair<int,float>(Branch::_APICALDEN,length));
  }
#endif
  StructType* st = lc->sim->getStructType("DimensionStruct");
  std::auto_ptr<Struct> dims;
  st->getStruct(dims);
  dims->initialize(dimList);
  StructDataItem* dimsDI = new StructDataItem(dims);
  return dimsDI;
}

// GOAL: return the data structure representing the center point between two
// points
//
//  INPUT: cds1 = begin coord (1st point)
//         cds2 = end coord (2nd point)
//         radius = the common radius between 2 points
//         dist2soma = the dist2soma from cds1
//
StructDataItem* TissueFunctor::getDimension(LensContext* lc, double* cds1,
                                            double* cds2, dyn_var_t radius,
                                            dyn_var_t dist2soma,
                                            dyn_var_t surface_area,
                                            dyn_var_t volume, 
                                            dyn_var_t length)
{
  double center[3];
  double dsqrd = 0;
  //TUAN TODO: may need revision for the center point
  //when a compartment is multi-capsules
  for (int i = 0; i < 3; ++i)
  {
    center[i] = (cds1[i] + cds2[i]) / 2.0;
    double d = (cds1[i] - cds2[i]);
    dsqrd += d * d;
  }
  // dist2soma += 0.5 * sqrt(dsqrd);
  //dyn_var_t length = sqrt(dsqrd);
  //dist2soma += 0.5 * length;
  return getDimension(lc, center, radius, dist2soma, surface_area, volume,
                      length);
}

//GOAL: 
//suppose the NDPairList ndpl has one NDpair as:
//  nodekind=someCategory[name1][name2][name3]
//return: a vector holding (somecategory, name1, name2, name3)
// using '[' and ']' as separators
void TissueFunctor::getNodekind(const NDPairList* ndpl,
                                std::vector<std::string>& nodekind)
{
  nodekind.clear();
  assert(ndpl);
  NDPairList::const_iterator ndpiter, ndpend = ndpl->end();
  for (ndpiter = ndpl->begin(); ndpiter != ndpend; ++ndpiter)
  {
    if ((*ndpiter)->getName() == "nodekind")
    {
      StringDataItem* nkDI =
          dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
      if (nkDI == 0)
      {
        std::cerr << "TissueFunctor: nodekind parameter is not a string!"
                  << std::endl;
        exit(-1);
      }
      std::string kind = nkDI->getString();
      /*
  char* ckind = new char[kind.size() + 1];
  strcpy(ckind, kind.c_str());

  char* p = strtok(ckind, "][");
  while (p != 0) {
    nodekind.push_back(std::string(p));
    p = strtok(0, "][");
  }
      */
      std::vector<std::string> tokens;
      StringUtils::Tokenize(kind, tokens, "][");
      for (std::vector<std::string>::iterator i = tokens.begin();
           i != tokens.end(); ++i)
      {
        nodekind.push_back(*i);
      }

      break;
    }
  }
}

// GOAL: return the ComputeBranch of the branch
//   nodeType     = string of nodetype (e.g. Calcium (if BRANCH), Nat (if JUNCTION))
//   nodeIndex = grid's node
//   densityIndex = index in that grid's node
//   <"nodeType", <"nodeIndex", < density-index, ComputeBranch*>
ComputeBranch* TissueFunctor::findBranch(int nodeIndex, int densityIndex,
                                         std::string const& nodeType)
{
  ComputeBranch* rval = 0;
  std::map<std::string,
           std::map<int, std::map<int, ComputeBranch*> > >::iterator mapiter1 =
      _indexBranchMap.find(nodeType);
  if (mapiter1 == _indexBranchMap.end())
  {
    std::cerr << "Tissue Functor::findBranch, branch node type " << nodeType
              << " not found in Branch Index Map! rank=" << _rank << std::endl;
    exit(EXIT_FAILURE);
  }
  std::map<int, std::map<int, ComputeBranch*> >::iterator mapiter2 =
      mapiter1->second.find(nodeIndex);
  if (mapiter2 == mapiter1->second.end())
  {
    std::cerr << "Tissue Functor::findBranch, branch index not found in Branch "
                 "Index Map! rank=" << _rank
              << ", nodeIndex (failed)=" << nodeIndex
              << ", densityIndex=" << densityIndex << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  std::map<int, ComputeBranch*>::iterator mapiter3 =
      mapiter2->second.find(densityIndex);
  if (mapiter3 == mapiter2->second.end())
  {
    std::cerr << "Tissue Functor::findBranch, branch density index not found "
                 "in Branch Map! rank=" << _rank << ", nodeIndex=" << nodeIndex
              << ", densityIndex=" << densityIndex << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  rval = mapiter3->second;
  return rval;
}

// GOAL:
//   given nodeType (e.g. Calcium) and a given ComputeBranch 'b'
//   return the vector of 2 elements 
//   {gridnode-index, //which is MPI-rank
//    index-of-element-in-that-gridnode //which
//   }
std::vector<int>& TissueFunctor::findBranchIndices(ComputeBranch* b,
                                                   std::string const& nodeType)
{
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >::iterator
      mapiter1 = _branchIndexMap.find(nodeType);
  if (mapiter1 == _branchIndexMap.end())
  {
    // if the NodeType as defined in the Param file
    // but not being defined in the GSL file
    std::cerr << "Tissue Functor::findBranchIndices, branch node type "
              << nodeType << " not found in Branch Index Map! rank=" << _rank
              << std::endl;
    std::cerr << "HINTS: Maybe the nodeType " << nodeType
              << " is not defined in COMPARTMENT_VARIABLE_TARGET section for BRANCHTYPE " 
              << _segmentDescriptor.getBranchType(b->_capsules[0].getKey()) + 1
              << " in CptParams file"
              << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  std::map<ComputeBranch*, std::vector<int> >::iterator mapiter2 =
      mapiter1->second.find(b);
  if (mapiter2 == mapiter1->second.end())
  {
    std::cerr << "Tissue Functor::findBranchIndices, branch indices not found "
                 "in Branch Index Map! rank=" << _rank << std::endl;
    std::cerr << ".. from node type " << nodeType << std::endl;
    std::cerr << "HINTS: Maybe the channel associated with a given branch, and "
                 "depends on the nodeType" << nodeType << "; but\n"
              << " this nodeType is not associated with that branch. Check "
                 "COMPARTMENT_VARIABLE_TARGETS in CptParams files" << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  return mapiter2->second;
}

// GOAL: return the last capsule of the ComputeBranch
// (computeOrder=MAX_COMPUTE_ORDER)
//         of proximal-side of the explicit junction
//      based on the given node-name (i.e. nodeType)
//      and the specific index in the grid (i.e. nodeIndex)
//      as well as the specific index in the density of that grid-index (i.e.
//      densityIndex)
//      nodeType --> e.g. Junctions[Voltage] layer
//      nodeIndex --> grid's node's index
//      densityIndex --> index of the explicit junction for given nodeType in
//      given nodeIndex
// TODO: update the order to nodeType, nodeIndex, densityIndex
Capsule* TissueFunctor::findJunction(int nodeIndex, int densityIndex,
                                     std::string const& nodeType)
{
  std::map<std::string, std::map<int, std::map<int, Capsule*> > >::iterator
      mapiter1 = _indexJunctionMap.find(nodeType);
  if (mapiter1 == _indexJunctionMap.end())
  {
    std::cerr << "Tissue Functor::findJunction, junction node type " << nodeType
              << " not found in Junction Map! rank=" << _rank << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  std::map<int, std::map<int, Capsule*> >::iterator mapiter2 =
      mapiter1->second.find(nodeIndex);
  if (mapiter2 == mapiter1->second.end())
  {
    std::cerr << "Tissue Functor::findJunction, junction index not found in "
                 "Junction Map! rank=" << _rank << ", nodeIndex=" << nodeIndex
              << ", densityIndex=" << densityIndex << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  std::map<int, Capsule*>::iterator mapiter3 =
      mapiter2->second.find(densityIndex);
  if (mapiter3 == mapiter2->second.end())
  {
    std::cerr << "Tissue Functor::findJunction, junction density index not "
                 "found in Junction Map! rank=" << _rank
              << ", nodeIndex=" << nodeIndex
              << ", densityIndex=" << densityIndex << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  return mapiter3->second;
}

// GOAL: return a two-element vector [index-junction-layer,
// index-density-of-junction]
//  that tell the MPI-rank, and the index in the vector holding Junction Compartment
//   based on
//      nodeType --> e.g. Junctions[Voltage] layer
std::vector<int>& TissueFunctor::findJunctionIndices(
    Capsule* c, std::string const& nodeType)
{
  std::map<std::string, std::map<Capsule*, std::vector<int> > >::iterator
      mapiter1 = _junctionIndexMap.find(nodeType);
  if (mapiter1 == _junctionIndexMap.end())
  {
    std::cerr << "Tissue Functor::findJunctionIndices, junction type not found "
                 "in Junction Index Map! rank=" << _rank 
                 << ", nodeType=" << nodeType << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  std::map<Capsule*, std::vector<int> >::iterator mapiter2 =
      mapiter1->second.find(c);
  if (mapiter2 == mapiter1->second.end())
  {
    std::cerr << "Tissue Functor::findJunction, junction not found in Junction "
                 "Index Map! rank=" << _rank 
                 << ", nodeType=" << nodeType << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  return mapiter2->second;
}

std::vector<int>& TissueFunctor::findForwardSolvePointIndices(
    ComputeBranch* b, std::string& nodeType)
{
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >::iterator
      mapiter1 = _branchForwardSolvePointIndexMap.find(nodeType);
  if (mapiter1 == _branchForwardSolvePointIndexMap.end())
  {
    std::cerr << "Tissue Functor: forward solve point node type " << nodeType
              << " not found in Forward Solve Point Index Map! rank=" << _rank
              << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  std::map<ComputeBranch*, std::vector<int> >::iterator mapiter2 =
      mapiter1->second.find(b);
  if (mapiter2 == mapiter1->second.end())
  {
    std::cerr << "Tissue Functor: forward solve point index not found in "
                 "Forward Solve Point Index Map! rank=" << _rank << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  return mapiter2->second;
}

std::vector<int>& TissueFunctor::findBackwardSolvePointIndices(
    ComputeBranch* b, std::string& nodeType)
{
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >::iterator
      mapiter1 = _branchBackwardSolvePointIndexMap.find(nodeType);
  if (mapiter1 == _branchBackwardSolvePointIndexMap.end())
  {
    std::cerr << "Tissue Functor: backward solve point node type " << nodeType
              << " not found in Branch Backward Solve Point Index Map! rank="
              << _rank << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  std::map<ComputeBranch*, std::vector<int> >::iterator mapiter2 =
      mapiter1->second.find(b);
  if (mapiter2 == mapiter1->second.end())
  {
    std::cerr << "Tissue Functor: backward solve point index not found in "
                 "Branch Backward Solve Point Index Map! rank=" << _rank
              << std::endl;
    assert(0);
    exit(EXIT_FAILURE);
  }
  return mapiter2->second;
}

// GOAL: this method performs the connection from any nodetype
//    (channel, connexon, junction)
//    to another nodetype
//    for a given connector based on the given InAttrPset in NDPairList 'ndpl'
void TissueFunctor::connect(Simulation* sim, Connector* connector,
                            NodeDescriptor* from, NodeDescriptor* to,
                            NDPairList& ndpl)
{
  std::auto_ptr<ParameterSet> inAttrPSet, outAttrPSet;
  to->getGridLayerDescriptor()->getNodeType()->getInAttrParameterSet(
      inAttrPSet);
  from->getGridLayerDescriptor()->getNodeType()->getOutAttrParameterSet(
      outAttrPSet);
  inAttrPSet->set(ndpl);
  connector->nodeToNode(from, outAttrPSet.get(), to, inAttrPSet.get(), sim);
}

std::auto_ptr<Functor> TissueFunctor::userExecute(LensContext* CG_c,
                                                  String& tissueElement,
                                                  NDPairList*& params)
{
  params->duplicate(_params);
  std::auto_ptr<Functor> rval;

  if (tissueElement == "Layout")
  {
    NDPairList* ndpl=_params.get();
    assert(ndpl);
    assert(ndpl->size()>0);
    NDPair* ndp=ndpl->back();
    if (ndp->getName()!="PROBED") {
      //normal NTS "Layout"
      TissueElement* element = dynamic_cast<TissueElement*>(_layoutFunctor.get());
      element->setTissueFunctor(this);
      _layoutFunctor->duplicate(rval);
    }
    else {
      //special "Layout" for being used in Zipper
      //do 
      //    1. configure _probedLayoutsMap 
      //    2. call _layoutFunctor which will derive density information
      //        from _probedLayoutsMap 
      //NOTE: A Layout has a unique name given via 'PROBED=name'
      // it basically do layout inside probe (which store Grid*, NodeDescriptor*)
      // and the Layout to find the probed Grid; 
      std::vector<NodeDescriptor*> nodeDescriptors;
      Grid* grid = doProbe(CG_c, nodeDescriptors);

      TissueElement* element=dynamic_cast<TissueElement*>(_layoutFunctor.get());
      element->setTissueFunctor(this);
      _layoutFunctor->duplicate(rval);
    }
  }
  else if (tissueElement == "NodeInit")
  {
    TissueElement* element =
        dynamic_cast<TissueElement*>(_nodeInitFunctor.get());
    element->setTissueFunctor(this);
    _nodeInitFunctor->duplicate(rval);
  }
  else if (tissueElement == "Connector")
  {
    TissueElement* element =
        dynamic_cast<TissueElement*>(_connectorFunctor.get());
    element->setTissueFunctor(this);
    _connectorFunctor->duplicate(rval);
  }
  else if (tissueElement == "Probe")
  {
    TissueElement* element = dynamic_cast<TissueElement*>(_probeFunctor.get());
    element->setTissueFunctor(this);
    _probeFunctor->duplicate(rval);
  }
  else if (tissueElement=="MGSify") 
  {
    assert(0); // not support
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
  else if (tissueElement=="Tissue->MGS") 
  {
    assert(0); //not support
    doMGSify(CG_c);
  }
  else {
    std::cerr<<"Unrecognized tissue element specifier: "<<tissueElement<<std::endl;
    exit(1);
  }
  return rval;
}

// GOAL : calcualte the density vector (which is local for each gridnode
//     and each gridnode is mapped to 1 MPI process)
//    It tells how many instance for the NodeType associated with the given layer
//    to be created at each grid-location
//    Shallow<int> rval;
//    with rval.size() == _nbrGridNodes
//       rval.assign(_nbrGridNodes, 0);
ShallowArray<int> TissueFunctor::doLayout(LensContext* lc)
{
  NDPairList* ndpl=_params.get();
  NDPair* ndp=ndpl->back();
  assert(ndpl);
  assert(ndpl->size()>0);

  // rval[gridnode-index] = how-many-capsules-in-that-gridnode
  ShallowArray<int> rval;

  if (ndp->getName()=="PROBED") {
    rval = doLayoutHybrid(lc);
  }
  else
  {//"CATEGORY"
    rval = doLayoutNTS(lc);
  }
  return rval;
}

ShallowArray<int> TissueFunctor::doLayoutHybrid(LensContext* lc)
{
  NDPairList* ndpl=_params.get();
  NDPair* ndp=ndpl->back();
  std::map<std::string, ShallowArray<int> >::iterator 
    miter=_probedLayoutsMap.end();

  ShallowArray<int> rval;
  assert(ndp->getName()=="PROBED");

  if (ndp->getName()=="PROBED") {
    StringDataItem* prDI = dynamic_cast<StringDataItem*>(ndp->getDataItem());
    if (prDI == 0) {
      std::cerr<<"TissueFunctor: probed parameter is not a string!"<<std::endl;
      exit(-1);
    }
    miter=_probedLayoutsMap.find(prDI->getString());
    if (miter==_probedLayoutsMap.end()) {
      std::cerr<<"PROBED identifier not recognized on Layout!"<<std::endl;
      exit(-1);
    }
    rval=miter->second;
  }
  return rval;
}

// NOTE:
//  Example: Layer (..., <nodekind="SynapticClefts[Voltage][2]">, ...)
//  then
//   nodeCategory = "SynapticClefts"
//   nodeType  = "Voltage"
//   nodeComputeOrder = 2
ShallowArray<int> TissueFunctor::doLayoutNTS(LensContext* lc)
{
  assert(_params.get());
  std::vector<std::string> nodekind;
  getNodekind(_params.get(), nodekind);
  assert(nodekind.size() > 0);
  std::string& nodeCategory = nodekind[0];
  std::string nodeType = "";
  int nodeComputeOrder = -1;
  if (nodekind.size() > 1) nodeType = nodekind[1];
  if (nodekind.size() > 2) nodeComputeOrder = atoi(nodekind[2].c_str());
  if (nodeCategory != "CompartmentVariables" && nodeCategory != "Junctions" &&
      nodeCategory != "EndPoints" && nodeCategory != "JunctionPoints" &&
      nodeCategory != "ForwardSolvePoints" &&
      nodeCategory != "BackwardSolvePoints" && nodeCategory != "Channels" &&
      nodeCategory != "ElectricalSynapses" &&
      nodeCategory != "BidirectionalConnections" &&
      nodeCategory != "ChemicalSynapses" &&
      nodeCategory != "PreSynapticPoints" && nodeCategory != "SynapticClefts")
  {  // validation purpose
    std::cerr << "Unrecognized nodeCategory parameter on Layer : "
              << nodeCategory << std::endl;
    exit(EXIT_FAILURE);
  }

  if (nodeCategory == "Channels")
  {  // create tracker mapping channel to branch and explicit junction
#ifdef MICRODOMAIN_CALCIUM
    _channelBranchIndices1.push_back(
        std::vector<std::vector<std::tuple<int, int, std::string > > >());
    _channelJunctionIndices1.push_back(
        std::vector<std::vector<std::tuple<int, int, std::string > > >());
    _channelBranchIndices2.push_back(
        std::vector<std::vector<std::tuple<int, int, std::string > > >());
    _channelJunctionIndices2.push_back(
        std::vector<std::vector<std::tuple<int, int, std::string > > >());
#else
    _channelBranchIndices1.push_back(
        std::vector<std::vector<std::pair<int, int> > >());
    _channelJunctionIndices1.push_back(
        std::vector<std::vector<std::pair<int, int> > >());
    _channelBranchIndices2.push_back(
        std::vector<std::vector<std::pair<int, int> > >());
    _channelJunctionIndices2.push_back(
        std::vector<std::vector<std::pair<int, int> > >());
#endif
    if (_channelTypesMap.find(nodeType) != _channelTypesMap.end())
    {
      std::cerr << "Unrecognized " << nodeType << " on nodeCategory : "
                << nodeCategory << std::endl;
    }
    assert(_channelTypesMap.find(nodeType) == _channelTypesMap.end());
    _channelTypesMap[nodeType] = _channelTypeCounter;
  }

  if (nodeCategory == "CompartmentVariables")
  {  // create tracker of different compartment variables
    assert(_compartmentVariableTypesMap.find(nodeType) ==
           _compartmentVariableTypesMap.end());
    assert(_compartmentVariableTypesMap.size() ==
           _compartmentVariableTypeCounter);
    _compartmentVariableTypesMap[nodeType] = _compartmentVariableTypeCounter;
    _compartmentVariableTypes.push_back(nodeType);
  }

  if (nodeCategory == "Junctions")
  {
    assert(_junctionTypesMap.find(nodeType) == _junctionTypesMap.end());
    _junctionTypesMap[nodeType] = _junctionTypeCounter;
  }

  if (nodeCategory == "EndPoints")
  {
    assert(_endPointTypesMap.find(nodeType) == _endPointTypesMap.end());
    _endPointTypesMap[nodeType] = _endPointTypeCounter;
  }

  if (nodeCategory == "JunctionPoints")
  {
    assert(_junctionPointTypesMap.find(nodeType) ==
           _junctionPointTypesMap.end());
    _junctionPointTypesMap[nodeType] = _junctionPointTypeCounter;
  }

  if (nodeCategory == "ForwardSolvePoints")
  {
    assert(_forwardSolvePointTypesMap.find(nodeComputeOrder) ==
               _forwardSolvePointTypesMap.end() ||
           _forwardSolvePointTypesMap[nodeComputeOrder].find(nodeType) ==
               _forwardSolvePointTypesMap[nodeComputeOrder].end());
    _forwardSolvePointTypesMap[nodeComputeOrder][nodeType] =
        _forwardSolvePointTypeCounter;
  }

  if (nodeCategory == "BackwardSolvePoints")
  {
    assert(_backwardSolvePointTypesMap.find(nodeComputeOrder) ==
               _backwardSolvePointTypesMap.end() ||
           _backwardSolvePointTypesMap[nodeComputeOrder].find(nodeType) ==
               _backwardSolvePointTypesMap[nodeComputeOrder].end());
    _backwardSolvePointTypesMap[nodeComputeOrder][nodeType] =
        _backwardSolvePointTypeCounter;
  }

  /* this to
   make sure also the bi-directional connection is defined in SynParams.par
   (this is for ElectricalSynapse (Branch-Branch connection) and
   SpineNeck-Branch connection)
   with SpineNeck is treated as a apical-dendrite branch on a two-compartment
   neuron,
   with the
   SpineHead is treated as the soma on that two-compartment neuron
   */
  bool electrical = (nodeCategory == "ElectricalSynapses" &&
                     _tissueParams.electricalSynapses());
  /* bidirectional connection = electrical + chemical couplin between spine-neck
   *   and dendritic shaft
   */
  bool bidirectional = (nodeCategory == "BidirectionalConnections" &&
                        _tissueParams.bidirectionalConnections());
  /* Chemical synapse is the SpineHead (soma) - Bouton (axon of another neuron)
   * connection
   */
  bool chemical =
      (nodeCategory == "ChemicalSynapses" && _tissueParams.chemicalSynapses());
  // bool point =
  //    (nodeCategory == "PreSynapticPoints" &&
  //    _tissueParams.chemicalSynapses());
  bool point = ((nodeCategory == "PreSynapticPoints" ||
                 nodeCategory == "SynapticClefts") &&
                _tissueParams.chemicalSynapses());

  Grid* grid = lc->layerContext->grid;
  if (_nbrGridNodes == 0)
    _nbrGridNodes = grid->getNbrGridNodes();
  else if (_nbrGridNodes != grid->getNbrGridNodes())
  {
    std::cerr << "Error, number of grid nodes has changed! " << _nbrGridNodes
              << "!=" << grid->getNbrGridNodes() << std::endl;
    assert(0);
  }
  // rval[gridnode-index] = how-many-capsules-in-that-gridnode
  ShallowArray<int> rval;
  rval.assign(_nbrGridNodes, 0);

  int counter = 0;
  // create tracker for Layers associated with ...
  if (electrical)
  {
    _electricalSynapseTypesMap[nodeType] = counter =
        _electricalSynapseTypeCounter;
  }
  else if (bidirectional)
  {
    _bidirectionalConnectionTypesMap[nodeType] = counter =
        _bidirectionalConnectionTypeCounter;
  }
  else if (chemical)
  {
    _chemicalSynapseTypesMap[nodeType] = counter = _chemicalSynapseTypeCounter;
     assert(_chemicalSynapseTypeCounter == _synapseReceptorMaps.size());
    _synapseReceptorMaps.push_back(std::map<Touch*, int>());
  }
  else if (point)
  {
    if (nodeCategory == "PreSynapticPoints")
    {
      assert(_preSynapticPointTypesMap.find(nodeType) ==
             _preSynapticPointTypesMap.end());
      _preSynapticPointTypesMap[nodeType] = counter =
          _preSynapticPointTypeCounter;
    }
    if (nodeCategory == "SynapticClefts")
    {
      assert(_synapticCleftTypesMap.find(nodeType) ==
             _synapticCleftTypesMap.end());
      _synapticCleftTypesMap[nodeType] = counter = _synapticCleftTypeCounter;
      assert(_synapticCleftTypeCounter == _synapticCleftMaps.size());
      _synapticCleftMaps.push_back(std::map<Touch*, int>());
    }
  }

  // if a Layer of a nodetype as electrical synapse (gap junction), 
  // bidirectional (spineattachment), chemical synapse, or point (synapticcleft)
  //  i.e. not compartment variables
  //  then traverse the touchVectors to determine if an instance of such nodetype 
  //  should be created   (through the increase of rval[grid-index])
  //   (and if created, create-on the MPI rank having pre-side Capsule)
  //  which is later can be used by ::doConnector() to establish connection
  //  NOTE: the symbol '|' represents MPI border if present
  //     cpt<--[connexon]-->'|' cpt
  //     cpt<--[spineconnexon]-->'|'<--[spineconnexon]-->cpt
  //     bouton --[presynapticpoint]-->'|' one-or-many-ChemicalSynapses-receptors-on-spine-head
  //                                ?(pass Vpre) 
  // IDEA: bouton --[synapticCleft]-->'|' one-or-many-ChemicalSynapses-receptors-on-spine-head
  //                              ?(pass [NT])
  // ORDER:
  //      preCpt -> synapticCleftNode  (always on the same rank)
  //      postCpt -> synapticCleftNode (can be different rank)
  //      synapticCleftNode -> receptor (can be different rank)
  //      postCpt -> receptor
  //      receptor -> postCpt 
  //      receptorA -> receptorB (two receptors on the same synapse
  //                             for plasticity purpose, i.e. mixed-synapse)
  //  Here, one or two instances may needed be created using one or two sides of a Touch
  //  then find the preCapsule and postCapsule for each Touch
  //  identify the grid-index (indexPre) to which preCapsule belongs
  //           the grid-index (indexPost) to which postCapsule belongs
  //  and decide if an instance for the NodeType to be created or not at indexPre
  //         and/or indexPost
  if (electrical || chemical || point || bidirectional)
  {
    // REMEBER that all the touches (within the current MPI process) have been detected,
    // we just need to find out which compartment (from which branch) connect
    // to which compartment (in which branch)
    // and use the branches (as well as other keyfields, e.g. MTYPE) to
    // connect
    // the input/output to the receptors/channels
#ifdef IDEA1
    int nn = 0;
    int mm = 0;
    int bb = 0;
#endif

    TouchVector::TouchIterator titer = _tissueContext->_touchVector.begin(),
                               tend = _tissueContext->_touchVector.end();

    for (; titer != tend; ++titer)
    {//traverse all the recorded Touch(es)
      //Check to make sure only consider the touch with at least 
      //one capsule supposed to be handled
      //by the current MPI process
      //assert(_tissueContext->isLensTouch(*titer, _rank));
#ifdef IDEA1
      ++mm;
      //if (!_tissueContext->isLensTouch(*titer, _rank)) continue;
      if (!_tissueContext->isLensTouch(*titer, _rank)) 
      {
        if (bidirectional)
        {
          for (int ii = 0; ii < _size; ii++)
          {
            //MPI_Barrier(MPI_COMM_WORLD);
            if (ii == _rank)
            {
              std::cout << "rank " << _rank << " with touch-- " << 
                _segmentDescriptor.getLongKey(titer->getKey1()) << "," << 
                _segmentDescriptor.getLongKey(titer->getKey2()) << " to be handled in rank \n " ;
              std::cout << "neuron index : " << _segmentDescriptor.getNeuronIndex(titer->getKey1())  << ", " << _segmentDescriptor.getNeuronIndex(titer->getKey2()) << std::endl;
            }

          }
        }
       continue; 
      }
      ++nn;
#else
      if (!_tissueContext->isLensTouch(*titer, _rank))  continue;
#endif

      key_size_t key1, key2;
      key1 = titer->getKey1();
      key2 = titer->getKey2();
      Capsule* preCapsule =
          &_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key1)];
      Capsule* postCapsule =
          &_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key2)];
      ComputeBranch* postBranch = postCapsule->getBranch();
      assert(postBranch);

      int indexPre, indexPost;  // GSL Grid's index at which preCapsule
      // and postCapsule belongs to, respectively
      bool preJunction = false;
      bool postJunction = false;

#ifdef IDEA1
      /* NOTE: if a touch belong to an explicit junction
       * we need to keep track of the capsule at which the junction is associated with
       */
      Capsule* jctCapsulePreCapsule = preCapsule;
      Capsule* jctCapsulePostCapsule = postCapsule;
      if (_tissueContext->isPartOfExplicitJunction(*preCapsule, *titer, indexPre, &jctCapsulePreCapsule))
      {
          if (point &&
                  _capsuleJctPointIndexMap[nodeType].find(jctCapsulePreCapsule) !=
                  _capsuleJctPointIndexMap[nodeType].end())
              continue;
          //if (jctCapsulePreCapsule == NULL)
          //{
          //    assert(indexPre != _rank);
          //    continue;
          //}
          //if (indexPre != _rank)
          //  continue;
          preJunction = true;
      }
#else
      if (_segmentDescriptor.getFlag(key1) &&
          _tissueContext->isTouchToEnd(*preCapsule, *titer))
      {  // pre component is LENS junction (i.e. explicit junction)
#ifdef SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE
        if (point &&
            _capsuleJctPointIndexMap[nodeType].find(std::make_pair(preCapsule, postCapsule)) !=
                _capsuleJctPointIndexMap[nodeType].end())
          continue;
#else
        if (point &&
            _capsuleJctPointIndexMap[nodeType].find(preCapsule) !=
                _capsuleJctPointIndexMap[nodeType].end())
          continue;
#endif
        preJunction = true;
        indexPre = _tissueContext->getRankOfEndPoint(preCapsule->getBranch());
      }
#endif
      else
      {  // pre component is LENS branch
        if (point &&
            _capsuleCptPointIndexMap[nodeType].find(preCapsule) !=
                _capsuleCptPointIndexMap[nodeType].end())
          continue;
        preJunction = false;
        indexPre = _tissueContext->getRankOfBeginPoint(preCapsule->getBranch());
      }

      std::vector<double> probabilities;
      // assign the probability for the touch to be connected
      if (point)  // bouton-side?
      {
        std::list<std::string>& synapseTypes =
            _tissueParams.getPreSynapticPointSynapseTypes(nodeType);
        std::list<std::string>::iterator synIter, synEnd = synapseTypes.end();
        for (synIter = synapseTypes.begin(); synIter != synEnd; ++synIter)
        {
          if (isPointRequired(titer, *synIter))
          {
            probabilities.push_back(1.0);
            break;
          }
          if (probabilities.size() > 0) break;
        }
      }
      else if (electrical)
        getElectricalSynapseProbabilities(probabilities, titer, nodeType);
      else if (chemical)
        getChemicalSynapseProbabilities(probabilities, titer, nodeType);
      else if (bidirectional)
        getBidirectionalConnectionProbabilities(probabilities, titer, nodeType);
      else
        assert(0);  // end assign probability

      for (int i = 0; i < probabilities.size(); ++i)
      {  // each structure (electrical, chemical, bidirectional) that is based on Touch, 
		  // such structure (e.g. chemical synapse)
		  // has a series of probabilities (each prob. for one type (e.g. DenSpine))
        if (probabilities[i] > 0)
        {
#ifdef IDEA1
          if (_tissueContext->isPartOfExplicitJunction(*postCapsule, *titer, indexPost, &jctCapsulePostCapsule))
          {
            //if (jctCapsulePostCapsule == NULL)
            //{
            //    assert(indexPost != _rank);
            //    continue;
            //}
            //if (indexPost != _rank)
            //  continue;
            postJunction = true;
          }
#else
          if (_tissueContext->isTouchToEnd(*postCapsule, *titer) &&
              _segmentDescriptor.getFlag(postCapsule->getKey()))
          {
            // post component is LENS junction
            postJunction = true;
            indexPost = _tissueContext->getRankOfEndPoint(postBranch);
            Sphere postEndSphere;
            postCapsule->getEndSphere(postEndSphere);
            // assert(indexPost==_tissueContext->_decomposition->getRank(postEndSphere));
          }
#endif
          else
          {
            // post component is LENS branch
            postJunction = false;
            indexPost = _tissueContext->getRankOfBeginPoint(postBranch);
            // assert(indexPost==_tissueContext->_decomposition->getRank(postCapsule->getSphere()));
          }
          assert(indexPre==_rank || indexPost==_rank);

          // as it loops through all touches, it only handle the touches
          // that are designed for the current MPI process, i.e.
          // based on the indexPre and indexPost
          if (indexPre == _rank || indexPost == _rank)
          {
            if (point)
            {
              // NEW CODE: constraint the number of PreSynapticPoint or
              // SynapticCleft instances
              // to be instantiated to the exact number of chemical synapses
              bool result = setGenerated(_generatedSynapticClefts, titer,
                                         counter, i, nodeCategory);
              if (result)
              {
#ifdef IDEA1
                if (preJunction)
                  _capsuleJctPointIndexMap[nodeType][jctCapsulePreCapsule] =
                      rval[indexPre];
                else
                  _capsuleCptPointIndexMap[nodeType][preCapsule] =
                      rval[indexPre];
#else
                if (preJunction)
                {
#ifdef SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE
                  _capsuleJctPointIndexMap[nodeType][std::make_pair(preCapsule, postCapsule)] =
                      rval[indexPre];
#else
                  _capsuleJctPointIndexMap[nodeType][preCapsule] =
                      rval[indexPre];
#endif

                }
                else
                  _capsuleCptPointIndexMap[nodeType][preCapsule] =
                      rval[indexPre];
#endif
                if (nodeCategory == "SynapticClefts")
                {
                  if (indexPre == _rank)
                    _synapticCleftMaps[counter][&*titer] = rval[_rank];
                }
                rval[indexPre]++;
              }
            }
            else
            {
              if (electrical)
              {
                if (probabilities[i] >=
                    drandom(findSynapseGenerator(indexPre, indexPost)))
                {
                  rval[indexPre]++;
                  rval[indexPost]++;
                  setGenerated(_generatedElectricalSynapses, titer, counter, i);
                }
              }
              if (bidirectional)
              {  // TUAN TODO: plan to make this as part of generating the chemical
                 // synapse below
                // in such cases, we don't need an explicit statement
                // for 'BidirectionalConnections[DenSpine]' in the GSL
                // instead, a new node for that should be created and the
                // density should be increase for

                // TUAN TODO NOTE: for the given touch, find the spine neuron of
                // the spineneck
                //    then check if the soma forms the chemical synapse or not
                //    if yes, then create the spineattachment
#ifdef IDEA1
                ++bb;
#endif
                if (probabilities[i] >=
                    drandom(findSynapseGenerator(indexPre, indexPost)))
                {
                  // NOTE: rval control the number of instances to be created
                  // bool result =
                  // setGenerated(_generatedBidirectionalConnections,
                  //                           titer, counter, i, nodeCategory);
                  bool result = setGenerated(_generatedBidirectionalConnections,
                                             titer, counter, i);
                  // bool result;
                  // if (indexPre == indexPost)
                  // result = setGenerated(_generatedBidirectionalConnections,
                  //                       titer, counter, i, nodeCategory);
                  // else result =
                  // setGenerated(_generatedBidirectionalConnections,
                  //                           titer, counter, i );
                  if (result)
                  {
                    rval[indexPre]++;
                    rval[indexPost]++;
                  }
                }
              }
              else if (chemical)
              {
                if (touchIsChemicalSynapse(_generatedSynapticClefts, titer))
                {
                  //NOTE: It's ok to create SynapticCleft, but does not form 
                  //a true chemical synapse as there is no receptor there to sense the signal
                  if (probabilities[i] >=
                      drandom(findSynapseGenerator(indexPre, indexPost)))
                  {
                    if (indexPost == _rank)
                      _synapseReceptorMaps[counter][&*titer] = rval[_rank];
                    //NOTE: Assume postCapsule is always postsynaptic side (SynParam.par)
                    //for receptors, create on the postSynaptic side
                    rval[indexPost]++;
                    setGenerated(_generatedChemicalSynapses, titer, counter, i,
                                 nodeCategory);
                  }
                  else
                  {
                    setNonGenerated(titer, nodeType, i);
                  }
                }
                else
                  setNonGenerated(titer, nodeType, i);
              }
            }
          }
        }
      }
    }
#ifdef IDEA1
    if (bidirectional)
    {
    std::cout << "nn"  << (_rank) << " = " << nn << std::endl;
    std::cout << "mm"  << (_rank) << " = " << mm << std::endl;
    std::cout << "bb"  << (_rank) << " = " << bb << std::endl;

    }
#endif
  }

  std::map<unsigned int, std::vector<ComputeBranch*> >::iterator mapIter,
      mapEnd = _tissueContext->_neurons.end();
  // as the layer depends on the neuron
  // for each layer, we ...
  for (mapIter = _tissueContext->_neurons.begin(); mapIter != mapEnd; ++mapIter)
  {  // traverse through all neurons (i.e. via ComputeBranch )
    std::vector<ComputeBranch*>& branches = mapIter->second;
    std::vector<ComputeBranch*>::iterator iter, end = branches.end();
    for (iter = branches.begin(); iter != end; ++iter)
    {  // traverse through all ComputeBranches in that neuron
#ifdef IDEA1
        _tissueContext->getNumCompartments(*iter);
#endif
      Capsule* branchCapsules = (*iter)->_capsules;
      int nCapsules = (*iter)->_nCapsules;

      volatile unsigned int index, indexJct;

      key_size_t key = branchCapsules[0].getKey();
      // find out if the first capsule in that branch that match the key-mask
      // defined for that channel in parameter file
      if (nodeCategory == "Channels" ||
          // or if the compartment variable being layout is the target of any
          // channel
          // on that Compartment Computebranch (NOTE: not Junction ComputeBranch)
          _tissueParams.isCompartmentVariableTarget(key, nodeType))
      {  //  - if YES, then
        unsigned int computeOrder = _segmentDescriptor.getComputeOrder(key);
        unsigned int branchOrder = _segmentDescriptor.getBranchOrder(key);

        index = _tissueContext->getRankOfBeginPoint(*iter);
        indexJct = _tissueContext->getRankOfEndPoint(*iter);
#ifdef IDEA1
        //NO need to add code here
#endif
        bool channelTarget = false;
        if (nodeCategory == "Channels")
          channelTarget = isChannelTarget(key, nodeType);
        if (branchOrder != 0 &&
            (nodeCategory == "CompartmentVariables" ||
             (nodeCategory == "EndPoints" && (*iter)->_parent &&
              computeOrder == 0) ||
             (nodeCategory == "ForwardSolvePoints" && (*iter)->_parent &&
              computeOrder == nodeComputeOrder) ||
             (channelTarget && index == _rank)))
        {
          if (nodeCategory == "CompartmentVariables")
          {
            _indexBranchMap[nodeType][index][rval[index]] = (*iter);
            std::vector<int> indices;
            indices.push_back(index);
            indices.push_back(rval[index]);
            _branchIndexMap[nodeType][(*iter)] = indices;
            rval[index]++;
          }
          else if (nodeCategory == "EndPoints")
            rval[index]++;
          else if (nodeCategory == "ForwardSolvePoints")
          {
            std::vector<int> indices;
            indices.push_back(index);
            indices.push_back(rval[index]);
            _branchForwardSolvePointIndexMap[nodeType][*iter] = indices;
            rval[index]++;
          }
          else if (channelTarget)
          {
            std::list<Params::ChannelTarget>* targets =
                _tissueParams.getChannelTargets(key);
            if (targets)
            {
              std::list<Params::ChannelTarget>::iterator iiter =
                                                             targets->begin(),
                                                         iend = targets->end();
              for (; iiter != iend; ++iiter)
              {
                if (iiter->_type == nodeType)
                {
                  rval[index]++;
#ifdef MICRODOMAIN_CALCIUM
                  std::vector<std::tuple<int, int, std::string> > targetVector;
#else
                  std::vector<std::pair<int, int> > targetVector;
#endif
                  std::list<std::string>::iterator viter,
                      vend = iiter->_target1.end();
                  assert(iiter->_target1.size() > 0); //channel must receive 'input', e.g. Na [input] [output]
                  for (viter = iiter->_target1.begin(); viter != vend; ++viter)
                  {
#ifdef MICRODOMAIN_CALCIUM
                    //NOTE: As the name may contains domain names
                    // Calcium(domain1, domainA)
                    // we need to split them
                    std::string compartmentNameWithOptionalMicrodomainName(*viter);
                    std::string compartmentNameOnly("");
                    std::string microdomainName("");
                    Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                    checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                    if (not microdomainName.empty())
                      _microdomainOnBranch[*iter].insert(microdomainName);
                    //find out a Channel-branch associate input connection 
                    // to what (can be many )compartment-branch (e.g. Voltage, Calcium)
                    //  --> _channelBranchIndices1
                    std::vector<int>& branchIndices =
                        findBranchIndices(*iter, compartmentNameOnly);
                    assert(branchIndices[0] == _rank);
                    targetVector.push_back(std::tuple<int, int, std::string>(
                        branchIndices[1],
                        _compartmentVariableTypesMap[compartmentNameOnly],
                        microdomainName));
#else
                    std::vector<int>& branchIndices =
                        findBranchIndices(*iter, *viter);
                    assert(branchIndices[0] == _rank);
                    targetVector.push_back(std::pair<int, int>(
                        branchIndices[1],
                        _compartmentVariableTypesMap[*viter]));
#endif
                  }
                  _channelBranchIndices1[_channelTypeCounter].push_back(
                      targetVector);

                  targetVector.clear();
                  vend = iiter->_target2.end();
                  assert(iiter->_target2.size() > 0);
                  for (viter = iiter->_target2.begin(); viter != vend; ++viter)
                  {
#ifdef MICRODOMAIN_CALCIUM
                    //find out a Channel-branch associate output data
                    // to what (can be many )compartment-branch (e.g. Voltage, Calcium)
                    //  --> _channelBranchIndices2
                    std::string compartmentNameWithOptionalMicrodomainName(*viter);
                    std::string compartmentNameOnly("");
                    std::string microdomainName("");
                    Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                    checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                    if (not microdomainName.empty())
                      _microdomainOnBranch[*iter].insert(microdomainName);
                    std::vector<int>& branchIndices =
                        findBranchIndices(*iter, compartmentNameOnly);
                    assert(branchIndices[0] == _rank);
                    targetVector.push_back(std::tuple<int, int, std::string>(
                        branchIndices[1],
                        _compartmentVariableTypesMap[compartmentNameOnly],
                        microdomainName));
#else
                    std::vector<int>& branchIndices =
                        findBranchIndices(*iter, *viter);
                    assert(branchIndices[0] == _rank);
                    targetVector.push_back(std::pair<int, int>(
                        branchIndices[1],
                        _compartmentVariableTypesMap[*viter]));
#endif
                  }
                  _channelBranchIndices2[_channelTypeCounter].push_back(
                      targetVector);
                }
              }
            }
          }
        }

        if (nodeCategory != "CompartmentVariables" &&
            nodeCategory != "ForwardSolvePoints")
        {//Junctions, JunctionPoints, EndPoints, ...
#ifdef IDEA1
            {
                //NO need to add code here
                //if (computeOrder == 0)
                //{
                //    indexJct = _tissueContext->getRankOfBeginPoint(*iter);
                //    if (nodeCategory == "EndPoints" && branchOrder != 0)
                //    {
                //        //rval[index]++;
                //    }
                //    else
                //    {
                //        if (nodeCategory == "Junctions")
                //        {  // Junction compartment that resides on the same MPI processs
                //            // of the ComputeBranch under investigation
                //            _indexJunctionMap[nodeType][indexJct][rval[indexJct]] =
                //                &((*iter)->_parent->lastCapsule());  // lastcapsule in the branch
                //            // (*iter)
                //            std::vector<int> indices;
                //            indices.push_back(indexJct);
                //            indices.push_back(rval[indexJct]);
                //            _junctionIndexMap[nodeType][&((*iter)->_parent->lastCapsule())] =
                //                indices;
                //            rval[indexJct]++;
                //        }
                //        else if (nodeCategory == "JunctionPoints")
                //            rval[indexJct]++;
                //        else if (channelTarget && indexJct == _rank)
                //        {
                //            std::list<Params::ChannelTarget>* targets =
                //                _tissueParams.getChannelTargets(key);
                //            if (targets)
                //            {
                //                std::list<Params::ChannelTarget>::iterator
                //                    iiter = targets->begin(),
                //                          iend = targets->end();
                //                for (; iiter != iend; ++iiter)
                //                {
                //                    if (iiter->_type == nodeType)
                //                    {
                //                        rval[indexJct]++;
                //                        std::vector<std::pair<int, int> > targetVector;
                //                        std::list<std::string>::iterator viter,
                //                            vend = iiter->_target1.end();
                //                        assert(iiter->_target1.size() > 0);
                //                        for (viter = iiter->_target1.begin(); viter != vend;
                //                                ++viter)
                //                        {
                //                            std::map<std::string,
                //                                std::map<Capsule*, std::vector<int> > >::
                //                                    iterator jmapiter1 =
                //                                    _junctionIndexMap.find(*viter);
                //                            std::map<Capsule*, std::vector<int> >::iterator
                //                                jmapiter2;
                //                            if (jmapiter1 != _junctionIndexMap.end() &&
                //                                    (jmapiter2 = jmapiter1->second.find(
                //                                                                        &(*iter)->_parent->lastCapsule())) !=
                //                                    jmapiter1->second.end())
                //                            {
                //                                std::vector<int>& junctionIndices =
                //                                    jmapiter2->second;
                //                                targetVector.push_back(std::pair<int, int>(
                //                                            junctionIndices[1],
                //                                            _compartmentVariableTypesMap[*viter]));
                //                            }
                //                        }
                //                        _channelJunctionIndices1[_channelTypeCounter].push_back(
                //                                targetVector);
                //                        targetVector.clear();
                //                        vend = iiter->_target2.end();
                //                        assert(iiter->_target2.size() > 0);
                //                        for (viter = iiter->_target2.begin(); viter != vend;
                //                                ++viter)
                //                        {
                //                            std::map<std::string,
                //                                std::map<Capsule*, std::vector<int> > >::
                //                                    iterator jmapiter1 =
                //                                    _junctionIndexMap.find(*viter);
                //                            std::map<Capsule*, std::vector<int> >::iterator
                //                                jmapiter2;
                //                            if (jmapiter1 != _junctionIndexMap.end() &&
                //                                    (jmapiter2 = jmapiter1->second.find(
                //                                                                        &(*iter)->_parent->lastCapsule())) !=
                //                                    jmapiter1->second.end())
                //                            {
                //                                std::vector<int>& junctionIndices =
                //                                    jmapiter2->second;
                //                                targetVector.push_back(std::pair<int, int>(
                //                                            junctionIndices[1],
                //                                            _compartmentVariableTypesMap[*viter]));
                //                            }
                //                        }
                //                        _channelJunctionIndices2[_channelTypeCounter].push_back(
                //                                targetVector);
                //                    }
                //                }
                //            }
                //        }
                //    }
                //}
                ////else if (nodeCategory == "BackwardSolvePoints" &&
                ////        computeOrder == nodeComputeOrder)
                ////{
                ////    std::vector<int> indices;
                ////    indices.push_back(index);
                ////    indices.push_back(rval[index]);
                ////    _branchBackwardSolvePointIndexMap[nodeType][*iter] = indices;
                ////    rval[index]++;
                ////}

            }
          if ((*iter)->_daughters.size() > 0)
          {
            if (computeOrder == MAX_COMPUTE_ORDER)
            {
              assert(
                  _segmentDescriptor.getFlag((*iter)->lastCapsule().getKey()));
              if (nodeCategory == "EndPoints" && branchOrder != 0)
                rval[index]++;
              else
              {
                if (nodeCategory == "Junctions")
                {  // Junction compartment that resides on the same MPI processs
                   // of the ComputeBranch under investigation
                  _indexJunctionMap[nodeType][indexJct][rval[indexJct]] =
                      &((*iter)->lastCapsule());  // lastcapsule in the branch
                                                  // (*iter)
                  std::vector<int> indices;
                  indices.push_back(indexJct);
                  indices.push_back(rval[indexJct]);
                  _junctionIndexMap[nodeType][&((*iter)->lastCapsule())] =
                      indices;
                  rval[indexJct]++;
                }
                else if (nodeCategory == "JunctionPoints")
                  rval[indexJct]++;
                else if (channelTarget && indexJct == _rank)
                {
                  std::list<Params::ChannelTarget>* targets =
                      _tissueParams.getChannelTargets(key);
                  if (targets)
                  {
                    std::list<Params::ChannelTarget>::iterator
                        iiter = targets->begin(),
                        iend = targets->end();
                    for (; iiter != iend; ++iiter)
                    {
                      if (iiter->_type == nodeType)
                      {
                        rval[indexJct]++;
#ifdef MICRODOMAIN_CALCIUM
                        std::vector<std::tuple<int, int, std::string> > targetVector;
#else
                        std::vector<std::pair<int, int> > targetVector;
#endif
                        std::list<std::string>::iterator viter,
                            vend = iiter->_target1.end();
                        assert(iiter->_target1.size() > 0);
                        for (viter = iiter->_target1.begin(); viter != vend;
                             ++viter)
                        {
#ifdef MICRODOMAIN_CALCIUM
                          //NOTE: This is inside IDEA1
                          //find out a Channel-branch associate output data
                          // to what (can be many )junction-branch (e.g. Voltage, Calcium)
                          //  --> _channelJunctionIndices1
                          std::string compartmentNameWithOptionalMicrodomainName(*viter);
                          std::string compartmentNameOnly("");
                          std::string microdomainName("");
                          Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                          checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                          if (not microdomainName.empty())
                            _microdomainOnJunction[&((*iter)->lastCapsule())].insert(microdomainName);
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(compartmentNameOnly);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::tuple<int, int, std:string>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[compartmentNameOnly],
                                microdomainName));
                          }
#else
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(*viter);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::pair<int, int>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[*viter]));
                          }
#endif
                        }
                        _channelJunctionIndices1[_channelTypeCounter].push_back(
                            targetVector);
                        targetVector.clear();
                        vend = iiter->_target2.end();
                        assert(iiter->_target2.size() > 0);
                        for (viter = iiter->_target2.begin(); viter != vend;
                             ++viter)
                        {
#ifdef MICRODOMAIN_CALCIUM
                          //NOTE: This is inside IDEA1
                          //find out a Channel-branch associate output data
                          // to what (can be many )junction-branch (e.g. Voltage, Calcium)
                          //  --> _channelJunctionIndices2
                          std::string compartmentNameWithOptionalMicrodomainName(*viter);
                          std::string compartmentNameOnly("");
                          std::string microdomainName("");
                          Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                          checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                          if (not microdomainName.empty())
                            _microdomainOnJunction[&((*iter)->lastCapsule())].insert(microdomainName);
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(compartmentNameOnly);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::tuple<int, int, std::string>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[compartmentNameOnly],
                                microdomainName));
                          }
#else
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(*viter);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::pair<int, int>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[*viter]));
                          }
#endif
                        }
                        _channelJunctionIndices2[_channelTypeCounter].push_back(
                            targetVector);
                      }
                    }
                  }
                }
              }
            }
            else if (nodeCategory == "BackwardSolvePoints" &&
                     computeOrder == nodeComputeOrder)
            {
              std::vector<int> indices;
              indices.push_back(index);
              indices.push_back(rval[index]);
              _branchBackwardSolvePointIndexMap[nodeType][*iter] = indices;
              rval[index]++;
            }
          }
          else if ((nodeCategory == "Junctions" ||
                    nodeCategory == "JunctionPoints") &&
                   _segmentDescriptor.getFlag((*iter)->lastCapsule().getKey()))
          {  // Junction compartment or JunctionPoint node that resides on a
             // different MPI processs
             // of the ComputeBranch under investigation
            if (nodeCategory == "Junctions")
            {
              if (_segmentDescriptor.getBranchType((*iter)->lastCapsule().getKey()) !=
                  Branch::_SOMA)  // not soma - to handle soma-only model
                assert(indexJct != _rank);
              _indexJunctionMap[nodeType][indexJct][rval[indexJct]] =
                  &((*iter)->lastCapsule());
              std::vector<int> indices;
              indices.push_back(indexJct);
              indices.push_back(rval[indexJct]);
              _junctionIndexMap[nodeType][&((*iter)->lastCapsule())] = indices;
            }
            rval[indexJct]++;
          }
#else
          if ((*iter)->_daughters.size() > 0)
          {
            if (computeOrder == MAX_COMPUTE_ORDER)
            {
              assert(
                  _segmentDescriptor.getFlag((*iter)->lastCapsule().getKey()));
              if (nodeCategory == "EndPoints" && branchOrder != 0)
                rval[index]++;
              else
              {
                if (nodeCategory == "Junctions")
                {  // Junction compartment that resides on the same MPI processs
                   // of the ComputeBranch under investigation
                  _indexJunctionMap[nodeType][indexJct][rval[indexJct]] =
                      &((*iter)->lastCapsule());  // lastcapsule in the branch
                                                  // (*iter)
                  std::vector<int> indices;
                  indices.push_back(indexJct);
                  indices.push_back(rval[indexJct]);
                  _junctionIndexMap[nodeType][&((*iter)->lastCapsule())] =
                      indices;
                  rval[indexJct]++;
                }
                else if (nodeCategory == "JunctionPoints")
                  rval[indexJct]++;
                else if (channelTarget && indexJct == _rank)
                {
                  std::list<Params::ChannelTarget>* targets =
                      _tissueParams.getChannelTargets(key);
                  if (targets)
                  {
                    std::list<Params::ChannelTarget>::iterator
                        iiter = targets->begin(),
                        iend = targets->end();
                    for (; iiter != iend; ++iiter)
                    {
                      if (iiter->_type == nodeType)
                      {
                        rval[indexJct]++;
#ifdef MICRODOMAIN_CALCIUM
                        std::vector<std::tuple<int, int, std::string> > targetVector;
#else
                        std::vector<std::pair<int, int> > targetVector;
#endif
                        std::list<std::string>::iterator viter,
                            vend = iiter->_target1.end();
                        assert(iiter->_target1.size() > 0);
                        for (viter = iiter->_target1.begin(); viter != vend;
                             ++viter)
                        {
#ifdef MICRODOMAIN_CALCIUM
                          //find out a Channel-branch associate input connection 
                          // to what (can be many )compartment-branch (e.g. Voltage, Calcium)
                          //  --> _channelJunctionIndices1
                          std::string compartmentNameWithOptionalMicrodomainName(*viter);
                          std::string compartmentNameOnly("");
                          std::string microdomainName("");
                          Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                          checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                          if (not microdomainName.empty())
                            _microdomainOnJunction[&((*iter)->lastCapsule())].insert(microdomainName);
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(compartmentNameOnly);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::tuple<int, int, std::string>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[compartmentNameOnly],
                                microdomainName));
                          }
#else
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(*viter);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::pair<int, int>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[*viter]));
                          }
#endif
                        }
                        _channelJunctionIndices1[_channelTypeCounter].push_back(
                            targetVector);
                        targetVector.clear();
                        vend = iiter->_target2.end();
                        assert(iiter->_target2.size() > 0);
                        for (viter = iiter->_target2.begin(); viter != vend;
                             ++viter)
                        {
#ifdef MICRODOMAIN_CALCIUM
                          //find out a Channel-branch associate output data
                          // to what (can be many )junction-branch (e.g. Voltage, Calcium)
                          //  --> _channelJunctionIndices2
                          std::string compartmentNameWithOptionalMicrodomainName(*viter);
                          std::string compartmentNameOnly("");
                          std::string microdomainName("");
                          Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                          checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                          if (not microdomainName.empty())
                            _microdomainOnJunction[&((*iter)->lastCapsule())].insert(microdomainName);
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(compartmentNameOnly);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::tuple<int, int, std::string>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[compartmentNameOnly],
                                microdomainName));
                          }
#else
                          std::map<std::string,
                                   std::map<Capsule*, std::vector<int> > >::
                              iterator jmapiter1 =
                                  _junctionIndexMap.find(*viter);
                          std::map<Capsule*, std::vector<int> >::iterator
                              jmapiter2;
                          if (jmapiter1 != _junctionIndexMap.end() &&
                              (jmapiter2 = jmapiter1->second.find(
                                   &(*iter)->lastCapsule())) !=
                                  jmapiter1->second.end())
                          {
                            std::vector<int>& junctionIndices =
                                jmapiter2->second;
                            targetVector.push_back(std::pair<int, int>(
                                junctionIndices[1],
                                _compartmentVariableTypesMap[*viter]));
                          }
#endif
                        }
                        _channelJunctionIndices2[_channelTypeCounter].push_back(
                            targetVector);
                      }
                    }
                  }
                }
              }
            }
            else if (nodeCategory == "BackwardSolvePoints" &&
                     computeOrder == nodeComputeOrder)
            {
              std::vector<int> indices;
              indices.push_back(index);
              indices.push_back(rval[index]);
              _branchBackwardSolvePointIndexMap[nodeType][*iter] = indices;
              rval[index]++;
            }
          }
          else if ((nodeCategory == "Junctions" ||
                    nodeCategory == "JunctionPoints") &&
                   _segmentDescriptor.getFlag((*iter)->lastCapsule().getKey()))
          {  // Junction compartment or JunctionPoint node that resides on a
             // different MPI processs
             // of the ComputeBranch under investigation
            if (nodeCategory == "Junctions")
            {
              if (_segmentDescriptor.getBranchType((*iter)->lastCapsule().getKey()) !=
                  Branch::_SOMA)  // not soma - to handle soma-only model
                assert(indexJct != _rank);
              _indexJunctionMap[nodeType][indexJct][rval[indexJct]] =
                  &((*iter)->lastCapsule());
              std::vector<int> indices;
              indices.push_back(indexJct);
              indices.push_back(rval[indexJct]);
              _junctionIndexMap[nodeType][&((*iter)->lastCapsule())] = indices;
            }
            rval[indexJct]++;
          }
          else if ((nodeCategory == "Channels") && 
            (_segmentDescriptor.getBranchType((*iter)->lastCapsule().getKey()) ==
                  Branch::_SOMA))
          {//handle some-only model
            if (channelTarget && indexJct == _rank)
            {
              std::list<Params::ChannelTarget>* targets =
                _tissueParams.getChannelTargets(key);
              if (targets)
              {
                std::list<Params::ChannelTarget>::iterator
                  iiter = targets->begin(),
                        iend = targets->end();
                for (; iiter != iend; ++iiter)
                {
                  if (iiter->_type == nodeType)
                  {
                    rval[indexJct]++;
#ifdef MICRODOMAIN_CALCIUM
                    std::vector<std::tuple<int, int, std::string> > targetVector;
#else
                    std::vector<std::pair<int, int> > targetVector;
#endif
                    std::list<std::string>::iterator viter,
                      vend = iiter->_target1.end();
                    assert(iiter->_target1.size() > 0);
                    for (viter = iiter->_target1.begin(); viter != vend;
                        ++viter)
                    {
#ifdef MICRODOMAIN_CALCIUM
                      //SOMA-only model case
                      //find out a Channel-branch associate output data
                      // to what (can be many )junction-branch (e.g. Voltage, Calcium)
                      //  --> _channelJunctionIndices1
                      std::string compartmentNameWithOptionalMicrodomainName(*viter);
                      std::string compartmentNameOnly("");
                      std::string microdomainName("");
                      Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                      checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                      if (not microdomainName.empty())
                        _microdomainOnJunction[&((*iter)->lastCapsule())].insert(microdomainName);
                      std::map<std::string,
                        std::map<Capsule*, std::vector<int> > >::
                          iterator jmapiter1 =
                          _junctionIndexMap.find(compartmentNameOnly);
                      std::map<Capsule*, std::vector<int> >::iterator
                        jmapiter2;
                      if (jmapiter1 != _junctionIndexMap.end() &&
                          (jmapiter2 = jmapiter1->second.find(
                                                              &(*iter)->lastCapsule())) !=
                          jmapiter1->second.end())
                      {
                        std::vector<int>& junctionIndices =
                          jmapiter2->second;
                        targetVector.push_back(std::tuple<int, int, std::string>(
                              junctionIndices[1],
                              _compartmentVariableTypesMap[compartmentNameOnly],
                              microdomainName));
                      }
#else
                      std::map<std::string,
                        std::map<Capsule*, std::vector<int> > >::
                          iterator jmapiter1 =
                          _junctionIndexMap.find(*viter);
                      std::map<Capsule*, std::vector<int> >::iterator
                        jmapiter2;
                      if (jmapiter1 != _junctionIndexMap.end() &&
                          (jmapiter2 = jmapiter1->second.find(
                                                              &(*iter)->lastCapsule())) !=
                          jmapiter1->second.end())
                      {
                        std::vector<int>& junctionIndices =
                          jmapiter2->second;
                        targetVector.push_back(std::pair<int, int>(
                              junctionIndices[1],
                              _compartmentVariableTypesMap[*viter]));
                      }
#endif
                    }
                    _channelJunctionIndices1[_channelTypeCounter].push_back(
                        targetVector);
                    targetVector.clear();
                    vend = iiter->_target2.end();
                    assert(iiter->_target2.size() > 0);
                    for (viter = iiter->_target2.begin(); viter != vend;
                        ++viter)
                    {
#ifdef MICRODOMAIN_CALCIUM
                      //SOMA-only model case
                      //find out a Channel-branch associate output data
                      // to what (can be many )junction-branch (e.g. Voltage, Calcium)
                      //  --> _channelJunctionIndices2
                      std::string compartmentNameWithOptionalMicrodomainName(*viter);
                      std::string compartmentNameOnly("");
                      std::string microdomainName("");
                      Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                      checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
                      if (not microdomainName.empty())
                        _microdomainOnJunction[&((*iter)->lastCapsule())].insert(microdomainName);
                      std::map<std::string,
                        std::map<Capsule*, std::vector<int> > >::
                          iterator jmapiter1 =
                          _junctionIndexMap.find(compartmentNameOnly);
                      std::map<Capsule*, std::vector<int> >::iterator
                        jmapiter2;
                      if (jmapiter1 != _junctionIndexMap.end() &&
                          (jmapiter2 = jmapiter1->second.find(
                                                              &(*iter)->lastCapsule())) !=
                          jmapiter1->second.end())
                      {
                        std::vector<int>& junctionIndices =
                          jmapiter2->second;
                        targetVector.push_back(std::tuple<int, int, std::string>(
                              junctionIndices[1],
                              _compartmentVariableTypesMap[compartmentNameOnly],
                              microdomainName));
                      }
#else
                      std::map<std::string,
                        std::map<Capsule*, std::vector<int> > >::
                          iterator jmapiter1 =
                          _junctionIndexMap.find(*viter);
                      std::map<Capsule*, std::vector<int> >::iterator
                        jmapiter2;
                      if (jmapiter1 != _junctionIndexMap.end() &&
                          (jmapiter2 = jmapiter1->second.find(
                                                              &(*iter)->lastCapsule())) !=
                          jmapiter1->second.end())
                      {
                        std::vector<int>& junctionIndices =
                          jmapiter2->second;
                        targetVector.push_back(std::pair<int, int>(
                              junctionIndices[1],
                              _compartmentVariableTypesMap[*viter]));
                      }
#endif
                    }
                    _channelJunctionIndices2[_channelTypeCounter].push_back(
                        targetVector);
                  }
                }
              }
            }

          }

#endif
        }
      }
    }
  }

  if (nodeCategory == "Channels") ++_channelTypeCounter;
  if (nodeCategory == "ElectricalSynapses") ++_electricalSynapseTypeCounter;
  if (nodeCategory == "BidirectionalConnections")
    ++_bidirectionalConnectionTypeCounter;
  if (nodeCategory == "ChemicalSynapses") ++_chemicalSynapseTypeCounter;
  if (nodeCategory == "CompartmentVariables") ++_compartmentVariableTypeCounter;
  if (nodeCategory == "Junctions") ++_junctionTypeCounter;
  if (nodeCategory == "PreSynapticPoints") ++_preSynapticPointTypeCounter;
  if (nodeCategory == "SynapticClefts") ++_synapticCleftTypeCounter;
  if (nodeCategory == "EndPoints") ++_endPointTypeCounter;
  if (nodeCategory == "JunctionPoints") ++_junctionPointTypeCounter;
  if (nodeCategory == "BackwardSolvePoints") ++_backwardSolvePointTypeCounter;
  if (nodeCategory == "ForwardSolvePoints") ++_forwardSolvePointTypeCounter;
  
  return rval;
}

// GOAL: perform initialization for different node instances (i.e. nodeset)
// NOTE:  the nodeset must come from the same layer, i.e. of the same NodeType
//      'gld' variable represents the layer to which these node instances belong
//  the initial data can be passed through via
//     1. shared data: via name declaration in GSL (e.g. NodeType ...)
//     2. specific data:
//         + via NodeInit() in GSL
//         + via data in *params.par files
void TissueFunctor::doNodeInit(LensContext* lc)
{
  //NOTE: _params holds the NDPairList data passed to tissueFunctor as part of the InitNodes
  // statement
  assert(_params.get());
  std::auto_ptr<ParameterSet> initPset;
  std::vector<NodeDescriptor*> nodes;  // pointer to instances of nodes in grids
  std::vector<NodeDescriptor*>::iterator node, nodesEnd;

  NodeSet* nodeset =
      lc->layerContext->nodeset;  // set of instances of nodes to be initialized
  std::vector<GridLayerDescriptor*> const& layers =
      nodeset->getLayers();  // determine the layer to which this nodeset belong
  assert(layers.size() == 1);  // make sure they come from the same layer
  std::vector<GridLayerDescriptor*>::const_iterator gld = layers.begin();

  std::vector<std::string> nodekind;
  getNodekind(&(*gld)->getNDPList(), nodekind);
  assert(nodekind.size() > 0);
  std::string& nodeCategory = nodekind[0];
  std::string& nodeType = nodekind[1];

  // add the new layer to be tracked by the right bookkeeper
  if (nodeCategory == "CompartmentVariables")
  {
    _compartmentVariableLayers.push_back(*gld);
  }
  else if (nodeCategory == "Junctions")
  {
    _junctionLayers.push_back(*gld);
  }
  else if (nodeCategory == "EndPoints")
  {
    _endPointLayers.push_back(*gld);
  }
  else if (nodeCategory == "ForwardSolvePoints")
  {
    _forwardSolvePointLayers.push_back(*gld);
  }
  else if (nodeCategory == "BackwardSolvePoints")
  {
    _backwardSolvePointLayers.push_back(*gld);
  }
  else if (nodeCategory == "JunctionPoints")
  {
    _junctionPointLayers.push_back(*gld);
  }
  else if (nodeCategory == "Channels")
  {
    _channelLayers.push_back(*gld);
  }
  else if (nodeCategory == "ElectricalSynapses")
  {
    _electricalSynapseLayers.push_back(*gld);
  }
  else if (nodeCategory == "BidirectionalConnections")
  {
    _bidirectionalConnectionLayers.push_back(*gld);
  }
  else if (nodeCategory == "ChemicalSynapses")
  {
    _chemicalSynapseLayers.push_back(*gld);
    //TUAN TODO BUG: there is a potential bug at all locations like this
    //when the InitNode of Layers is declared not the same order as Layer defined
    //for a particular type of Layer
  }
  else if (nodeCategory == "PreSynapticPoints")
  {
    _preSynapticPointLayers.push_back(*gld);
  }
  else if (nodeCategory == "SynapticClefts")
  {
    _synapticCleftLayers.push_back(*gld);
  }
  else
  {
    std::cerr << "Unrecognized nodeCategory parameter on NodeInit : "
              << nodeCategory << std::endl;
    exit(EXIT_FAILURE);
  }

  if (_preSynapticPointLayers.size() > 0 && _synapticCleftLayers.size() > 0)
  {//validation check
    std::cerr << "We cannot have both PreSynapticPoints and SynapticClefts"
              << " in the system\n";
    exit(EXIT_FAILURE);
  }
  // make sure the order of layer's nodes' initialization is correct
  if (((nodeCategory == "Channels" || nodeCategory == "ElectricalSynapses" ||
        nodeCategory == "BidirectionalConnections" ||
        nodeCategory == "ChemicalSynapses" ||
        nodeCategory == "PreSynapticalPoints" ||
        nodeCategory == "SynapticClefts") &&
       (_junctionLayers.size() == 0 ||
        _compartmentVariableLayers.size() == 0)) ||
      (_junctionLayers.size() > 0 && _compartmentVariableLayers.size() == 0) ||
      ((_preSynapticPointLayers.size() == 0 &&
        _chemicalSynapseLayers.size() > 0) &&
       (_synapticCleftLayers.size() == 0 && _chemicalSynapseLayers.size() > 0)))
  {
    std::cerr
        << "TissueFunctor:" << std::endl
        << "Layers (Branches, Junctions, PreSynapticPoints | SynapticClefts,"
        << "Channels | Synapses "
           ") must be initialized in order." << std::endl
        << "Synapses can be either ElectricalSynapses, ChemicalSynapses, "
           "BidirectionalConnections" << std::endl;
    exit(EXIT_FAILURE);
  }

  // reset and start pointing to the instances for
  // the nodeset to be initialized, which is accessed via the LayerDescriptor 'gld'
  nodes.clear();
  nodeset->getNodes(nodes, *gld);

  node = nodes.begin();
  nodesEnd = nodes.end();
  NDPairList emptyOutAttr;
  NDPairList dim2cpt;
  // NOTE: inAttr that we use to connect 'DimensionStruct' and 'BranchData'
  //   to the each node instance
  dim2cpt.push_back(new NDPair("identifier", "dimension"));
  NDPairList brd2cpt;
  brd2cpt.push_back(new NDPair("identifier", "branchData"));

  NDPairList paramsLocal = *(_params.get());
  NDPairList::iterator ndpiter = _params->begin(); 
  Functor* ifunctor = 0;
  while (ndpiter!=_params->end() ) {
    if ( (*ndpiter)->getName()=="initializer") {
      FunctorDataItem* functorDI =
      dynamic_cast<FunctorDataItem*>((*ndpiter)->getDataItem());
      if (functorDI == 0)
	  std::cerr << "Reserved \"initializer\" parameter of TissueProbe must be a functor!"
		    << std::endl;
      else {
	ifunctor = functorDI->getFunctor();
	_params->clear();
	break;
      }
    }
    ++ndpiter;
  }
  for (; node != nodesEnd; ++node)
  {  // traverse all instances of the nodeset
    if ((*node)->getNode())
    {
      NDPairList paramsLocal = *(_params.get());
      (*gld)->getNodeType()->getInitializationParameterSet(initPset);
      if (ifunctor) {
	std::vector<DataItem*> nullArgs;
	std::auto_ptr<DataItem> rval_ap;
	ifunctor->execute(lc, nullArgs, rval_ap);
	NDPairListDataItem* ndpldi = 
	  dynamic_cast<NDPairListDataItem*>(rval_ap.get());
	if (ndpldi == 0) {
	  throw SyntaxErrorException(
	    "Dynamic cast of DataItem to NDPairListDataItem failed on TissueFunctor");
	}
	else {
	  initPset->set(*(ndpldi->getNDPairList()));
	}
      }
      ParameterSet* pset = initPset.get();
      int nodeIndex = (*node)->getNodeIndex();
      int densityIndex = (*node)->getDensityIndex();
      if (nodeCategory == "CompartmentVariables" || nodeCategory == "Junctions")
      {
        int size = compartmentalize(lc, &paramsLocal, nodeCategory, nodeType,
                                    nodeIndex, densityIndex);
        StructType* st = lc->sim->getStructType("BranchDataStruct");
        ConstantType* ct = lc->sim->getConstantType("BranchData");
        NDPairList branchDataStructParams;
        IntDataItem* sizeDI = new IntDataItem(size);
        std::auto_ptr<DataItem> sizeDI_ap(sizeDI);
        NDPair* ndp = new NDPair("size", sizeDI_ap);
        branchDataStructParams.push_back(ndp);
        if (nodeCategory == "CompartmentVariables")
        {  // initialize: branchData, dimensions
          // now start initializing data members, e.g.
          // HHVoltage{
          // BranchDataStruct* branchData; //key_size_t key; unsigned size;
          // DimensionStruct* [] dimensions; //
          //}
          ComputeBranch* branch = findBranch(nodeIndex, densityIndex, nodeType);
          CG_BranchData*& branchData =
              _tissueContext->_branchBranchDataMap[branch];
          key_size_t key = branch->_capsules[0].getKey();
          getModelParams(Params::COMPARTMENT, paramsLocal, nodeType, key);
          pset->set(paramsLocal);
          (*node)->getNode()->initialize(pset);
          // TUAN NOTE: another location where bug may occur if we
          // change the key size (key_size_t)
          DoubleDataItem* keyDI =
              new DoubleDataItem(branch->_capsules[0].getKey());
          std::auto_ptr<DataItem> keyDI_ap(keyDI);
          ndp = new NDPair("key", keyDI_ap);
          branchDataStructParams.push_back(ndp);
          std::auto_ptr<DataItem> aptr_st;
          st->getInstance(aptr_st, branchDataStructParams, lc);
          ndp = new NDPair("branchData", aptr_st);
          NDPairList branchDataParams;
          branchDataParams.push_back(ndp);
          std::auto_ptr<DataItem> aptr_cst;
          ct->getInstance(aptr_cst, branchDataParams, lc);
          ConstantDataItem* cdi =
              dynamic_cast<ConstantDataItem*>(aptr_cst.get());
          std::auto_ptr<Constant> aptr_brd;
          cdi->getConstant()->duplicate(aptr_brd);
          Constant* brd = aptr_brd.release();
          branchData = (dynamic_cast<CG_BranchData*>(brd));
          assert(branchData);
          {
            std::map<ComputeBranch*,
              std::vector<CG_CompartmentDimension*> >::iterator miter =
                _tissueContext->_branchDimensionsMap.find(branch);
            assert(miter != _tissueContext->_branchDimensionsMap.end());
            std::vector<CG_CompartmentDimension*>& dimensions = miter->second;
            std::vector<CG_CompartmentDimension*>::iterator diter,
              dend = dimensions.end();
            // step 1: connect every DimensionStruct element into the array
            // 'dimensions'
            //  of each HHVoltage node instance, for example
            //  REMEMBER: Even one is element, one is array,
            //   the connection is established in such a way that the
            //   first connection fills into the first position in the array
            //   next connection fills into the next position in the array
            for (diter = dimensions.begin(); diter != dend; ++diter)
              _lensConnector.constantToNode(*diter, *node, &emptyOutAttr,
                  &dim2cpt, lc->sim);
          }
#ifdef MICRODOMAIN_CALCIUM
          // step 2: connect branchData to the data member 'branchData'
          //  of each HHVoltage node instance, for example
          //  NOTE: Here we also put the information about the list of all microdomain
          //   to 'Calcium' compartment
          std::set< std::string>  setMicroDomains = _microdomainOnBranch[branch];
          std::string result("");
          if (setMicroDomains.size() > 0 and nodeTypeWithAllowedMicrodomain(nodeType))
          {
            std::ostringstream stream;
            std::copy(setMicroDomains.begin(), setMicroDomains.end(), std::ostream_iterator<std::string>(stream, ","));
            result = stream.str();
            result.pop_back();
            NDPairList brd2cptWithMicroDomainSupport = brd2cpt;
            //NOTE: 'result' holds comma-separated string of name of all microdomains
            brd2cptWithMicroDomainSupport.push_back(new NDPair("domainName", result));
            _lensConnector.constantToNode(branchData, *node, &emptyOutAttr,
                &brd2cptWithMicroDomainSupport, lc->sim);
          }
          else
          {
            _lensConnector.constantToNode(branchData, *node, &emptyOutAttr,
                &brd2cpt, lc->sim);
          }
#else
          // step 2: connect branchData to the data member 'branchData'
          //  of each HHVoltage node instance, for example
          _lensConnector.constantToNode(branchData, *node, &emptyOutAttr,
                                        &brd2cpt, lc->sim);
#endif

        }
        else
        {  // explicit Junctions
          // now start initializing data members, e.g.
          // HHVoltageJunction{
          // BranchDataStruct* branchData; //key_size_t key; unsigned size;
          // DimensionStruct* [] dimensions; //
          //}
          Capsule* junctionCapsule =
              findJunction(nodeIndex, densityIndex, nodeType);
          CG_BranchData*& branchData =
              _tissueContext->_junctionBranchDataMap[junctionCapsule];
          key_size_t key = junctionCapsule->getKey();
          getModelParams(Params::COMPARTMENT, paramsLocal, nodeType, key);
          pset->set(paramsLocal);
          (*node)->getNode()->initialize(pset);
          DoubleDataItem* keyDI = new DoubleDataItem(key);
          std::auto_ptr<DataItem> keyDI_ap(keyDI);
          NDPair* ndp = new NDPair("key", keyDI_ap);
          branchDataStructParams.push_back(ndp);
          std::auto_ptr<DataItem> aptr_st;
          st->getInstance(aptr_st, branchDataStructParams, lc);
          ndp = new NDPair("branchData", aptr_st);
          NDPairList branchDataParams;
          branchDataParams.push_back(ndp);
          std::auto_ptr<DataItem> aptr_cst;
          ct->getInstance(aptr_cst, branchDataParams, lc);
          ConstantDataItem* cdi =
              dynamic_cast<ConstantDataItem*>(aptr_cst.get());
          std::auto_ptr<Constant> aptr_brd;
          cdi->getConstant()->duplicate(aptr_brd);
          Constant* brd = aptr_brd.release();
          branchData = (dynamic_cast<CG_BranchData*>(brd));
          assert(branchData);
          {
            std::map<Capsule*, CG_CompartmentDimension*>::iterator miter =
              _tissueContext->_junctionDimensionMap.find(junctionCapsule);
            assert(miter != _tissueContext->_junctionDimensionMap.end());
            // step 1: connect every DimensionStruct element into the array
            // 'dimensions'
            //  of each HHVoltageJunction node instance, for example
            //  REMEMBER: Even one is element, one is array,
            //   the connection is established in such a way that the
            //   first connection fills into the first position in the array
            //   next connection fills into the next position in the array
            _lensConnector.constantToNode(miter->second, *node, &emptyOutAttr,
                &dim2cpt, lc->sim);
          }

#ifdef MICRODOMAIN_CALCIUM
          // step 2: connect branchData to the data member 'branchData'
          //  of each HHVoltageJunction node instance, for example
          //  NOTE: Here we also put the information about the list of all microdomain
          //   to 'Calcium' compartment
          std::set< std::string>  setMicroDomains = _microdomainOnJunction[junctionCapsule];
          std::string result("");
          if (setMicroDomains.size() > 0 and nodeTypeWithAllowedMicrodomain(nodeType))
          {
            std::ostringstream stream;
            std::copy(setMicroDomains.begin(), setMicroDomains.end(), std::ostream_iterator<std::string>(stream, ","));
            result = stream.str();
            result.pop_back();
            NDPairList brd2cptWithMicroDomainSupport = brd2cpt;
            //NOTE: 'result' holds comma-separated string of name of all microdomains
            brd2cptWithMicroDomainSupport.push_back(new NDPair("domainName", result));
            _lensConnector.constantToNode(branchData, *node, &emptyOutAttr,
                &brd2cptWithMicroDomainSupport, lc->sim);
          }
          else
          {
            _lensConnector.constantToNode(branchData, *node, &emptyOutAttr,
                &brd2cpt, lc->sim);
          }
#else
          // step 2: connect branchData to the data member 'branchData'
          //  of each HHVoltageJunction node instance, for example
          _lensConnector.constantToNode(branchData, *node, &emptyOutAttr,
                                        &brd2cpt, lc->sim);
#endif

        }
      }
      else if (nodeCategory == "Channels")
      {
        assert(nodeIndex == _rank);
        std::string channelCategory;
        assert(_channelLayers.size() > 0);
        int nChannelBranches =
            _channelBranchIndices1[_channelLayers.size() - 1].size();
        key_size_t key;
        assert(_channelBranchIndices1[_channelLayers.size() - 1].size() ==
               _channelBranchIndices2[_channelLayers.size() - 1].size());
        if (densityIndex < nChannelBranches)
        {
          channelCategory = "BranchChannels";
#ifdef MICRODOMAIN_CALCIUM
          std::tuple<int, int, std::string>& channelBranchIndexPair =
              _channelBranchIndices1[_channelLayers.size() -
                                     1][densityIndex][0];
          key = findBranch(
                    nodeIndex, std::get<0>(channelBranchIndexPair),
                    _compartmentVariableTypes[std::get<1>(channelBranchIndexPair)])
                    ->_capsules[0]
                    .getKey();
#else
          std::pair<int, int>& channelBranchIndexPair =
              _channelBranchIndices1[_channelLayers.size() -
                                     1][densityIndex][0];
          key = findBranch(
                    nodeIndex, channelBranchIndexPair.first,
                    _compartmentVariableTypes[channelBranchIndexPair.second])
                    ->_capsules[0]
                    .getKey();
#endif
        }
        else
        {
          channelCategory = "JunctionChannels";
#ifdef MICRODOMAIN_CALCIUM
          std::tuple<int, int, std::string>& channelJunctionIndexPair =
              _channelJunctionIndices1[_channelLayers.size() -
                                       1][densityIndex - nChannelBranches][0];
          key = findJunction(
                    nodeIndex, std::get<0>(channelJunctionIndexPair),
                    _compartmentVariableTypes[std::get<1>(channelJunctionIndexPair)])
                    ->getKey();
#else
          std::pair<int, int>& channelJunctionIndexPair =
              _channelJunctionIndices1[_channelLayers.size() -
                                       1][densityIndex - nChannelBranches][0];
          key = findJunction(
                    nodeIndex, channelJunctionIndexPair.first,
                    _compartmentVariableTypes[channelJunctionIndexPair.second])
                    ->getKey();
#endif
        }

        std::list<std::pair<std::string, float> > channelParams;
        getModelParams(Params::CHANNEL, paramsLocal, nodeType, key);
        compartmentalize(lc, &paramsLocal, channelCategory, nodeType, nodeIndex,
                         densityIndex);

        pset->set(paramsLocal);
        (*node)->getNode()->initialize(pset);
      }
      else
      {
        pset->set(paramsLocal);
        (*node)->getNode()->initialize(pset);
      }
    }
  }
}

// GOAL: perform necessary setup and then connect instances of
// all declared nodetypes (different channels, receptors, ...)
// to the proper instance of the associated compartment variable (Voltage,
// Calcium)
// based on the condition given in different parameter files (SynParams.par,
// ChanParams.par)
void TissueFunctor::doConnector(LensContext* lc)
{
  assert(_compartmentVariableLayers.size() ==
         _compartmentVariableTypesMap.size());
  assert(_junctionLayers.size() == _junctionTypesMap.size());
  assert(_compartmentVariableTypesMap.size() == _junctionTypesMap.size());

  std::map<int, int> cptVarJctTypeMap;
  std::map<std::string, int>::iterator mapIter,
      mapEnd = _compartmentVariableTypesMap.end();
  for (mapIter = _compartmentVariableTypesMap.begin(); mapIter != mapEnd;
       ++mapIter)
  {
    assert(_junctionTypesMap.find(mapIter->first) != _junctionTypesMap.end());
    cptVarJctTypeMap[mapIter->second] = _junctionTypesMap[mapIter->first];
  }

  assert(_forwardSolvePointLayers.size() == _forwardSolvePointTypeCounter);
  assert(_backwardSolvePointLayers.size() == _backwardSolvePointTypeCounter);

  // NOTE: idx = index of the compartment on the branch
  //                of the given compartment-variable connecting to
  //                channel|chemicalsynapse|electricalsynapse
  // Example: "Voltage", <"identifier"="compartment[Voltage]">
  std::map<std::string, NDPairList> cpt2chan;
#ifdef MICRODOMAIN_CALCIUM
  std::map<std::string, NDPairList> cptMicrodomain2chan;
#endif
  // Example: "Voltage", <"identifier"="compartment[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> cpt2syn;
#ifdef MICRODOMAIN_CALCIUM
  std::map<std::string, NDPairList> cptMicrodomain2syn;
#endif
  // Example: "Voltage", <"identifier"="compartment[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> cpt2cleft;
  // Example: "Voltage", <"identifier"="channels[Voltage]">
  std::map<std::string, NDPairList> chan2cpt;
#ifdef MICRODOMAIN_CALCIUM
  std::map<std::string, NDPairList> chan2cptMicrodomain;
#endif
  // Example: "Voltage", <"identifier"="electricalSynapse[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> esyn2cpt;
  // Example: "Voltage", <"identifier"="chemicalSynapse[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> csyn2cpt;
#ifdef MICRODOMAIN_CALCIUM
  std::map<std::string, NDPairList> csyn2cptMicrodomain;
#endif
  // Example: "Voltage", <"identifier"="IC[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> ic2syn;
  // Example: "Voltage", <"identifier"="IC[Voltage]">
  std::map<std::string, NDPairList> ic2chan;
  // Example: "Voltage", <"identifier"="connexon[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> cnnxn2cnnxn;
  // Example: "Voltage", <"identifier"="compartment[Voltage]",
  //                   "idx"=0, //index of compartment
  //                   "typeCpt" = "spine-neck" or "den-shaft"
  std::map<std::string, NDPairList> cpt2spineattach;
  // Example: "Voltage", <"identifier"="spineAttachment[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> spineattach2cpt;
  // Example: "Voltage", <"identifier"="spineConnexon[Voltage]",
  //                   "idx"=0>
  std::map<std::string, NDPairList> spineattach2spineattach;

  // Define InAttrPset for connection to comparment-nodetype
  // Here, we assign value to the 'map' object defined above
  // The values represents all possible InAttrPSet that can be used to establish
  // a connection from the given nodetype to the proper compartment nodetypes
  //      (Voltage, Calcium, CalciumER ...)
  // variable and put to the 'map' objects defined above
  std::map<std::string, int>::iterator cptVarTypesIter,
      cptVarTypesEnd = _compartmentVariableTypesMap.end();
  for (cptVarTypesIter = _compartmentVariableTypesMap.begin();
       cptVarTypesIter != cptVarTypesEnd; ++cptVarTypesIter)
  {
    std::ostringstream os;

    NDPairList Mcpt2chan;
    os << "compartment[" << cptVarTypesIter->first << "]";
    Mcpt2chan.push_back(new NDPair("identifier", os.str()));
    cpt2chan[cptVarTypesIter->first] = Mcpt2chan;

#ifdef MICRODOMAIN_CALCIUM
    std::ostringstream osDomain;
    NDPairList McptMicrodomain2chan;
    osDomain << "compartment[" << cptVarTypesIter->first << "(domain)]";
    McptMicrodomain2chan.push_back(new NDPair("identifier", osDomain.str()));
    McptMicrodomain2chan.push_back(new NDPair("domainName", ""));
    cptMicrodomain2chan[cptVarTypesIter->first] = McptMicrodomain2chan;
#endif

    NDPairList Mcpt2syn;
    Mcpt2syn.push_back(new NDPair("identifier", os.str()));
    Mcpt2syn.push_back(new NDPair("idx", 0));
    cpt2syn[cptVarTypesIter->first] = Mcpt2syn;

#ifdef MICRODOMAIN_CALCIUM
    NDPairList McptMicrodomain2syn;
    osDomain.str("");
    osDomain << "compartment[" << cptVarTypesIter->first << "(domain)]";
    McptMicrodomain2syn.push_back(new NDPair("identifier", osDomain.str()));
    McptMicrodomain2syn.push_back(new NDPair("domainName", ""));
    cptMicrodomain2syn[cptVarTypesIter->first] = McptMicrodomain2syn;
#endif


    NDPairList Mcpt2cleft;
    Mcpt2cleft.push_back(new NDPair("identifier", os.str()));
    Mcpt2cleft.push_back(new NDPair("idx", 0));
    Mcpt2cleft.push_back(new NDPair("side", "")); 
    cpt2cleft[cptVarTypesIter->first] = Mcpt2cleft;

    NDPairList Mcpt2spineattach;
    Mcpt2spineattach.push_back(new NDPair("identifier", os.str()));
    Mcpt2spineattach.push_back(new NDPair("idx", 0));
    Mcpt2spineattach.push_back(new NDPair("typeCpt", ""));
    cpt2spineattach[cptVarTypesIter->first] = Mcpt2spineattach;

    os.str("");
    NDPairList Mchan2cpt;
    os << "channels[" << cptVarTypesIter->first << "]";
    Mchan2cpt.push_back(new NDPair("identifier", os.str()));
    chan2cpt[cptVarTypesIter->first] = Mchan2cpt;

#ifdef MICRODOMAIN_CALCIUM
    osDomain.str("");
    NDPairList Mchan2cptMicrodomain;
    osDomain << "channels[" << cptVarTypesIter->first << "(domain)]";
    Mchan2cptMicrodomain.push_back(new NDPair("identifier", osDomain.str()));
    Mchan2cptMicrodomain.push_back(new NDPair("domainName", ""));
    chan2cptMicrodomain[cptVarTypesIter->first] = Mchan2cptMicrodomain;
#endif

    os.str("");
    NDPairList Mesyn2cpt;
    os << "electricalSynapse[" << cptVarTypesIter->first << "]";
    Mesyn2cpt.push_back(new NDPair("identifier", os.str()));
    Mesyn2cpt.push_back(new NDPair("idx", 0));
    esyn2cpt[cptVarTypesIter->first] = Mesyn2cpt;

    os.str("");
    NDPairList Mcsyn2cpt;
    os << "chemicalSynapse[" << cptVarTypesIter->first << "]";
    Mcsyn2cpt.push_back(new NDPair("identifier", os.str()));
    Mcsyn2cpt.push_back(new NDPair("idx", 0));
    csyn2cpt[cptVarTypesIter->first] = Mcsyn2cpt;
#ifdef MICRODOMAIN_CALCIUM
    osDomain.str("");
    NDPairList Mcsyn2cptMicrodomain;
    osDomain << "chemicalSynapse[" << cptVarTypesIter->first << "(domain)]";
    Mcsyn2cptMicrodomain.push_back(new NDPair("identifier", osDomain.str()));
    Mcsyn2cptMicrodomain.push_back(new NDPair("domainName", ""));
    csyn2cptMicrodomain[cptVarTypesIter->first] = Mcsyn2cptMicrodomain;
#endif

    os.str("");
    NDPairList Mic2syn;
    os << "IC[" << cptVarTypesIter->first << "]";
    Mic2syn.push_back(new NDPair("identifier", os.str()));
    Mic2syn.push_back(new NDPair("idx", 0));
    ic2syn[cptVarTypesIter->first] = Mic2syn;

    os.str("");
    NDPairList Mic2chan;
    os << "IC[" << cptVarTypesIter->first << "]";
    Mic2chan.push_back(new NDPair("identifier", os.str()));
    ic2chan[cptVarTypesIter->first] = Mic2chan;

    os.str("");
    NDPairList Mcnnxn2cnnxn;
    os << "connexon[" << cptVarTypesIter->first << "]";
    Mcnnxn2cnnxn.push_back(new NDPair("identifier", os.str()));
    cnnxn2cnnxn[cptVarTypesIter->first] = Mcnnxn2cnnxn;

    os.str("");  // to be used inside MDL for compartments
    NDPairList Mecplg2cpt;
    os << "spineAttachment[" << cptVarTypesIter->first << "]";
    Mecplg2cpt.push_back(new NDPair("identifier", os.str()));
    Mecplg2cpt.push_back(new NDPair("idx", 0));
    spineattach2cpt[cptVarTypesIter->first] = Mecplg2cpt;

    os.str("");  // to be used for
    NDPairList Mspineattach2spineattach;
    os << "spineConnexon[" << cptVarTypesIter->first << "]";
    Mspineattach2spineattach.push_back(new NDPair("identifier", os.str()));
    spineattach2spineattach[cptVarTypesIter->first] = Mspineattach2spineattach;
  }

  // Define InAttrPset for connection to non-compartment nodetype
  // Example of non-compartment nodetype:
  //  e.g. 1. 'point', i.e. endPoint, junctionPoint, SolvePoint,
  //                               presynapticPoint, SynapticCleft
  //       2. junction
  //       3. receptor
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

  NDPairList synCleft;
  synCleft.push_back(new NDPair("identifier", "synapticCleft"));

  NDPairList recp2recp;
  recp2recp.push_back(new NDPair("identifier", "receptor"));

  NDPairList br2fwdpt;
  NDPairList br2bwdpt;

  // get to the vector of all instances of nodetypes associated with all compartment-layers
  std::vector<NodeAccessor*> compartmentVariableAccessors;
  std::vector<GridLayerDescriptor*>::iterator layerIter,
      layerEnd = _compartmentVariableLayers.end();
  for (layerIter = _compartmentVariableLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    compartmentVariableAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all junction-layers
  layerEnd = _junctionLayers.end();
  std::vector<NodeAccessor*> junctionAccessors;
  for (layerIter = _junctionLayers.begin(); layerIter != layerEnd; ++layerIter)
  {
    junctionAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all endpoint-layers
  std::vector<NodeAccessor*> endPointAccessors;
  layerEnd = _endPointLayers.end();
  for (layerIter = _endPointLayers.begin(); layerIter != layerEnd; ++layerIter)
  {
    endPointAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all junctionpoint-layers
  std::vector<NodeAccessor*> junctionPointAccessors;
  layerEnd = _junctionPointLayers.end();
  for (layerIter = _junctionPointLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    junctionPointAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all
  // presynapticpoint-layers
  std::vector<NodeAccessor*> preSynapticPointAccessors;
  layerEnd = _preSynapticPointLayers.end();
  for (layerIter = _preSynapticPointLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    preSynapticPointAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all
  // synapticClefts-layers
  std::vector<NodeAccessor*> synapticCleftAccessors;
  layerEnd = _synapticCleftLayers.end();
  for (layerIter = _synapticCleftLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    synapticCleftAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all
  // forwardsolvepoint-layers
  std::vector<NodeAccessor*> forwardSolvePointAccessors;
  layerEnd = _forwardSolvePointLayers.end();
  for (layerIter = _forwardSolvePointLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    forwardSolvePointAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all
  // backwardsolvepoint-layers
  std::vector<NodeAccessor*> backwardSolvePointAccessors;
  layerEnd = _backwardSolvePointLayers.end();
  for (layerIter = _backwardSolvePointLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    backwardSolvePointAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all channel-layers
  std::vector<NodeAccessor*> channelAccessors;
  layerEnd = _channelLayers.end();
  for (layerIter = _channelLayers.begin(); layerIter != layerEnd; ++layerIter)
  {
    channelAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all
  // bidirectional-connection-layers
  // 2 types of bidirectional-connection
  // 1.1 = gap junction
  std::vector<NodeAccessor*> electricalSynapseAccessors;
  layerEnd = _electricalSynapseLayers.end();
  for (layerIter = _electricalSynapseLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    electricalSynapseAccessors.push_back((*layerIter)->getNodeAccessor());
  }
  // 1.2 = spine attachment
  //  loop through each Layer defined for BidirectionalConnections
  //  and get access to the list of instances for that Layer
  std::vector<NodeAccessor*> bidirectionalConnectionAccessors;
  layerEnd = _bidirectionalConnectionLayers.end();
  for (layerIter = _bidirectionalConnectionLayers.begin();
       layerIter != layerEnd; ++layerIter)
  {
    bidirectionalConnectionAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  // get to the vector of all instances of nodetypes associated with all
  // chemicalsynapse-layers
  std::vector<NodeAccessor*> chemicalSynapseAccessors;
  layerEnd = _chemicalSynapseLayers.end();
  for (layerIter = _chemicalSynapseLayers.begin(); layerIter != layerEnd;
       ++layerIter)
  {
    chemicalSynapseAccessors.push_back((*layerIter)->getNodeAccessor());
  }

  Simulation* sim = lc->sim;
  Connector* connector;

  if (sim->isGranuleMapperPass())
  {
    connector = &_noConnector;
  }
  else if (sim->isCostAggregationPass())
  {
    connector = &_granuleConnector;
  }
  else if (sim->isSimulatePass())
  {
    connector = &_lensConnector;
  }
  else
  {
    std::cerr << "Error, TissueFunctor : no connection context set!"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // GOAL: create connection
  //      Cpt1->EndPoint-->Junction-->JunctionPoint->Cpt2
  //      ImplicitJunction-->ForwardSolverPoint
  //      ImplicitJunction-->BackwardSolverPoint
  // HOW: traverse each grid's node, (NOTE: one grid node ~ 1 MPI process)
  //   at each grid'node, traverse different compartment nodeType (e.g. Voltage, Calcium)
  //   and get the density
  //    which is the number of ComputeBranch on that grid's node
  // .....traverse through all index in the density
  //    find out the compute-branch value on such ComputeBranch
  for (int i = 0; i < _nbrGridNodes; ++i)
  {
    //    then traverse through every ComputeBranch
    //       and connect to the proper nodetype
    std::map<std::string, int>::iterator cptVarTypesIter,
        cptVarTypesEnd = _compartmentVariableTypesMap.end();
    for (cptVarTypesIter = _compartmentVariableTypesMap.begin();
         cptVarTypesIter != cptVarTypesEnd; ++cptVarTypesIter)
    {
      std::string cptVarType = cptVarTypesIter->first;  // e.g. 'Voltage'
      int cptVarTypeIdx = cptVarTypesIter->second;  // index of Layer statement
      // branchDensity = how many elements (i.e. how many ComputeBranch)
      int branchDensity =
          _compartmentVariableLayers[cptVarTypeIdx]->getDensity(i);
      std::vector<int> endPointCounters;
      endPointCounters.resize(_endPointTypeCounter, 0);

      for (int j = 0; j < branchDensity; ++j)
      {  // get to the branch
        ComputeBranch* br = findBranch(i, j, cptVarType);
        if (br)
        {
          key_size_t key = br->_capsules[0].getKey();
          int computeOrder = _segmentDescriptor.getComputeOrder(key);
          assert(i == _tissueContext->getRankOfBeginPoint(br));
          int endPointType = _endPointTypesMap[cptVarType];
          int junctionPointType = _junctionPointTypesMap[cptVarType];

          if (br->_daughters.size() > 0)
          {  // check distal-end
            // if the ComputeBranch connects to other distal-ComputeBranch
            // get to all instances of the associated diffusible nodetype (e.g.
            //     Voltage, Calcium)
            //     at one ComputeBranch
            NodeDescriptor* compartmentVariable =
                compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(
                    i, j);
            assert(i ==
                   sim->getGranule(*compartmentVariable)->getPartitionId());
            // check the computeOrder of that branch (based on first-capsule)
#ifdef IDEA1
            //NO NEED to add code
#endif
            if (computeOrder == MAX_COMPUTE_ORDER)
            {  // start to have an explicit junction on distalEnd
              // connect (1) compartment to endpoint 'distal-end'
              //         (2) endpoint to explicit junction
              //         (3) explicit junction to junction point,
              //         (4) junction point to proximalEnd of another
              //         compartment
              NodeDescriptor* endPoint =
                  endPointAccessors[endPointType]->getNodeDescriptor(
                      i, endPointCounters[endPointType]);
              ++endPointCounters[endPointType];
              connect(sim, connector, compartmentVariable, endPoint, dist2end);

              Capsule* c = &br->lastCapsule();
              std::vector<int>& junctionIndices =
                  findJunctionIndices(c, cptVarType);
              NodeDescriptor* junction =
                  junctionAccessors[cptVarJctTypeMap[cptVarTypeIdx]]
                      ->getNodeDescriptor(junctionIndices[0],
                                          junctionIndices[1]);
              connect(sim, connector, endPoint, junction, end2jct);
              NodeDescriptor* junctionPoint =
                  junctionPointAccessors[junctionPointType]->getNodeDescriptor(
                      junctionIndices[0], junctionIndices[1]);
              connect(sim, connector, junction, junctionPoint, jct2jctpt);
              connect(sim, connector, junctionPoint, compartmentVariable,
                      jctpt2prox);
            }
            else
            {  // implicit junction at distal-end
               // connect (1) compartment to backward solver
              std::vector<int>& backwardSolvePointIndices =
                  findBackwardSolvePointIndices(br, cptVarType);
              assert(i == backwardSolvePointIndices[0]);
              NodeDescriptor* backwardSolvePoint =
                  backwardSolvePointAccessors
                      [_backwardSolvePointTypesMap[computeOrder][cptVarType]]
                          ->getNodeDescriptor(i, backwardSolvePointIndices[1]);
              connect(sim, connector, compartmentVariable, backwardSolvePoint,
                      br2bwdpt);
            }
          }

          if (br->_parent)
          {  // check proximal-end
            // if	it connects to a proximal-ComputeBranch
            NodeDescriptor* compartmentVariable =
                compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(
                    i, j);
            assert(i ==
                   sim->getGranule(*compartmentVariable)->getPartitionId());
            if (computeOrder == 0)
            {  // start to have an explicit junction at proximal-end
               // connect (1) compartment to endpoint
               //         (2) endpoint to explicit junction
               //         (3) explicit junction to junction point,
              //         (4) junction point to distalEnd of another compartment
              NodeDescriptor* endPoint =
                  endPointAccessors[endPointType]->getNodeDescriptor(
                      i, endPointCounters[endPointType]);
              ++endPointCounters[endPointType];
              connect(sim, connector, compartmentVariable, endPoint, prox2end);
              Capsule* c = &br->_parent->lastCapsule();
              std::vector<int>& junctionIndices =
                  findJunctionIndices(c, cptVarType);
              NodeDescriptor* junction =
                  junctionAccessors[cptVarJctTypeMap[cptVarTypeIdx]]
                      ->getNodeDescriptor(junctionIndices[0],
                                          junctionIndices[1]);
              connect(sim, connector, endPoint, junction, end2jct);
              NodeDescriptor* junctionPoint =
                  junctionPointAccessors[junctionPointType]->getNodeDescriptor(
                      junctionIndices[0], junctionIndices[1]);
              connect(sim, connector, junction, junctionPoint, jct2jctpt);
              connect(sim, connector, junctionPoint, compartmentVariable,
                      jctpt2dist);
            }
            else
            {  // implicit junction at proximal-end
               // connect (1) compartment to forward solver
              std::vector<int>& forwardSolvePointIndices =
                  findForwardSolvePointIndices(br, cptVarType);
              assert(i == forwardSolvePointIndices[0]);
              NodeDescriptor* forwardSolvePoint =
                  forwardSolvePointAccessors
                      [_forwardSolvePointTypesMap[computeOrder][cptVarType]]
                          ->getNodeDescriptor(i, forwardSolvePointIndices[1]);
              connect(sim, connector, compartmentVariable, forwardSolvePoint,
                      br2fwdpt);
            }
          }
        }
      }
    }
  }

  int i = _rank;
  assert(_channelTypeCounter == _channelLayers.size());
  // GOAL: connect channel to comparment variables
  // HOW: traverse all layers declared for channels
  // and connect to the proper compartment variables
  for (int ctype = 0; ctype < _channelTypeCounter; ++ctype)
  {
    int channelDensity = _channelLayers[ctype]->getDensity(i);
#ifdef MICRODOMAIN_CALCIUM
    std::vector<std::vector<std::tuple<int, int, std::string> > >::iterator iter, end;
#else
    std::vector<std::vector<std::pair<int, int> > >::iterator iter, end;
#endif
    iter = _channelBranchIndices1[ctype].begin();
    end = _channelBranchIndices1[ctype].end();
    // for a given channel, traverse all array elements
    // from the given grid's node index 'i'
    //      and density-index 'j'
#ifdef MICRODOMAIN_CALCIUM
    for (int j = 0; iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::tuple<int, int, std::string > >& channelBranchIndexPairs = (*iter);
      std::vector<std::tuple<int, int, std::string > >::iterator
          ctiter = channelBranchIndexPairs.begin(),
          ctend = channelBranchIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(std::get<1>(*ctiter) < _compartmentVariableTypesMap.size());
        compartmentVariable =
            compartmentVariableAccessors[std::get<1>(*ctiter)]->getNodeDescriptor(
                i, std::get<0>(*ctiter));
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        if (std::get<2>(*ctiter).empty())
        {//well-mixed region
          // connect to channel that use the compartment variable as 'IC[]'
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
          // connect to channel that use the compartment variable as
          // 'compartment[]'
          connect(sim, connector, compartmentVariable, channel,
              cpt2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
        }
        else{//if a compartment (e.g. Calcium) that allows microdomain
          // connect to channel that use the compartment variable as 'IC[]'
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
          // connect to channel that use the compartment variable as
          // 'compartment[]'
          NDPairList McptMicrodomain2chan = cptMicrodomain2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]];
          McptMicrodomain2chan.replace("domainName", std::get<2>(*ctiter));
          connect(sim, connector, compartmentVariable, channel,
              McptMicrodomain2chan);
        }
      }
    }
#else
    for (int j = 0; iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::pair<int, int> >& channelBranchIndexPairs = (*iter);
      std::vector<std::pair<int, int> >::iterator
          ctiter = channelBranchIndexPairs.begin(),
          ctend = channelBranchIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(ctiter->second < _compartmentVariableTypesMap.size());
        compartmentVariable =
            compartmentVariableAccessors[ctiter->second]->getNodeDescriptor(
                i, ctiter->first);
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        // connect to channel that use the compartment variable as 'IC[]'
        connect(sim, connector, compartmentVariable, channel,
                ic2chan[_compartmentVariableTypes[ctiter->second]]);
        // connect to channel that use the compartment variable as
        // 'compartment[]'
        connect(sim, connector, compartmentVariable, channel,
                cpt2chan[_compartmentVariableTypes[ctiter->second]]);
      }
    }
#endif

    iter = _channelBranchIndices2[ctype].begin(),
    end = _channelBranchIndices2[ctype].end();
    //channel affects branch-CB
#ifdef MICRODOMAIN_CALCIUM
    for (int j = 0; iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::tuple<int, int, std::string > >& channelBranchIndexPairs = (*iter);
      std::vector<std::tuple<int, int, std::string > >::iterator
          ctiter = channelBranchIndexPairs.begin(),
          ctend = channelBranchIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(std::get<1>(*ctiter) < _compartmentVariableTypesMap.size());
        compartmentVariable =
            compartmentVariableAccessors[std::get<1>(*ctiter)]->getNodeDescriptor(
                i, std::get<0>(*ctiter));
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        if (std::get<2>(*ctiter).empty())
        {//well-mixed region
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
          connect(sim, connector, channel, compartmentVariable,
              chan2cpt[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
        }
        else{//if a branch (e.g. Calcium) that allows microdomain
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
          NDPairList Mchan2cptMicrodomain = chan2cptMicrodomain[_compartmentVariableTypes[std::get<1>(*ctiter)]];
          Mchan2cptMicrodomain.replace("domainName", std::get<2>(*ctiter));
          connect(sim, connector, channel, compartmentVariable,
              Mchan2cptMicrodomain);
        }
      }
    }
#else
    for (int j = 0; iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::pair<int, int> >& channelBranchIndexPairs = (*iter);
      std::vector<std::pair<int, int> >::iterator
          ctiter = channelBranchIndexPairs.begin(),
          ctend = channelBranchIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(ctiter->second < _compartmentVariableTypesMap.size());
        compartmentVariable =
            compartmentVariableAccessors[ctiter->second]->getNodeDescriptor(
                i, ctiter->first);
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        connect(sim, connector, compartmentVariable, channel,
                ic2chan[_compartmentVariableTypes[ctiter->second]]);
        connect(sim, connector, channel, compartmentVariable,
                chan2cpt[_compartmentVariableTypes[ctiter->second]]);
      }
    }
#endif

    iter = _channelJunctionIndices1[ctype].begin(),
    end = _channelJunctionIndices1[ctype].end();
#ifdef MICRODOMAIN_CALCIUM
    for (int j = _channelBranchIndices1[ctype].size(); iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::tuple<int, int, std::string> >& channelJunctionIndexPairs = (*iter);
      std::vector<std::tuple<int, int, std::string> >::iterator ctiter, ctend;
      ctiter = channelJunctionIndexPairs.begin();
      ctend = channelJunctionIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(std::get<1>(*ctiter) < _compartmentVariableTypesMap.size());
        compartmentVariable =
            junctionAccessors[std::get<1>(*ctiter)]->getNodeDescriptor(i,
                                                                 std::get<0>(*ctiter));
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        if (std::get<2>(*ctiter).empty())
        {//well-mixed region
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
          connect(sim, connector, compartmentVariable, channel,
              cpt2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);

        }
        else{//if a junction (e.g. Calcium) that allows microdomain
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);

          NDPairList McptMicrodomain2chan = cptMicrodomain2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]];
          McptMicrodomain2chan.replace("domainName", std::get<2>(*ctiter));
          connect(sim, connector, compartmentVariable, channel,
              McptMicrodomain2chan);
        }
      }
    }
#else
    for (int j = _channelBranchIndices1[ctype].size(); iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::pair<int, int> >& channelJunctionIndexPairs = (*iter);
      std::vector<std::pair<int, int> >::iterator ctiter, ctend;
      ctiter = channelJunctionIndexPairs.begin();
      ctend = channelJunctionIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(ctiter->second < _compartmentVariableTypesMap.size());
        compartmentVariable =
            junctionAccessors[ctiter->second]->getNodeDescriptor(i,
                                                                 ctiter->first);
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        connect(sim, connector, compartmentVariable, channel,
                ic2chan[_compartmentVariableTypes[ctiter->second]]);
        connect(sim, connector, compartmentVariable, channel,
                cpt2chan[_compartmentVariableTypes[ctiter->second]]);
      }
    }
#endif
    //channel affects junctions-CB
    iter = _channelJunctionIndices2[ctype].begin(),
    end = _channelJunctionIndices2[ctype].end();
#ifdef MICRODOMAIN_CALCIUM
    for (int j = _channelBranchIndices2[ctype].size(); iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::tuple<int, int, std::string > >& channelJunctionIndexPairs = (*iter);
      std::vector<std::tuple<int, int, std::string > >::iterator
          ctiter = channelJunctionIndexPairs.begin(),
          ctend = channelJunctionIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(std::get<1>(*ctiter) < _compartmentVariableTypesMap.size());
        compartmentVariable =
            junctionAccessors[std::get<1>(*ctiter)]->getNodeDescriptor(i,
                                                                 std::get<0>(*ctiter));
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        if (std::get<2>(*ctiter).empty())
        {//well-mixed region
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
          connect(sim, connector, channel, compartmentVariable,
              chan2cpt[_compartmentVariableTypes[std::get<1>(*ctiter)]]);
        }
        else{//if a junction (e.g. Calcium) that allows microdomain
          connect(sim, connector, compartmentVariable, channel,
              ic2chan[_compartmentVariableTypes[std::get<1>(*ctiter)]]);

          NDPairList Mchan2cptMicrodomain = chan2cptMicrodomain[_compartmentVariableTypes[std::get<1>(*ctiter)]];
          Mchan2cptMicrodomain.replace("domainName", std::get<2>(*ctiter));
          connect(sim, connector, channel,compartmentVariable, 
              Mchan2cptMicrodomain);
        }
      }
    }
#else
    for (int j = _channelBranchIndices2[ctype].size(); iter != end; ++iter, ++j)
    {
      NodeDescriptor* channel =
          channelAccessors[ctype]->getNodeDescriptor(i, j);
      NodeDescriptor* compartmentVariable = 0;
      std::vector<std::pair<int, int> >& channelJunctionIndexPairs = (*iter);
      std::vector<std::pair<int, int> >::iterator
          ctiter = channelJunctionIndexPairs.begin(),
          ctend = channelJunctionIndexPairs.end();
      for (; ctiter != ctend; ++ctiter)
      {
        assert(ctiter->second < _compartmentVariableTypesMap.size());
        compartmentVariable =
            junctionAccessors[ctiter->second]->getNodeDescriptor(i,
                                                                 ctiter->first);
        assert(compartmentVariable);
        assert(sim->getGranule(*compartmentVariable)->getPartitionId() ==
               _rank);
        connect(sim, connector, compartmentVariable, channel,
                ic2chan[_compartmentVariableTypes[ctiter->second]]);
        connect(sim, connector, channel, compartmentVariable,
                chan2cpt[_compartmentVariableTypes[ctiter->second]]);
      }
    }
#endif
  }

  // GOAL: connect compartment to ForwardSolver and BackwardSolver
  //    Cpt->ForwardSolverPoint
  //    Cpt->BackwardSolverPoint
  // HOW: traverse through all grid's node
  // .. for each compartment variable (i.e. Calcium, Voltage) on that gridnode
  // .....traverse through all index in the density
  // .....(for each index) get the ComputeBranch
  // NOTE: Each ComputeBranch belong to the branch of a given compute-order
  //      from 0 to MAX_COMPUTE_ORDER-1
  //  and make the junction of this branch is implicit or explicit junction
  //  based on the branch's computeOrder
  //  GOAL: connect every compartment variables with the proper forward-solver
  //  and backward-solver
  for (int i = 0; i < _nbrGridNodes; ++i)
  {
    std::map<std::string, int>::iterator cptVarTypesIter,
        cptVarTypesEnd = _compartmentVariableTypesMap.end();
    for (cptVarTypesIter = _compartmentVariableTypesMap.begin();
         cptVarTypesIter != cptVarTypesEnd; ++cptVarTypesIter)
    {
      std::string cptVarType = cptVarTypesIter->first;
      int cptVarTypeIdx = cptVarTypesIter->second;
      int branchDensity =
          _compartmentVariableLayers[cptVarTypeIdx]->getDensity(i);
      for (int j = 0; j < branchDensity; ++j)
      {
        ComputeBranch* br = findBranch(i, j, cptVarType);
        key_size_t key = br->_capsules[0].getKey();
        int computeOrder = _segmentDescriptor.getComputeOrder(key);

        if (br->_daughters.size() > 0 && computeOrder != MAX_COMPUTE_ORDER)
        {  // connect the compartment variable to NodeType forward-solver
          NodeDescriptor* compartmentVariable =
              compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(i,
                                                                             j);
          std::list<ComputeBranch*>::iterator diter,
              dend = br->_daughters.end();
          for (diter = br->_daughters.begin(); diter != dend; ++diter)
          {  // connect the compartment variable to forward-solver of proper
             // computeOrder
            assert(_tissueContext->getPass((*diter)->_capsules->getKey()) ==
                   TissueContext::FIRST_PASS);
            std::vector<int>& forwardSolvePointIndices =
                findForwardSolvePointIndices(*diter, cptVarType);
            // This is hard! If you want to change it, you better be sure...
            NodeDescriptor* forwardSolvePoint =
                forwardSolvePointAccessors
                    [_forwardSolvePointTypesMap[computeOrder + 1][cptVarType]]
                        ->getNodeDescriptor(forwardSolvePointIndices[0],
                                            forwardSolvePointIndices[1]);
            connect(sim, connector, forwardSolvePoint, compartmentVariable,
                    fwdpt2br);
          }
        }
        if (br->_parent && computeOrder != 0)
        {  // not the branch with COMPUTEORDER==0
          // connect the compartment variable to backward-solver of proper
          // computeOrder
          assert(_tissueContext->getPass(br->_parent->_capsules->getKey()) ==
                 TissueContext::FIRST_PASS);
          NodeDescriptor* compartmentVariable =
              compartmentVariableAccessors[cptVarTypeIdx]->getNodeDescriptor(i,
                                                                             j);
          assert(computeOrder ==
                 _segmentDescriptor.getComputeOrder(
                     br->_parent->_capsules[0].getKey()) +
                     1);
          std::vector<int>& backwardSolvePointIndices =
              findBackwardSolvePointIndices(br->_parent, cptVarType);
          // This is hard! If you want to change it, you better be sure...
          NodeDescriptor* backwardSolvePoint =
              backwardSolvePointAccessors
                  [_backwardSolvePointTypesMap[computeOrder - 1][cptVarType]]
                      ->getNodeDescriptor(backwardSolvePointIndices[0],
                                          backwardSolvePointIndices[1]);
          connect(sim, connector, backwardSolvePoint, compartmentVariable,
                  bwdpt2br);
        }
      }
    }
  }

  std::vector<std::map<int, int> > electricalSynapseCounters,
      bidirectionalConnectionCounters, chemicalSynapseCounters;
  electricalSynapseCounters.resize(_electricalSynapseTypeCounter);
  bidirectionalConnectionCounters.resize(_bidirectionalConnectionTypeCounter);
  chemicalSynapseCounters.resize(_chemicalSynapseTypeCounter);
  // GOAL: 1. connect connexon (chemicalSynapse) to compartment variable
  //      2. connect spineneck-dendrite         to compartment variable
  //      3. connect bouton-compartment to preSynapticPoint or synapticCleft variable
  // HOW: traverse all touches, identify if the touch is 
  //      1. spineless chemicalSynapse (::touchIsChemicalSynapse)
  //      2. bidirectional (spineneck-denshaft attachment)
  //      3. spine chemicalSynapse (::touchIsChemicalSynapse)
  if (1)  // just for grouping long-code
  {
    TouchVector::TouchIterator titer = _tissueContext->_touchVector.begin(),
                               tend = _tissueContext->_touchVector.end();
    // loop through all detected touches 
    // (supposed to be handled by this MPI process)
    //  determine if the touch belong to
    //      1. chemicalsynapse, and if so, should we make it a chemical synapse
    //      2. electricalsynapse, and if so, should we make it bidirectional
    //      synapse, i.e. bidirecitonal flow
    for (; titer != tend; ++titer)
    {
      //Check to make sure only consider the touch with at least 
      //one capsule supposed to be handled
      //by the current MPI process
      if (!_tissueContext->isLensTouch(*titer, _rank)) continue;
      key_size_t key1, key2;
      key1 = titer->getKey1();
      key2 = titer->getKey2();

      Capsule* preCapsule =
          &_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key1)];
      Capsule* postCapsule =
          &_tissueContext->_capsules[_tissueContext->getCapsuleIndex(key2)];
      bool preJunction, postJunction;  // preJunction = check if the preCapsule
                                       // is part of a Junction or not
      int indexPre, indexPost;  // GSL Grid's index at which preCapsule
      // and postCapsule belongs to, respectively [ NOTE: Grid's index = MPI rank ]

#ifdef IDEA1
      Capsule* jctCapsulePreCapsule = preCapsule;
      Capsule* jctCapsulePostCapsule = postCapsule;
      if (_tissueContext->isPartOfExplicitJunction(*preCapsule, *titer, indexPre, &jctCapsulePreCapsule))
      {
          //if (jctCapsulePreCapsule == NULL)
          //{
          //    assert(indexPre != _rank);
          //    continue;
          //}
          //if (indexPre != _rank)
          //  continue;
          preJunction = true;
      }
#else
      if (_segmentDescriptor.getFlag(key1) &&
          _tissueContext->isTouchToEnd(*preCapsule, *titer))
      {
        preJunction = true;
        indexPre = _tissueContext->getRankOfEndPoint(preCapsule->getBranch());
      }
#endif
      else
      {
        preJunction = false;
        indexPre = _tissueContext->getRankOfBeginPoint(preCapsule->getBranch());
      }

#ifdef IDEA1
      if (_tissueContext->isPartOfExplicitJunction(*postCapsule, *titer, indexPost, &jctCapsulePostCapsule))
      {
          //if (jctCapsulePostCapsule == NULL)
          //{
          //    assert(indexPost != _rank);
          //    continue;
          //}
          //if (indexPost != _rank)
          //  continue;
          postJunction = true;
      }
#else
      if (_segmentDescriptor.getFlag(key2) &&
          _tissueContext->isTouchToEnd(*postCapsule, *titer))
      {
        postJunction = true;
        indexPost = _tissueContext->getRankOfEndPoint(postCapsule->getBranch());
      }
#endif
      else
      {
        postJunction = false;
        indexPost =
            _tissueContext->getRankOfBeginPoint(postCapsule->getBranch());
      }

      //TUAN can be removed
      assert((indexPre == _rank) or (indexPost == _rank));

      std::list<Params::ElectricalSynapseTarget>* esynTargets =
          _tissueParams.getElectricalSynapseTargets(key1, key2);
      if (esynTargets)
      {  // touch falls into electrical-synapse group
        std::list<Params::ElectricalSynapseTarget>::iterator esiter,
            esend = esynTargets->end();
        std::vector<int> typeCounter;
        typeCounter.resize(_electricalSynapseTypesMap.size(), 0);
        for (esiter = esynTargets->begin(); esiter != esend; ++esiter)
        {
          int synapseType = _electricalSynapseTypesMap[esiter->_type];
          if (isGenerated(_generatedElectricalSynapses, titer, synapseType,
                          typeCounter[synapseType]))
          {
            std::map<int, int>& ecounts =
                electricalSynapseCounters[synapseType];
            int preDI = getCountAndIncrement(ecounts, indexPre);
            int postDI = getCountAndIncrement(ecounts, indexPost);
            std::list<std::string>::iterator etiter = esiter->_target.begin(),
                                             etend = esiter->_target.end();
            for (; etiter != etend; ++etiter)
            {
              NodeDescriptor* preCpt = 0;
              int preIdx = 0;
              if (preJunction)
              {
#ifdef IDEA1
                  //NOTE: use jctCapsulePreCapsule
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(jctCapsulePreCapsule);
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                preCpt = junctionAccessors
                             [cptVarJctTypeMap
                                  [_compartmentVariableTypesMap[*etiter]]]
                                 ->getNodeDescriptor(junctionIndices[0],
                                                     junctionIndices[1]);
#else
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(preCapsule);
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                preCpt = junctionAccessors
                             [cptVarJctTypeMap
                                  [_compartmentVariableTypesMap[*etiter]]]
                                 ->getNodeDescriptor(junctionIndices[0],
                                                     junctionIndices[1]);
#endif
              }
              else
              {
                std::vector<int>& branchIndices =
                    findBranchIndices(preCapsule->getBranch(), *etiter);
                preCpt = compartmentVariableAccessors
                             [_compartmentVariableTypesMap[*etiter]]
                                 ->getNodeDescriptor(branchIndices[0],
                                                     branchIndices[1]);
                /*preIdx = getCptIndex(preCapsule) preIdx =
                    N_COMPARTMENTS(preCapsule->getBranch()->_nCapsules) -
                    ((preCapsule - preCapsule->getBranch()->_capsules) /
                     _compartmentSize) -
                    1;
                                                                */
#ifdef IDEA1
                preIdx = _tissueContext->getCptIndex(preCapsule, *titer);
#else
                preIdx = getCptIndex(preCapsule);
#endif
              }

              // bool electrical, chemical, generated;

              NodeDescriptor* postCpt = 0;
              int postIdx = 0;
              if (postJunction)
              {
#ifdef IDEA1
                  //NOTE: use jctCapsulePostCapsule
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(jctCapsulePostCapsule);
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                postCpt = junctionAccessors
                              [cptVarJctTypeMap
                                   [_compartmentVariableTypesMap[*etiter]]]
                                  ->getNodeDescriptor(junctionIndices[0],
                                                      junctionIndices[1]);
#else
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(postCapsule);
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                postCpt = junctionAccessors
                              [cptVarJctTypeMap
                                   [_compartmentVariableTypesMap[*etiter]]]
                                  ->getNodeDescriptor(junctionIndices[0],
                                                      junctionIndices[1]);
#endif
              }
              else
              {
                std::vector<int>& branchIndices =
                    findBranchIndices(postCapsule->getBranch(), *etiter);
                postCpt = compartmentVariableAccessors
                              [_compartmentVariableTypesMap[*etiter]]
                                  ->getNodeDescriptor(branchIndices[0],
                                                      branchIndices[1]);
                /*postIdx = N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules)
                   -
                          ((postCapsule - postCapsule->getBranch()->_capsules) /
                           _compartmentSize) -
                          1;
                                                                */
#ifdef IDEA1
                postIdx = _tissueContext->getCptIndex(postCapsule, *titer);
#else
                postIdx = getCptIndex(postCapsule);
#endif
              }

              NodeDescriptor* preConnexon =
                  electricalSynapseAccessors[synapseType]->getNodeDescriptor(
                      indexPre, preDI);

              NodeDescriptor* postConnexon =
                  electricalSynapseAccessors[synapseType]->getNodeDescriptor(
                      indexPost, postDI);

              NDPairList Mcpt2syn = cpt2syn[*etiter];
              NDPairList Mesyn2cpt = esyn2cpt[*etiter];
              NDPairList Mcnnxn2cnnxn = cnnxn2cnnxn[*etiter];

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

      std::list<Params::BidirectionalConnectionTarget>* spineattachTargets =
          _tissueParams.getBidirectionalConnectionTargets(key1, key2);
      if (spineattachTargets)
      {  // touch falls into spine-attachment group
        std::list<Params::BidirectionalConnectionTarget>::iterator bditer,
            bdend = spineattachTargets->end();
        std::vector<int> typeCounter;
        typeCounter.resize(_bidirectionalConnectionTypesMap.size(), 0);
        for (bditer = spineattachTargets->begin(); bditer != bdend; ++bditer)
        {
          int synapseType = _bidirectionalConnectionTypesMap[bditer->_type];
          if (isGenerated(_generatedBidirectionalConnections, titer,
                          synapseType, typeCounter[synapseType]))
          {
            std::map<int, int>& ecounts =
                bidirectionalConnectionCounters[synapseType];
            int preDI = getCountAndIncrement(ecounts, indexPre);
            int postDI = getCountAndIncrement(ecounts, indexPost);

            // list of compartment-node's name (e.g. 'Voltage', 'Calcium')
            // that are supposed to pass through
            std::list<std::string>::iterator etiter = bditer->_target.begin(),
                                             etend = bditer->_target.end();
            for (; etiter != etend; ++etiter)
            {
              NodeDescriptor* preCpt = 0;
              int preIdx = 0;
              if (preJunction)
              {
#ifdef IDEA1
                  assert((jctCapsulePreCapsule));
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(jctCapsulePreCapsule);
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                preCpt = junctionAccessors
                             [cptVarJctTypeMap
                                  [_compartmentVariableTypesMap[*etiter]]]
                                 ->getNodeDescriptor(junctionIndices[0],
                                                     junctionIndices[1]);
#else
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(preCapsule);
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                preCpt = junctionAccessors
                             [cptVarJctTypeMap
                                  [_compartmentVariableTypesMap[*etiter]]]
                                 ->getNodeDescriptor(junctionIndices[0],
                                                     junctionIndices[1]);
#endif
              }
              else
              {
                std::vector<int>& branchIndices =
                    findBranchIndices(preCapsule->getBranch(), *etiter);
                preCpt = compartmentVariableAccessors
                             [_compartmentVariableTypesMap[*etiter]]
                                 ->getNodeDescriptor(branchIndices[0],
                                                     branchIndices[1]);
                /*preIdx = getCptIndex(preCapsule) preIdx =
                    N_COMPARTMENTS(preCapsule->getBranch()->_nCapsules) -
                    ((preCapsule - preCapsule->getBranch()->_capsules) /
                     _compartmentSize) -
                    1;
                                                                */
#ifdef IDEA1
                preIdx = _tissueContext->getCptIndex(preCapsule, *titer);
#else
                preIdx = getCptIndex(preCapsule);
#endif
              }


              NodeDescriptor* postCpt = 0;
              int postIdx = 0;
#ifdef IDEA1
              if (postJunction and jctCapsulePostCapsule)
              //if (postJunction and indexPost == _rank)
              {
                //TUAN NOTICE
                //TUAN TODO
                //The problem is the children has no access (i.e. information)
                //about the parent 
                //the same problem to preJunction 
                //and not only for bidirectional but also for (point) (chemical) and (electrical)
                  assert((jctCapsulePostCapsule));
                  //assert(indexPost == _rank);
                  //if (indexPost != _rank)
                  //    continue;
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(jctCapsulePostCapsule);
                //if (jmapiter2 == jmapiter1->second.end())
                //{

                //}
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                postCpt = junctionAccessors
                              [cptVarJctTypeMap
                                   [_compartmentVariableTypesMap[*etiter]]]
                                  ->getNodeDescriptor(junctionIndices[0],
                                                      junctionIndices[1]);
              }
#else
              if (postJunction)
              {
                std::map<std::string,
                         std::map<Capsule*, std::vector<int> > >::iterator
                    jmapiter1 = _junctionIndexMap.find(*etiter);
                assert(jmapiter1 != _junctionIndexMap.end());
                std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                    jmapiter1->second.find(postCapsule);
                assert(jmapiter2 != jmapiter1->second.end());
                std::vector<int>& junctionIndices = jmapiter2->second;
                postCpt = junctionAccessors
                              [cptVarJctTypeMap
                                   [_compartmentVariableTypesMap[*etiter]]]
                                  ->getNodeDescriptor(junctionIndices[0],
                                                      junctionIndices[1]);
              }
#endif
              else
              {
                std::vector<int>& branchIndices =
                    findBranchIndices(postCapsule->getBranch(), *etiter);
                postCpt = compartmentVariableAccessors
                              [_compartmentVariableTypesMap[*etiter]]
                                  ->getNodeDescriptor(branchIndices[0],
                                                      branchIndices[1]);
                /*postIdx = N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules)
                   -
                          ((postCapsule - postCapsule->getBranch()->_capsules) /
                           _compartmentSize) -
                          1;
                                                                */
#ifdef IDEA1
                postIdx = _tissueContext->getCptIndex(postCapsule, *titer);
#else
                postIdx = getCptIndex(postCapsule);
#endif
              }

              NodeDescriptor* preSpineConnexon =
                  bidirectionalConnectionAccessors[synapseType]
                      ->getNodeDescriptor(indexPre, preDI);

              NodeDescriptor* postSpineConnexon =
                  bidirectionalConnectionAccessors[synapseType]
                      ->getNodeDescriptor(indexPost, postDI);

              NDPairList Mcpt2spineattach = cpt2spineattach[*etiter];
              NDPairList Mspineattach2cpt = spineattach2cpt[*etiter];
              NDPairList Mspineattach2spineattach =
                  spineattach2spineattach[*etiter];

              key_size_t keyneck = 0;
#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
              (*titer).hasSpineNeck(keyneck, _tissueParams);
#else
              (*titer).hasSpineNeck(keyneck);//obsolete
#endif
              Capsule* capsneck = (keyneck == key1) ? preCapsule : postCapsule;
              assert((keyneck == key1) || keyneck == key2);

#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
              //order is important:
              //preCpt - spineattach1
              //postCpt - spineattach2
              //spineattach1-spineattach2
              //spineattach2-spineattach1
              //spineattach1 - preCpt
              //spineattach2 - postCpt
              {
              if (keyneck == key1)
              {
                std::string name("typeCpt");
                std::string value("spine-neck");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "spine-neck"));
              }
              else
              {
                std::string name("typeCpt");
                std::string value("den-shaft");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "den-shaft"));
              }

              Mcpt2spineattach.replace("idx", preIdx);
              connect(sim, connector, preCpt, preSpineConnexon,
                      Mcpt2spineattach);
                
              }
              {
              if (keyneck == key2)
              {
                std::string name("typeCpt");
                std::string value("spine-neck");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "spine-neck"));
              }
              else
              {
                std::string name("typeCpt");
                std::string value("den-shaft");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "den-shaft"));
              }

              Mcpt2spineattach.replace("idx", postIdx);
              connect(sim, connector, postCpt, postSpineConnexon,
                      Mcpt2spineattach);

              }
              {
              //connect 2 spineattachment nodes together
              connect(sim, connector, preSpineConnexon, postSpineConnexon,
                      Mspineattach2spineattach);
              connect(sim, connector, postSpineConnexon, preSpineConnexon,
                      Mspineattach2spineattach);

              }
              {
              if (keyneck == key1)
              {
                std::string name("typeCpt");
                std::string value("spine-neck");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "spine-neck"));
              }
              else
              {
                std::string name("typeCpt");
                std::string value("den-shaft");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "den-shaft"));
              }

              Mspineattach2cpt.replace("idx", preIdx);
              connect(sim, connector, preSpineConnexon, preCpt,
                      Mspineattach2cpt);

              }
              {
              if (keyneck == key2)
              {
                std::string name("typeCpt");
                std::string value("spine-neck");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "spine-neck"));
              }
              else
              {
                std::string name("typeCpt");
                std::string value("den-shaft");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "den-shaft"));
              }

              Mspineattach2cpt.replace("idx", postIdx);
              connect(sim, connector, postSpineConnexon, postCpt,
                      Mspineattach2cpt);
              }

#else
              if (keyneck == key1)
              {
                std::string name("typeCpt");
                std::string value("spine-neck");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "spine-neck"));
              }
              else
              {
                std::string name("typeCpt");
                std::string value("den-shaft");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "den-shaft"));
              }

              Mspineattach2cpt.replace("idx", preIdx);
              connect(sim, connector, preSpineConnexon, preCpt,
                      Mspineattach2cpt);
              Mcpt2spineattach.replace("idx", preIdx);
              connect(sim, connector, preCpt, preSpineConnexon,
                      Mcpt2spineattach);

              if (keyneck == key2)
              {
                std::string name("typeCpt");
                std::string value("spine-neck");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "spine-neck"));
              }
              else
              {
                std::string name("typeCpt");
                std::string value("den-shaft");
                assert(Mcpt2spineattach.replace(name, value));
                // assert(Mcpt2spineattach.replace("typeCpt", "den-shaft"));
              }

              Mspineattach2cpt.replace("idx", postIdx);
              connect(sim, connector, postSpineConnexon, postCpt,
                      Mspineattach2cpt);
              Mcpt2spineattach.replace("idx", postIdx);
              connect(sim, connector, postCpt, postSpineConnexon,
                      Mcpt2spineattach);
              //finally connect 2 spineattachment nodes together
              connect(sim, connector, preSpineConnexon, postSpineConnexon,
                      Mspineattach2spineattach);
              connect(sim, connector, postSpineConnexon, preSpineConnexon,
                      Mspineattach2spineattach);
#endif
            }
          }
          typeCounter[synapseType]++;
        }
      }

      std::vector<NodeDescriptor*> preSynPoints;
      preSynPoints.resize(_chemicalSynapseTypeCounter, 0);

      std::vector<NodeDescriptor*> synapticCleftNodes;
      synapticCleftNodes.resize(_chemicalSynapseTypeCounter, 0);

      std::list<Params::ChemicalSynapseTarget>* csynTargets =
          _tissueParams.getChemicalSynapseTargets(key1, key2);
      if (csynTargets)
      {  // touch falls into chemical-synapse group
        // key1    key2
        // 2 1 0   1 3   [AMPAmush NMDAmush] [Voltage] [Voltage] [Voltage]
        //                                   [Voltage, Calcium]  1.0
        //--> converted into a list, each element is a map as
        //    std::map<std::string, std::pair<std::list<std::string>,
        //                                    std::list<std::string> > >
        //                                    _targets;
        // e.g.: the above example has a list of 2 elements, each element is
        //  <AMPAmush, pair( ["Voltage"], ["Voltage"] )
        //  <NMDAmush, pair( ["Voltage"], ["Voltage", "Calcium"] )
        // NOTE: csiter iterate through the list
        std::list<Params::ChemicalSynapseTarget>::iterator csiter,
            csend = csynTargets->end();
        std::vector<int> typeCounter;
        typeCounter.resize(_chemicalSynapseTypesMap.size(), 0);//number of Layers defined with "ChemicalSynapses"
        bool first_receptor = true;
        for (csiter = csynTargets->begin(); csiter != csend; ++csiter)
        {
          std::map<std::string, std::pair<std::list<std::string>,
                                          std::list<std::string> > >::iterator
              targetsIter,
              targetsEnd = csiter->_targets.end();
          std::vector<NodeDescriptor*> mixedSynapse;
          for (targetsIter = csiter->_targets.begin();
               targetsIter != targetsEnd; ++targetsIter)
          {//loops through 'AMPAmush', 'NMDAmush' (this name is mapped to targetsIter->first)
            std::map<std::string, int>::iterator miter =
                _chemicalSynapseTypesMap.find(targetsIter->first); //GOAL: get to the layer-index for layer of, say 'AMPAmush'
            // miter --> check if there is a layer name, say
            // 'ChemicalSynapses[AMPAmush]'
            assert(miter != _chemicalSynapseTypesMap.end());
            // if present, get the layer index
            int synapseType = miter->second;
            if (isGenerated(_generatedChemicalSynapses, titer, synapseType,
                            typeCounter[synapseType]))
            {
              std::map<int, int>& ccounts =
                  chemicalSynapseCounters[synapseType];
              NodeDescriptor* receptor =
                  chemicalSynapseAccessors[synapseType]->getNodeDescriptor(
                      indexPost, getCountAndIncrement(ccounts, indexPost));
              mixedSynapse.push_back(receptor);
//#define RECEPTOR_PRE_AS_INPUT_POST_AS_INPUT_OUTPUT
#ifdef  RECEPTOR_PRE_AS_INPUT_POST_AS_INPUT_OUTPUT
              // This is designed to work with PreSynapticPoint
              //NOTE: POST = postsynaptic side
              //      PRE  = presynaptic side
              //Here, the line [[original implementation]]
              // AMPAmush [Voltage] [Voltage, Calcium]
              // means [Voltage] <-- from pre-capsule, and play as input 
              // and   [Voltage, Calcium] <-- from post-capsule, and play as both input/output
              
              // Pre
              std::list<std::string>::iterator
                  ctiter = targetsIter->second.first.begin(),
                  ctend = targetsIter->second.first.end();
              for (; ctiter != ctend; ++ctiter)
              {
                NodeDescriptor* preCpt = 0;
                int preIdx = 0;
                if (preJunction)
                {  // presynaptic-compartment is a junction branch
                   // as the junction is always a single-compartment structure
                   //  the preIdx = 0 always
#ifdef IDEA1
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(*ctiter);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(jctCapsulePreCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  preCpt = junctionAccessors
                               [cptVarJctTypeMap
                                    [_compartmentVariableTypesMap[*ctiter]]]
                                   ->getNodeDescriptor(junctionIndices[0],
                                                       junctionIndices[1]);
#else
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(*ctiter);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(preCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  preCpt = junctionAccessors
                               [cptVarJctTypeMap
                                    [_compartmentVariableTypesMap[*ctiter]]]
                                   ->getNodeDescriptor(junctionIndices[0],
                                                       junctionIndices[1]);
#endif
                }
                else
                {  // presynaptic-compartment is part of a regular branch
                  std::vector<int>& branchIndices =
                      findBranchIndices(preCapsule->getBranch(), *ctiter);
                  preCpt = compartmentVariableAccessors
                               [_compartmentVariableTypesMap[*ctiter]]
                                   ->getNodeDescriptor(branchIndices[0],
                                                       branchIndices[1]);
                  /*preIdx = N_COMPARTMENTS(preCapsule->getBranch()->_nCapsules)
                     -
                           ((preCapsule - preCapsule->getBranch()->_capsules) /
                            _compartmentSize) -
                           1;*/
#ifdef IDEA1
                  preIdx = _tissueContext->getCptIndex(preCapsule, *titer);
#else
                  preIdx = getCptIndex(preCapsule);
#endif
                }
                if (_preSynapticPointTypeCounter > 0)
                {  // PreSynapticPoint layer is used
                  NodeAccessor* preSynapticPointAccessor = 0;
                  std::string preSynapticPointType =
                      _tissueParams.getPreSynapticPointTarget(
                          targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                      _preSynapticPointTypesMap.find(preSynapticPointType);
                  assert(tmapiter != _preSynapticPointTypesMap.end());
                  unsigned int preSynPointType = (tmapiter->second);
                  if (preSynPoints[preSynPointType] == 0)
                  {
                    assert(preSynapticPointAccessors.size() > preSynPointType);
                    preSynapticPointAccessor =
                        preSynapticPointAccessors[preSynPointType];
                    if (preJunction)
                    {
#ifdef IDEA1
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][jctCapsulePreCapsule]);
#else
#ifdef SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][std::make_pair(preCapsule, postCapsule)]);
#else
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][preCapsule]);
#endif

#endif
                    }
                    else
                    {
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleCptPointIndexMap
                                            [preSynapticPointType][preCapsule]);
                    }
                  }
                  NDPairList Mcpt2syn = cpt2syn[*ctiter];
                  Mcpt2syn.replace("idx", preIdx);
                  connect(sim, connector, preCpt, preSynPoints[preSynPointType],
                          Mcpt2syn);
                  connect(sim, connector, preSynPoints[preSynPointType],
                          receptor, presynpt);
                }
                else if (_synapticCleftTypeCounter > 0)
                {  // SynapticCleft layer is used
                  NodeAccessor* synapticCleftAccessor = 0;
                  std::string synapticCleftType =
                      _tissueParams.getPreSynapticPointTarget(
                          targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                      _synapticCleftTypesMap.find(synapticCleftType);
                  assert(tmapiter != _synapticCleftTypesMap.end());
                  unsigned int cleftType = (tmapiter->second);
                  if (synapticCleftNodes[cleftType] == 0)
                  {
                    assert(synapticCleftAccessors.size() > cleftType);
                    synapticCleftAccessor =
                        synapticCleftAccessors[cleftType];
                    if (preJunction)
                    {
#ifdef IDEA1
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [synapticCleftType][jctCapsulePostCapsule]);
#else
#ifdef SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [synapticCleftType][std::make_pair(preCapsule, postCapsule)]);
#else
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [synapticCleftType][preCapsule]);
#endif

#endif
                    }
                    else
                    {
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleCptPointIndexMap
                                            [synapticCleftType][preCapsule]);
                    }
                  }
                  NDPairList Mcpt2cleft = cpt2cleft[*ctiter];
                  Mcpt2cleft.replace("idx", preIdx);
                  Mcpt2cleft.replace("side", "pre");
                  connect(sim, connector, preCpt,
                          synapticCleftNodes[cleftType], Mcpt2cleft);
                  connect(sim, connector, synapticCleftNodes[cleftType],
                          receptor, synCleft);
                }
              }

              // Post
              ctiter = targetsIter->second.second.begin(),
              ctend = targetsIter->second.second.end();
              for (; ctiter != ctend; ++ctiter)
              {
                NodeDescriptor* postCpt = 0;
                int postIdx = 0;
                if (postJunction)
                {
#ifdef IDEA1
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(*ctiter);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(jctCapsulePostCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[*ctiter]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#else
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(*ctiter);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(postCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[*ctiter]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#endif
                }
                else
                {
                  std::vector<int>& branchIndices =
                      findBranchIndices(postCapsule->getBranch(), *ctiter);
                  postCpt = compartmentVariableAccessors
                                [_compartmentVariableTypesMap[*ctiter]]
                                    ->getNodeDescriptor(branchIndices[0],
                                                        branchIndices[1]);
                  /*postIdx =
                      N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules) -
                      ((postCapsule - postCapsule->getBranch()->_capsules) /
                       _compartmentSize) -
                      1;*/
#ifdef IDEA1
                postIdx = _tissueContext->getCptIndex(postCapsule, *titer);
#else
                  postIdx = getCptIndex(postCapsule);
#endif
                }
                //receptor->post
                NDPairList Mcsyn2cpt = csyn2cpt[*ctiter];
                Mcsyn2cpt.replace("idx", postIdx);
                connect(sim, connector, receptor, postCpt, Mcsyn2cpt);

                //post->receptor
                NDPairList Mcpt2syn = cpt2syn[*ctiter];
                Mcpt2syn.replace("idx", postIdx);
                NDPairList Mic2syn = ic2syn[*ctiter];
                Mic2syn.replace("idx", postIdx);
                connect(sim, connector, postCpt, receptor, Mcpt2syn);
                connect(sim, connector, postCpt, receptor, Mic2syn);

                //post->cleft
                if (_synapticCleftTypeCounter > 0)
                {  // SynapticCleft layer is used
                  //TUAN TODO: update when 'tight' and 'bulk'
                  ////with new syntax is used
                  NodeAccessor* synapticCleftAccessor = 0;
                  std::string synapticCleftType =
                    _tissueParams.getPreSynapticPointTarget(
                        targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                    _synapticCleftTypesMap.find(synapticCleftType);
                  assert(tmapiter != _synapticCleftTypesMap.end());
                  unsigned int cleftType = (tmapiter->second);

                  NDPairList Mcpt2cleft = cpt2cleft[*ctiter];
                  Mcpt2cleft.replace("idx", postIdx);
                  Mcpt2cleft.replace("side", "post");
                  connect(sim, connector, postCpt,
                      synapticCleftNodes[cleftType], Mcpt2cleft);
                }  
              }
#else //RECEPTOR_POST_AS_INPUT_POST_AS_OUTPUT
              //NOTE: POST = postsynaptic side
  //NOTE: In this mode, the receptor receives the proper Neurotransmitter
  //from the SynapticCleft
  //and compartment name such as Voltage, Calcium always refers to post-side
              //Here, the line [[new implementation]]
              // AMPAmush [Voltage] [Voltage, Calcium]
              // means [Voltage] <-- from post-capsule, and play as input 
              // and   [Voltage, Calcium] <-- from post-capsule, and play as output
              
              // Pre - as input to the cleft/presynapticPoint
              std::list<std::string> preData;
              //TUAN TODO : add a section to define what pre-data to be obsorved by 
              //the PreSynapticPoint or SynapticCleft
              //Here we assume always 'Voltage' only
              //PLAN: We will use 'Calcium' when 'Calcium' is
              //      used to calculate NT release
              preData.push_front("Voltage");
              //PLAN --> preData.push_front("Calcium");
              std::list<std::string>::iterator
                  ctiter = preData.begin(),
                  ctend = preData.end();
              for (; first_receptor && ctiter != ctend; ++ctiter)
              {//Pre-compartment(presume only Voltage-pre) project to SynapticCleft/PreSynapticPoint
                NodeDescriptor* preCpt = 0;
                int preIdx = 0;
                if (preJunction)
                {  // presynaptic-compartment is a junction branch
                   // as the junction is always a single-compartment structure
                   //  the preIdx = 0 always
#ifdef IDEA1
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(*ctiter);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(jctCapsulePreCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  preCpt = junctionAccessors
                               [cptVarJctTypeMap
                                    [_compartmentVariableTypesMap[*ctiter]]]
                                   ->getNodeDescriptor(junctionIndices[0],
                                                       junctionIndices[1]);
#else
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(*ctiter);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(preCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  preCpt = junctionAccessors
                               [cptVarJctTypeMap
                                    [_compartmentVariableTypesMap[*ctiter]]]
                                   ->getNodeDescriptor(junctionIndices[0],
                                                       junctionIndices[1]);
#endif
                }
                else
                {  // presynaptic-compartment is part of a regular branch
                  std::vector<int>& branchIndices =
                      findBranchIndices(preCapsule->getBranch(), *ctiter);
                  preCpt = compartmentVariableAccessors
                               [_compartmentVariableTypesMap[*ctiter]]
                                   ->getNodeDescriptor(branchIndices[0],
                                                       branchIndices[1]);
                  /*preIdx = N_COMPARTMENTS(preCapsule->getBranch()->_nCapsules)
                     -
                           ((preCapsule - preCapsule->getBranch()->_capsules) /
                            _compartmentSize) -
                           1;*/
#ifdef IDEA1
                  preIdx = _tissueContext->getCptIndex(preCapsule, *titer);
#else
                  preIdx = getCptIndex(preCapsule);
#endif
                }
                if (_preSynapticPointTypeCounter > 0)
                {  // PreSynapticPoint layer is used
                  NodeAccessor* preSynapticPointAccessor = 0;
                  std::string preSynapticPointType =
                      _tissueParams.getPreSynapticPointTarget(
                          targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                      _preSynapticPointTypesMap.find(preSynapticPointType);
                  assert(tmapiter != _preSynapticPointTypesMap.end());
                  unsigned int preSynPointType = (tmapiter->second);
                  if (preSynPoints[preSynPointType] == 0)
                  {
                    assert(preSynapticPointAccessors.size() > preSynPointType);
                    preSynapticPointAccessor =
                        preSynapticPointAccessors[preSynPointType];
                    if (preJunction)
                    {
#ifdef IDEA1
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][jctCapsulePreCapsule]);
#else
#ifdef SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][std::make_pair(preCapsule, postCapsule)]);
#else
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][preCapsule]);
#endif

#endif
                    }
                    else
                    {
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleCptPointIndexMap
                                            [preSynapticPointType][preCapsule]);
                    }
                  }
                  NDPairList Mcpt2syn = cpt2syn[*ctiter];
                  Mcpt2syn.replace("idx", preIdx);
                  connect(sim, connector, preCpt, preSynPoints[preSynPointType],
                          Mcpt2syn);
                }
                else if (_synapticCleftTypeCounter > 0)
                {  // SynapticCleft layer is used
                  NodeAccessor* synapticCleftAccessor = 0;
                  std::string synapticCleftType =
                      _tissueParams.getPreSynapticPointTarget(
                          targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                      _synapticCleftTypesMap.find(synapticCleftType);
                  assert(tmapiter != _synapticCleftTypesMap.end());
                  unsigned int cleftType = (tmapiter->second);
                  if (synapticCleftNodes[cleftType] == 0)
                  {
                    assert(synapticCleftAccessors.size() > cleftType);
                    synapticCleftAccessor =
                        synapticCleftAccessors[cleftType];
                    if (preJunction)
                    {
#ifdef IDEA1
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [synapticCleftType][jctCapsulePostCapsule]);
#else
#ifdef SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [synapticCleftType][std::make_pair(preCapsule, postCapsule)]);
#else
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [synapticCleftType][preCapsule]);
#endif

#endif
                    }
                    else
                    {
                      synapticCleftNodes[cleftType] =
                          synapticCleftAccessor->getNodeDescriptor(
                              indexPre, _capsuleCptPointIndexMap
                                            [synapticCleftType][preCapsule]);
                    }
                  }
                  NDPairList Mcpt2cleft = cpt2cleft[*ctiter];
                  Mcpt2cleft.replace("idx", preIdx);
                  Mcpt2cleft.replace("side", "pre");
                  connect(sim, connector, preCpt,
                          synapticCleftNodes[cleftType], Mcpt2cleft);
                }
              }

              // Post - as input to cleft/presynapticPoint
              // IMPORTANT: This is not about passing voltage/calcium data, it pass 'information'
              //            about 'compartment' on post-side
              //            --> so need only once
              //  (only Voltage is being used and this is enforced by SynapticCleft model)
              //  --> so it won't work if a receptor doesn't have Voltage as input
              //std::list<std::string>::iterator
              ctiter = targetsIter->second.first.begin(),
                     ctend = targetsIter->second.first.end();
              for (; first_receptor && ctiter != ctend; ++ctiter)
              {//input to cleft/presynapticPoint
                //NOTE: As the name may contains domain names
                // Calcium(domain1, domainA)
                // we need to split them
                std::string microdomainName("");
#ifdef MICRODOMAIN_CALCIUM
                std::string compartmentNameWithOptionalMicrodomainName(*ctiter);
                std::string compartmentNameOnly("");
                Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
#else
                std::string compartmentNameOnly(*ctiter);
#endif
                NodeDescriptor* postCpt = 0;
                int postIdx = 0;
                if (postJunction)
                {
#ifdef IDEA1
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(compartmentNameOnly);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(jctCapsulePostCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[compartmentNameOnly]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#else
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(compartmentNameOnly);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(postCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[compartmentNameOnly]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#endif
                }
                else
                {
                  std::vector<int>& branchIndices =
                      findBranchIndices(postCapsule->getBranch(), compartmentNameOnly);
                  postCpt = compartmentVariableAccessors
                                [_compartmentVariableTypesMap[compartmentNameOnly]]
                                    ->getNodeDescriptor(branchIndices[0],
                                                        branchIndices[1]);
                  /*postIdx =
                      N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules) -
                      ((postCapsule - postCapsule->getBranch()->_capsules) /
                       _compartmentSize) -
                      1;*/
#ifdef IDEA1
                  postIdx = _tissueContext->getCptIndex(postCapsule, *titer);
#else
                  postIdx = getCptIndex(postCapsule);
#endif
                }

                //post->cleft
                if (_synapticCleftTypeCounter > 0)
                {  // SynapticCleft layer is used
                  //TUAN TODO: update when 'tight' and 'bulk'
                  ////with new syntax is used
                  NodeAccessor* synapticCleftAccessor = 0;
                  std::string synapticCleftType =
                    _tissueParams.getPreSynapticPointTarget(
                        targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                    _synapticCleftTypesMap.find(synapticCleftType);
                  assert(tmapiter != _synapticCleftTypesMap.end());
                  unsigned int cleftType = (tmapiter->second);

                  NDPairList Mcpt2cleft = cpt2cleft[compartmentNameOnly];
                  Mcpt2cleft.replace("idx", postIdx);
                  Mcpt2cleft.replace("side", "post");
                  connect(sim, connector, postCpt,
                      synapticCleftNodes[cleftType], Mcpt2cleft);
                }  
              }


              
              {//connect synapticCleft/preSynapticPoint to all receptors 
                if (_preSynapticPointTypeCounter > 0)
                {  // PreSynapticPoint layer is used
                  NodeAccessor* preSynapticPointAccessor = 0;
                  std::string preSynapticPointType =
                      _tissueParams.getPreSynapticPointTarget(
                          targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                      _preSynapticPointTypesMap.find(preSynapticPointType);
                  assert(tmapiter != _preSynapticPointTypesMap.end());
                  unsigned int preSynPointType = (tmapiter->second);
                  if (preSynPoints[preSynPointType] == 0)
                  {
                    assert(preSynapticPointAccessors.size() > preSynPointType);
                    preSynapticPointAccessor =
                        preSynapticPointAccessors[preSynPointType];
                    if (preJunction)
                    {
#ifdef IDEA1
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][jctCapsulePreCapsule]);
#else
#ifdef SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][std::make_pair(preCapsule, postCapsule)]);
#else
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleJctPointIndexMap
                                            [preSynapticPointType][preCapsule]);
#endif

#endif
                    }
                    else
                    {
                      preSynPoints[preSynPointType] =
                          preSynapticPointAccessor->getNodeDescriptor(
                              indexPre, _capsuleCptPointIndexMap
                                            [preSynapticPointType][preCapsule]);
                    }
                  }
                  connect(sim, connector, preSynPoints[preSynPointType],
                          receptor, presynpt);
                }
                else if (_synapticCleftTypeCounter > 0)
                {  // SynapticCleft layer is used
                  //TUAN TODO: update when 'tight' and 'bulk'
                  ////with new syntax is used
                  NodeAccessor* synapticCleftAccessor = 0;
                  std::string synapticCleftType =
                    _tissueParams.getPreSynapticPointTarget(
                        targetsIter->first);
                  std::map<std::string, int>::iterator tmapiter =
                    _synapticCleftTypesMap.find(synapticCleftType);
                  assert(tmapiter != _synapticCleftTypesMap.end());
                  unsigned int cleftType = (tmapiter->second);
                  connect(sim, connector, synapticCleftNodes[cleftType],
                          receptor, synCleft);
                }
              }
              // Post - as input to receptor
              // NMDAR [ Voltage, Calcium ] [...]
              //std::list<std::string>::iterator
              ctiter = targetsIter->second.first.begin(),
                     ctend = targetsIter->second.first.end();
              for (; ctiter != ctend; ++ctiter)
              {//input to receptor
                //NOTE: As the name may contains domain names
                // Calcium(domain1, domainA)
                // we need to split them
                std::string microdomainName("");
#ifdef MICRODOMAIN_CALCIUM
                std::string compartmentNameWithOptionalMicrodomainName(*ctiter);
                std::string compartmentNameOnly("");
                Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
#else
                std::string compartmentNameOnly(*ctiter);
#endif
                NodeDescriptor* postCpt = 0;
                int postIdx = 0;
                if (postJunction)
                {
#ifdef IDEA1
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(compartmentNameOnly);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(jctCapsulePostCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[compartmentNameOnly]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#else
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(compartmentNameOnly);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(postCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[compartmentNameOnly]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#endif
                }
                else
                {
                  std::vector<int>& branchIndices =
                      findBranchIndices(postCapsule->getBranch(), compartmentNameOnly);
                  postCpt = compartmentVariableAccessors
                                [_compartmentVariableTypesMap[compartmentNameOnly]]
                                    ->getNodeDescriptor(branchIndices[0],
                                                        branchIndices[1]);
                  /*postIdx =
                      N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules) -
                      ((postCapsule - postCapsule->getBranch()->_capsules) /
                       _compartmentSize) -
                      1;*/
#ifdef IDEA1
                  postIdx = _tissueContext->getCptIndex(postCapsule, *titer);
#else
                  postIdx = getCptIndex(postCapsule);
#endif
                }

                //post->receptor
#ifdef MICRODOMAIN_CALCIUM
                if (not microdomainName.empty())
                {
                  NDPairList McptMicrodomain2syn = cptMicrodomain2syn[compartmentNameOnly];
                  McptMicrodomain2syn.replace("domainName", microdomainName);
                  McptMicrodomain2syn.replace("idx", postIdx);
                  //NDPairList MicMicrodomain2syn = icMicrodomain2syn[compartmentNameOnly];
                  //MicMicrodomain2csyn.replace("domainName", microdomainName);
                  //MicMicrodomain2syn.replace("idx", postIdx);
                  connect(sim, connector, postCpt, receptor, McptMicrodomain2syn);
                  //connect(sim, connector, postCpt, receptor, MicMicrodomain2syn);
                }
                else{
                  NDPairList Mcpt2syn = cpt2syn[compartmentNameOnly];
                  Mcpt2syn.replace("idx", postIdx);
                  NDPairList Mic2syn = ic2syn[compartmentNameOnly];
                  Mic2syn.replace("idx", postIdx);
                  connect(sim, connector, postCpt, receptor, Mcpt2syn);
                  connect(sim, connector, postCpt, receptor, Mic2syn);
                }
#else
                NDPairList Mcpt2syn = cpt2syn[compartmentNameOnly];
                Mcpt2syn.replace("idx", postIdx);
                NDPairList Mic2syn = ic2syn[compartmentNameOnly];
                Mic2syn.replace("idx", postIdx);
                connect(sim, connector, postCpt, receptor, Mcpt2syn);
                connect(sim, connector, postCpt, receptor, Mic2syn);
#endif

              }
              // Post - receptor output data to Post
              ctiter = targetsIter->second.second.begin(),
              ctend = targetsIter->second.second.end();
              for (; ctiter != ctend; ++ctiter)
              {//receptor output to Post-compartments
                //NOTE: As the name may contains domain names
                // Calcium(domain1, domainA)
                // we need to split them
                std::string microdomainName("");
#ifdef MICRODOMAIN_CALCIUM
                std::string compartmentNameWithOptionalMicrodomainName(*ctiter);
                std::string compartmentNameOnly("");
                Params::separateCompartmentName_and_microdomainName(compartmentNameWithOptionalMicrodomainName, compartmentNameOnly, microdomainName);
                checkValidUseMicrodomain(compartmentNameOnly, microdomainName);
#else
                std::string compartmentNameOnly(*ctiter);
#endif
                
                NodeDescriptor* postCpt = 0;
                int postIdx = 0;
                if (postJunction)
                {
#ifdef IDEA1
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(compartmentNameOnly);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(jctCapsulePostCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[compartmentNameOnly]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#else
                  std::map<std::string,
                           std::map<Capsule*, std::vector<int> > >::iterator
                      jmapiter1 = _junctionIndexMap.find(compartmentNameOnly);
                  assert(jmapiter1 != _junctionIndexMap.end());
                  std::map<Capsule*, std::vector<int> >::iterator jmapiter2 =
                      jmapiter1->second.find(postCapsule);
                  assert(jmapiter2 != jmapiter1->second.end());
                  std::vector<int>& junctionIndices = jmapiter2->second;
                  postCpt = junctionAccessors
                                [cptVarJctTypeMap
                                     [_compartmentVariableTypesMap[compartmentNameOnly]]]
                                    ->getNodeDescriptor(junctionIndices[0],
                                                        junctionIndices[1]);
#endif
                }
                else
                {
                  std::vector<int>& branchIndices =
                      findBranchIndices(postCapsule->getBranch(), compartmentNameOnly);
                  postCpt = compartmentVariableAccessors
                                [_compartmentVariableTypesMap[compartmentNameOnly]]
                                    ->getNodeDescriptor(branchIndices[0],
                                                        branchIndices[1]);
                  /*postIdx =
                      N_COMPARTMENTS(postCapsule->getBranch()->_nCapsules) -
                      ((postCapsule - postCapsule->getBranch()->_capsules) /
                       _compartmentSize) -
                      1;*/
#ifdef IDEA1
                  postIdx = _tissueContext->getCptIndex(postCapsule, *titer);
#else
                  postIdx = getCptIndex(postCapsule);
#endif
                }
                //receptor->post
                NDPairList Mcsyn2cpt = csyn2cpt[compartmentNameOnly];
                Mcsyn2cpt.replace("idx", postIdx);
#ifdef MICRODOMAIN_CALCIUM
                if (not microdomainName.empty())
                {
                  NDPairList Mcsyn2cptMicrodomain = csyn2cptMicrodomain[compartmentNameOnly];
                  Mcsyn2cptMicrodomain.replace("domainName", microdomainName);
                  connect(sim, connector, receptor, postCpt, Mcsyn2cptMicrodomain);
                }
                else
                {
                  connect(sim, connector, receptor, postCpt, Mcsyn2cpt);
                }
#else
                connect(sim, connector, receptor, postCpt, Mcsyn2cpt);
#endif
              }
#endif
              first_receptor = false;
            }
            typeCounter[synapseType]++;
          }
          for (int i = 0; i < mixedSynapse.size(); ++i)
          {//this is for plasticity
            for (int j = 0; j < mixedSynapse.size(); ++j)
            {
              if (i != j)
                connect(sim, connector, mixedSynapse[i], mixedSynapse[j],
                        recp2recp);
            }
          }
        }
      }
    }
  }

  if (sim->isSimulatePass())
  {  // print-out information about how many instances are created 
    CountableModel* countableModel = 0;

    std::vector<GridLayerDescriptor*>::iterator layerIter, layerEnd;
    std::list<CountableModel*> models;

    layerEnd = _compartmentVariableLayers.end();
    for (layerIter = _compartmentVariableLayers.begin(); layerIter != layerEnd;
         ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();

    layerEnd = _junctionLayers.end();
    for (layerIter = _junctionLayers.begin(); layerIter != layerEnd;
         ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();

    layerEnd = _channelLayers.end();
    for (layerIter = _channelLayers.begin(); layerIter != layerEnd; ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();

    layerEnd = _electricalSynapseLayers.end();
    for (layerIter = _electricalSynapseLayers.begin(); layerIter != layerEnd;
         ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();

    layerEnd = _chemicalSynapseLayers.end();
    for (layerIter = _chemicalSynapseLayers.begin(); layerIter != layerEnd;
         ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();

    layerEnd = _bidirectionalConnectionLayers.end();
    for (layerIter = _bidirectionalConnectionLayers.begin();
         layerIter != layerEnd; ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();

    layerEnd = _synapticCleftLayers.end();
    for (layerIter = _synapticCleftLayers.begin(); layerIter != layerEnd;
         ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();

    layerEnd = _preSynapticPointLayers.end();
    for (layerIter = _preSynapticPointLayers.begin(); layerIter != layerEnd;
         ++layerIter)
    {
      countableModel =
          dynamic_cast<CountableModel*>((*layerIter)->getNodeType());
      if (countableModel)
      {
        if (find(models.begin(), models.end(), countableModel) == models.end())
        {
          countableModel->count();
          models.push_back(countableModel);
        }
      }
    }
    models.clear();
  }
#ifdef DEBUG_CPTS
  //surfaceArea
  std::pair<float, float> result,result2;
  float minVal, maxVal;
  int brType;
  std::cerr << "Compartments AXON: \n" ;
  brType = Branch::_AXON;
  result = getMeanSTD(brType, cpt_surfaceArea, minVal, maxVal);//mean+/-STD
  std::cerr<< "  surfaceArea = " << result.first << " +/- " << result.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;
  result2 = getMeanSTD(brType, cpt_volume, minVal, maxVal);//mean+/-STD
  std::cerr<< "  volume = " << result2.first << " +/- " << result2.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;
  result2 = getMeanSTD(brType, cpt_length, minVal, maxVal);//mean+/-STD
  std::cerr<< "  length = " << result2.first << " +/- " << result2.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;

  std::cerr << "Compartments BASALDEN: \n";
  brType = Branch::_BASALDEN;
  result = getMeanSTD(brType, cpt_surfaceArea, minVal, maxVal);//mean+/-STD
  std::cerr<< "  surfaceArea = " << result.first << " +/- " << result.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;
  result2 = getMeanSTD(brType, cpt_volume, minVal, maxVal);//mean+/-STD
  std::cerr<< "  volume = " << result2.first << " +/- " << result2.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;
  result2 = getMeanSTD(brType, cpt_length, minVal, maxVal);//mean+/-STD
  std::cerr<< "  length = " << result2.first << " +/- " << result2.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;

  std::cerr << "Compartments APICALDEN: \n";
  brType = Branch::_APICALDEN;
  result = getMeanSTD(brType, cpt_surfaceArea, minVal, maxVal);//mean+/-STD
  std::cerr<< " surfaceArea = " << result.first << " +/- " << result.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;
  result2 = getMeanSTD(brType, cpt_volume, minVal, maxVal);//mean+/-STD
  std::cerr<< " volume = " << result2.first << "+/- " << result2.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;
  result2 = getMeanSTD(brType, cpt_length, minVal, maxVal);//mean+/-STD
  std::cerr<< "  length = " << result2.first << " +/- " << result2.second << "\n";
  std::cerr << "   min = " << minVal <<"; max = " << maxVal << std::endl;
//    << "           volume = " << meanVolume << "+/- " << stdevVolume << std::endl;
//  std::vector<float>* v;
//  v = &cpt_surfaceArea;
//  float sum = std::accumulate(v->begin(), v->end(), 0.0);
//  float meanSurfaceArea = sum/v->size();
//  std::vector<float> diff(v->size());
//  std::transform(v->begin(), v->end(), diff.begin(), [meanSurfaceArea](double x) { return x - meanSurfaceArea; });
////  float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
////  float stdevSurfaceArea = std::sqrt(sq_sum / v->size());
//  float sq_sum = std::inner_product(v->begin(), v->end(), v->begin(), 0.0);
//  float stdevSurfaceArea = std::sqrt(sq_sum / v->size() - meanSurfaceArea * meanSurfaceArea);
//
//  //volume
//  v = &cpt_volume;
//  std::fill(diff.begin(), diff.end(), 0.0);
//  sum = std::accumulate(v->begin(), v->end(), 0.0);
//  float meanVolume = sum/v->size();
//  std::transform(v->begin(), v->end(), diff.begin(), [meanVolume](double x) { return x - meanVolume; });
//  sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//  float stdevVolume = std::sqrt(sq_sum / v->size());
//  std::cerr<< "Compartments: surfaceArea = " << meanSurfaceArea << "+/- " << stdevSurfaceArea << "\n"
//    << "           volume = " << meanVolume << "+/- " << stdevVolume << std::endl;
#endif
}

// GOAL: returns a nodeset satisfying the given criteria
//    that will be used in a connection functor, e.g. PolyConnector
//  E.g.: this nodeset will be used to connect to a variable for outputing data 
//
// The criteria to select the nodeset
//    1. CATEGORY (a string) which can be either "BRANCH", "JUNCTION",
//    "CHANNEL", "SYNAPSE" or [new] "CLEFT"
//        This must match with the category of the type-name given next
//    2. TYPE (a string) which is the name of argument passed to the 'nodekind',
//    e.g. nodekind="Channels[Nat]",
//                           then the name to be used is 'Nat'
//  NOTE:  for BRANCH, use name passed in 'CompartmentVariables[]'
//         for JUNCTION, use name passed in 'Junctions[]' (i.e. name of compartmental variables)
//         for CHANNEL, use name passed in 'Channels[]' (i.e. names of channels)
//         for SYNAPSE, use name passed in 'ElectricalSynapses[]' or
//                      'ChemicalSynapses[]' (which means name of receptors)
//         for CLEFT, use name passed in 'SynapticClefts[]'  (which means name of receptors)
//    3. extra information about locations of data
//        NOTE: only required if CATEGORY is BRANCH, JUNCTION, or CHANNEL
//    e.g. 
//      NEURON_INDEX (an integer, maybe optional) which is the index of the
//                    neuron based on the order given in neurons.txt
//
// NOTE: It is mainly used identify the data (to be recorded) 
void TissueFunctor::doProbe(LensContext* lc, std::auto_ptr<NodeSet>& rval)
{
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  NDPairList::iterator ndpiter = _params->end(),
                       ndpend_reverse = _params->begin();
  --ndpiter;
  --ndpend_reverse;

  if ((*ndpiter)->getName() != "CATEGORY")
  {
    std::cerr << "First parameter of TissueProbe must be CATEGORY!"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  StringDataItem* categoryDI =
      dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
  if (categoryDI == 0)
  {
    std::cerr << "CATEGORY parameter of TissueProbe must be a string!"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string category = categoryDI->getString();
  std::string task(("TissueProbe"));
  assert(isValidCategoryString(category, task));
  
  --ndpiter;

  if ((*ndpiter)->getName() != "TYPE")
  {
    std::cerr << "Second parameter of TissueProbe must be TYPE!" << std::endl;
    exit(EXIT_FAILURE);
  }
  StringDataItem* typeDI =
      dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
  if (typeDI == 0)
  {
    std::cerr << "TYPE parameter of TissueProbe must be a string!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string type = typeDI->getString();

  int typeIdx;
  bool esyn;
  typeIdx = getTypeLayerIdx(category, type, esyn);

  unsigned int idx = -1;

  --ndpiter;

  unsigned long long mask = 0;
  key_size_t targetKey = 0;

  if (category == "BRANCH" || category == "JUNCTION" || category == "CHANNEL"
      || category == "SYNAPSE" || category == "CLEFT"
      )
  {//find: mask + targetKey
    unsigned int* ids = new unsigned int[_params->size() - 2];
    for (; ndpiter != ndpend_reverse; --ndpiter)
    {
      NumericDataItem* ndi =
          dynamic_cast<NumericDataItem*>((*ndpiter)->getDataItem());
      if (ndi <= 0)
      {
        std::cerr << "TissueProbe parameter specification must comprise "
                     "unsigned integers!" << std::endl;
        exit(EXIT_FAILURE);
      }
      maskVector.push_back(
          _segmentDescriptor.getSegmentKeyData((*ndpiter)->getName()));

      int val = ndi->getUnsignedInt();
      Params::reviseParamValue((unsigned int&)val, (*ndpiter)->getName());
      if (val < 0)
      {
        std::cerr << "ERROR: The value of " << (*ndpiter)->getName() << " is in invalid range" << std::endl;
        assert(val >= 0);
      }
      ids[++idx] = val; 
    }

    mask = _segmentDescriptor.getMask(maskVector);
    targetKey = _segmentDescriptor.getSegmentKey(maskVector, ids);
    delete[] ids;
  }
  std::vector<NodeDescriptor*> nodeDescriptors;
  GridLayerDescriptor* layer = 0;
  std::map<ComputeBranch*, std::vector<int> >* indexMap;
  if (category == "BRANCH")
  {
    layer = _compartmentVariableLayers[typeIdx];
    assert(layer);
    std::map<ComputeBranch*, std::vector<int> >::iterator mapiter,
        mapend = _branchIndexMap[type].end();
    for (mapiter = _branchIndexMap[type].begin(); mapiter != mapend; ++mapiter)
    {
      key_size_t key = mapiter->first->_capsules->getKey();
      if ((mapiter->second)[0] == _rank &&
          _segmentDescriptor.getSegmentKey(key, mask) == targetKey)
        nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(
            (mapiter->second)[0], (mapiter->second)[1]));
    }
  }
  if (category == "JUNCTION")
  {
    layer = _junctionLayers[typeIdx];
    assert(layer);
    std::map<Capsule*, std::vector<int> >::iterator mapiter,
        mapend = _junctionIndexMap[type].end();
    for (mapiter = _junctionIndexMap[type].begin(); mapiter != mapend;
         ++mapiter)
    {
      key_size_t key = mapiter->first->getKey();
      if ((mapiter->second)[0] == _rank &&
          _segmentDescriptor.getSegmentKey(key, mask) == targetKey)
        nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(
            (mapiter->second)[0], (mapiter->second)[1]));
    }
  }
  if (category == "CHANNEL")
  {
    layer = _channelLayers[typeIdx];
    assert(layer);
    int density = layer->getDensity(_rank);
    int nChannelBranches = _channelBranchIndices1[typeIdx].size();
    key_size_t key;
    for (int i = 0; i < density; ++i)
    {
      if (i < nChannelBranches)
      {
#ifdef MICRODOMAIN_CALCIUM
        std::tuple<int, int, std::string>& channelBranchIndexPair =
            _channelBranchIndices1[typeIdx][i][0];
        key =
            findBranch(_rank, std::get<0>(channelBranchIndexPair),
                       _compartmentVariableTypes[std::get<1>(channelBranchIndexPair)])
                ->_capsules[0]
                .getKey();
#else
        std::pair<int, int>& channelBranchIndexPair =
            _channelBranchIndices1[typeIdx][i][0];
        key =
            findBranch(_rank, channelBranchIndexPair.first,
                       _compartmentVariableTypes[channelBranchIndexPair.second])
                ->_capsules[0]
                .getKey();
#endif
      }
      else
      {
#ifdef MICRODOMAIN_CALCIUM
        std::tuple<int, int, std::string>& channelJunctionIndexPair =
            _channelJunctionIndices1[typeIdx][i - nChannelBranches][0];
        key = findJunction(
                  _rank, std::get<0>(channelJunctionIndexPair),
                  _compartmentVariableTypes[std::get<1>(channelJunctionIndexPair)])
                  ->getKey();
#else
        std::pair<int, int>& channelJunctionIndexPair =
            _channelJunctionIndices1[typeIdx][i - nChannelBranches][0];
        key = findJunction(
                  _rank, channelJunctionIndexPair.first,
                  _compartmentVariableTypes[channelJunctionIndexPair.second])
                  ->getKey();
#endif
      }
//#ifdef MICRODOMAIN_CALCIUM
////not implemented yet
//      if (_segmentDescriptor.getSegmentKey(key, mask) == targetKey &&
//          layer->getNodeAccessor()->getNodeDescriptor(_rank, i))->microdomainName.size() > 0   //TUAN HERE
//          )
//        nodeDescriptors.push_back(
//            layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
//#else
//It's ok to print current that direct toward domain just by specifying the branch criteria
      if (_segmentDescriptor.getSegmentKey(key, mask) == targetKey)
        nodeDescriptors.push_back(
            layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
//#endif
    }
  }
  if (category == "SYNAPSE")
  {
    layer = esyn ? _electricalSynapseLayers[typeIdx]
                 : _chemicalSynapseLayers[typeIdx];
    assert(layer);
    int density = layer->getDensity(_rank);

    if (maskVector.size() == 0)
    {
      for (int i = 0; i < density; ++i)
      {
        nodeDescriptors.push_back(
            layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
      }
    }
    //else{//not an option to use
    //  for (int i = 0; i < density; ++i)
    //  {
    //    //process the check here
    //    //
    //    NodeDescriptor* nd = layer->getNodeAccessor()->getNodeDescriptor(_rank, i);
    //    ShallowArray< BranchDataStruct* >* 
    //      bdsArray = 
    //      (dynamic_cast<BranchDataArrayProducer>(nd))->CG_get_BranchDataArrayProducer_branchDataArray();

    //    BranchDataStruct* postSide = bdsArray[1];

    //    key_size_t key = postSide->key;
    //    if ((mapiter->second)[0] == _rank &&
    //        _segmentDescriptor.getSegmentKey(key, mask) == targetKey)
    //    {
    //      nodeDescriptors.push_back(
    //          layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
    //    }
    //  }
    //}
    else{
      std::map<Touch*, int>& tmap = _synapseReceptorMaps[typeIdx];
      std::map<Touch*, int>::iterator siter,
        siend = tmap.end();
      for (siter = tmap.begin(); siter != siend; siter++)
      {
        //Post-side
        key_size_t key = (*siter).first->getKey2();
        if (_segmentDescriptor.getSegmentKey(key, mask) == targetKey)
        {
          int i = (*siter).second;
          nodeDescriptors.push_back(
              layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
        }
      }
    }
  }
  else if (category == "CLEFT" and _synapticCleftLayers.size() > 0)
  {
    layer = _synapticCleftLayers[typeIdx];
    if (not layer)
    {
      std::cerr << "ERROR: there is no SynapticCleft of type " << type << std::endl;
    }
    assert(layer);
    if (maskVector.size() == 0)
    {//get all layers
      //layer = preSynPoints ? _preSynapticPointLayers[typeIdx]
      //             : _synapticCleftLayers[typeIdx];
      int density = layer->getDensity(_rank);
      for (int i = 0; i < density; ++i)
      {
        nodeDescriptors.push_back(
            layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
      }

    }
    else{
      std::map<Touch*, int>& tmap = _synapticCleftMaps[typeIdx];
      std::map<Touch*, int>::iterator siter,
        siend = tmap.end();
      for (siter = tmap.begin(); siter != siend; siter++)
      {
        // Pre-side and Post-side
        key_size_t key1= (*siter).first->getKey1();
        key_size_t key2= (*siter).first->getKey2();
        if (_segmentDescriptor.getSegmentKey(key1, mask) == targetKey
            || _segmentDescriptor.getSegmentKey(key2, mask) == targetKey 
           )
        {
          int i = (*siter).second;
          nodeDescriptors.push_back(
              layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
        }
      }
    }
  }

  NodeSet* ns = 0;
  if (nodeDescriptors.size() > 0)
  {
    ns = new NodeSet(
        (*nodeDescriptors.begin())->getGridLayerDescriptor()->getGrid(),
        nodeDescriptors);
  }
  else
  {
    ns = new NodeSet(layer->getGrid());
    ns->empty();
  }
  rval.reset(ns);
}
// GOAL:  <internal use by tissueFunctor("Layout", <PROBE="pr0"...>)>
//    return the set of node instances to be selected
//      via the variable nodeDescriptors
Grid* TissueFunctor::doProbe(LensContext* lc, std::vector<NodeDescriptor*>& nodeDescriptors)
{
  Grid* rval=0;
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  NDPairList::iterator ndpiter=_params->end(), ndpend_reverse=_params->begin();
  --ndpiter;
  --ndpend_reverse;

  std::string layout="NO_LAYOUT_ID_SPECIFIED";
  int remaining=_params->size();

  assert((*ndpiter)->getName()=="PROBED");
  StringDataItem* layoutDI = dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
  layout=layoutDI->getString();

  std::vector<float> vN;
  int NumCptsToExtract = -1;
  ProbedCategory_t probeCat = ProbedCategory_t::NOT_SET;

  --ndpiter;
  --remaining;
  if (ndpiter!=ndpend_reverse && (*ndpiter)->getName()=="N") {
    NumericDataItem* nDI = dynamic_cast<NumericDataItem*>((*ndpiter)->getDataItem());
    if (nDI==0) {
#ifdef NVU_NTS_EXTENSION
      FloatArrayDataItem* arrayDI = dynamic_cast<FloatArrayDataItem*>((*ndpiter)->getDataItem());
      if (arrayDI==0) {
        std::cerr<<"N parameter of TissueProbe must be a number (NVU-NVU,NVU-MGS) " <<
          "or an array (NVU-NTS) of " <<
          "{ NVU_X, NVU_Y, NVVU_Z, RADIUS} !"<<std::endl;
        exit(0);
      }
      else{
        vN = *(arrayDI->getFloatVector());
        probeCat = ProbedCategory_t::NTS_NVU;
      }
#else
      std::cerr<<"N parameter of TissueProbe must be a number!"<<std::endl;
      exit(0);
#endif
    }
    else{
        NumCptsToExtract=nDI->getInt();
        assert(NumCptsToExtract > 0);
        probeCat = ProbedCategory_t::NTS_MGS;
    }
    --ndpiter;
    --remaining;
  }
  std::map<std::string, std::map<std::pair<std::string, std::string>, std::pair<Grid*, std::vector<NodeDescriptor*> > > > ::iterator miter=_probedNodesMap.end();
  if (layout!="NO_LAYOUT_ID_SPECIFIED") miter=_probedNodesMap.find(layout);
  if (miter!=_probedNodesMap.end()) {
    //PROBE-name exist, reuse them
    if (remaining<2)
    {
      if (_rank == 0){
        std::cerr<<"Error on TissueFunctor Probe! No mask specified!"<<std::endl;
        std::cerr<<".... at least CATEGORY and TYPE."<<std::endl;
      }
      assert(0);
    }
//#ifdef NVU_NTS_EXTENSION
//    std::map<std::string, ProbedCategory_t> ::iterator 
//        probiter = _probedCategory.find(layout);
//    if (probiter != _probedCategory.end() &&
//        (probeCat != (*probiter).second))
//    {
//      if (_rank == 0){
//        std::cerr<<"Error on TissueFunctor Probe! The probe "
//          << layout << " was reused on a different ProbedCategory."
//          << " It was assigned to " << ProbedCategory_ToString((*probiter).second) 
//          << "; and now is used in " << ProbedCategory_ToString(probeCat)<<std::endl;
//      }
//      assert(0);
//    }
//    else{
//      if (_rank == 0){
//        std::cerr<<" You forgot to add the probe" << std::endl;
//      }
//      assert(0);
//    }
//#endif

    std::pair<std::string, std::string> cattype=getCategoryTypePair(ndpiter, remaining);
    std::map<std::pair<std::string, std::string>, 
      std::pair<Grid*, std::vector<NodeDescriptor*> > > ::iterator 
        mmiter = miter->second.find(cattype);
    if (mmiter!=miter->second.end()) {
      //found the existing (CATEGORY, TYPE) list of nodes
      rval = mmiter->second.first;
      nodeDescriptors = mmiter->second.second;
    }
    else {
      //existing PROBE-name; but pass in different (CATEGORY,TYPE) selection criteria
      // It means we need to get nodes in that layer but keep using the same index
      // IMPORTANT: any value of N=? is ignored
      nodeDescriptors.clear();
      std::string category=cattype.first;
      std::string type=cattype.second;
      bool esyn=false;
      int typeIdx=getTypeLayerIdx(category, type, esyn);
      GridLayerDescriptor* layer=0;
      layer = getGridLayerDescriptor(category, typeIdx, esyn);
      assert(layer);

      rval = layer->getGrid();

      std::vector<NodeDescriptor*>& pattern=miter->second.begin()->second.second;
      std::vector<NodeDescriptor*>::iterator vend=pattern.end(), viter;
      //if (N != pattern->size())
      //{//we cannot check, as N is splitted across processes
      //  std::cerr << "WARNING: Please make sure you use the same N's value for the same PROBED name " << std::endl;
      //  assert(0);
      //}
      //or maybe we can sum all the 'size' and check (using rank0) if the sum is equal to 'N'
      for (viter=pattern.begin(); viter!=vend; ++viter) {
        NodeDescriptor* nd = *viter;
        nodeDescriptors.push_back(layer->getNodeAccessor()->
            getNodeDescriptor(nd->getNodeIndex(), nd->getDensityIndex()));
      }
    }
  }
  else{
    //first time use the PROBE-name or we use non-PROBE-layer
    nodeDescriptors.clear();
    if (remaining<2)
    {
      std::cerr<<"Error on TissueFunctor Probe! No mask specified!"<<std::endl;
      assert(0);
    }
    _probedCategory[layout] = probeCat;
    if (NumCptsToExtract == -1)
    {
      rval = doProbe_Region(lc, nodeDescriptors,
          layout, 
          ndpiter, ndpend_reverse,
          remaining,
          vN);
    }else{
      rval = doProbe_Number(lc, nodeDescriptors,
          layout, 
          ndpiter, ndpend_reverse,
          remaining,
          NumCptsToExtract);
    }
  }
  return rval;
}
Grid* TissueFunctor::doProbe_Number(LensContext* lc, std::vector<NodeDescriptor*>& nodeDescriptors,
        const std::string layout,
        NDPairList::iterator& ndpiter, NDPairList::iterator& ndpend_reverse,
        int& remaining,
        int NumCptsToExtract)
{
  Grid* rval=0;
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  {
    //first time use the PROBE-name or we use non-PROBE-layer
    std::pair<std::string, std::string> cattype=getCategoryTypePair(ndpiter, remaining);
    std::string category = cattype.first;
    std::string type = cattype.second;

    bool esyn=false;
    int typeIdx=getTypeLayerIdx(category, type, esyn);

    --ndpiter;
    --remaining;

    unsigned long long mask=0;
    key_size_t targetKey=0;

    //TUAN ADD PROBE SYNAPSERECEPTOR + CLEFT
    //if (category == "BRANCH" || category == "JUNCTION" || category == "CHANNEL"
    //    || category == "SYNAPSE" || category == "CLEFT"
    //   )
    if (category=="BRANCH" || category=="JUNCTION" || category=="CHANNEL") 
    {
      //find: mask + targetKey
      unsigned int* ids=new unsigned int[remaining];
      unsigned int idx=-1;
      for (; ndpiter!=ndpend_reverse; --ndpiter, --remaining) {
        NumericDataItem* ndi=dynamic_cast<NumericDataItem*>((*ndpiter)->getDataItem());
        if (ndi<=0) {
          std::cerr<<"TissueProbe parameter specification must comprise unsigned integers!"<<std::endl;
          exit(0);
        }
        maskVector.push_back(_segmentDescriptor.getSegmentKeyData((*ndpiter)->getName()));
        int val = ndi->getUnsignedInt();
        std::string fieldName ((*ndpiter)->getName());
        Params::reviseParamValue((unsigned int&)val, fieldName);
        if (val < 0)
        {
          std::cerr << "ERROR: The value of " << (*ndpiter)->getName() << " is in invalid range" << std::endl;
          assert(val >= 0);
        }
        ids[++idx] = val; 
      }

      mask=_segmentDescriptor.getMask(maskVector);
      targetKey=_segmentDescriptor.getSegmentKey(maskVector, ids);
      delete ids;
    }

    //SELECTION PROCESS
    // build vector -> nodeDescriptors
    std::vector<double> surfaceAreas;
    GridLayerDescriptor* layer=0;
    std::map<ComputeBranch*, std::vector<int> >* indexMap;

    //Step 1 (case 1): any things associated with a given branch (e.g. CB or Junction)
    //             density assignment based on surfaceArea of all matched-compartment area
    if (category=="BRANCH") {
      layer=_compartmentVariableLayers[typeIdx];
      assert(layer);
      std::map<ComputeBranch*, std::vector<int> >::iterator mapiter, mapend=_branchIndexMap[type].end();
      for (mapiter=_branchIndexMap[type].begin(); mapiter!=mapend; ++mapiter) {
        key_size_t key=mapiter->first->_capsules->getKey();
        if ( (mapiter->second)[0]==_rank && _segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
          nodeDescriptors.push_back(layer->getNodeAccessor()->
              getNodeDescriptor((mapiter->second)[0], (mapiter->second)[1]));
          if (NumCptsToExtract>=0) surfaceAreas.push_back(mapiter->first->getSurfaceArea());
        }
      }
    }
    if (category=="JUNCTION") {
      layer=_junctionLayers[typeIdx];
      assert(layer);
      std::map<Capsule*, std::vector<int> >::iterator mapiter, mapend=_junctionIndexMap[type].end();
      for (mapiter=_junctionIndexMap[type].begin(); mapiter!=mapend; ++mapiter) {
        key_size_t key=mapiter->first->getKey();
        if ( (mapiter->second)[0]==_rank && _segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
          nodeDescriptors.push_back(layer->getNodeAccessor()->
              getNodeDescriptor((mapiter->second)[0], (mapiter->second)[1]));
          if (NumCptsToExtract>=0) surfaceAreas.push_back(mapiter->first->getEndSphereSurfaceArea());
        }
      }
    }
    if (category=="CHANNEL") {
      //NOTE: The density of channels come from  (1) those associated with CBs;
      //               and then (2) those associated with Junctions
      layer=_channelLayers[typeIdx];
      assert(layer);
      int density=layer->getDensity(_rank);
      int nChannelBranches=_channelBranchIndices1[typeIdx].size(); 
      key_size_t key;
      for (int i=0; i<density; ++i) {
        double surfaceArea=0;
        if (i < nChannelBranches) {
#ifdef MICRODOMAIN_CALCIUM
          std::tuple<int, int, std::string>& channelBranchIndexPair
            =_channelBranchIndices1[typeIdx][i][0];
          ComputeBranch* branch=findBranch(_rank, std::get<0>(channelBranchIndexPair), 
              _compartmentVariableTypes[std::get<1>(channelBranchIndexPair)]);
#else
          std::pair<int, int>& channelBranchIndexPair=
            _channelBranchIndices1[typeIdx][i][0];
          ComputeBranch* branch=findBranch(_rank, channelBranchIndexPair.first, 
              _compartmentVariableTypes[channelBranchIndexPair.second]);
#endif
          key=branch->_capsules[0].getKey();
          if (NumCptsToExtract>=0) surfaceArea=branch->getSurfaceArea();
        }
        else {
#ifdef MICRODOMAIN_CALCIUM
          std::tuple<int, int, std::string>& channelJunctionIndexPair=
            _channelJunctionIndices1[typeIdx][i-nChannelBranches][0];
          Capsule* junction=findJunction(_rank, std::get<0>(channelJunctionIndexPair), 
              _compartmentVariableTypes[std::get<1>(channelJunctionIndexPair)]);
#else
          std::pair<int, int>& channelJunctionIndexPair=
            _channelJunctionIndices1[typeIdx][i-nChannelBranches][0];
          Capsule* junction=findJunction(_rank, channelJunctionIndexPair.first, 
              _compartmentVariableTypes[channelJunctionIndexPair.second]);
#endif
          key=junction->getKey();
          if (NumCptsToExtract>=0) surfaceArea=junction->getEndSphereSurfaceArea();
        }
        if (_segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
          nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
          if (NumCptsToExtract>=0) surfaceAreas.push_back(surfaceArea);
        }
      }
    }
    //Step 1 (case 2): any things independent from a neuron (e.g. synapse, cleft)
    //            density assignment based on counts (i.e. same surface Area)
    if (category=="SYNAPSE") {
      //use Post-side
      layer = esyn ? _electricalSynapseLayers[typeIdx] : _chemicalSynapseLayers[typeIdx];
      assert(layer);
      int density=layer->getDensity(_rank);
      std::map<Touch*, int> mymap = _synapseReceptorMaps[typeIdx]; 
      for (auto const &entity : mymap )
      {
        Touch* touch = entity.first;
        int i = entity.second;
        key_size_t key = touch->getKey2();
        if (_segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
          nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
          if (NumCptsToExtract>=0) surfaceAreas.push_back(1.0);
        }
      }
      //for (int i=0; i<density; ++i) {
      //  nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
      //}
    }
    if (category=="CLEFT") {
      //use Pre-side 
      layer = _synapticCleftLayers[typeIdx] ;
      assert(layer);
      int density=layer->getDensity(_rank);
      std::map<Touch*, int> mymap = _synapticCleftMaps[typeIdx]; 
      for (auto const &entity : mymap )
      {
        Touch* touch = entity.first;
        int i = entity.second;
        key_size_t key = touch->getKey1();
        if (_segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
          nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
          if (NumCptsToExtract>=0) surfaceAreas.push_back(1.0);
        }
      }
      //for (int i=0; i<density; ++i) {
      //  nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
      //}
    }

    rval = layer->getGrid();
    int nds=nodeDescriptors.size();

    // revise vector -> nodeDescriptors
    if (layout!="NO_LAYOUT_ID_SPECIFIED") {
      ShallowArray< int > lytr; //keep the density
      int* lytc = new int[_size]();
      MPI_Allgather(&nds, 1, MPI_INT, lytc, 1, MPI_INT, MPI_COMM_WORLD);

      if (NumCptsToExtract>=0) {
        RNG rng;
        rng.reSeedShared(layout[0]);
        int seed=lrandom(rng);
        for (int n=1; n<layout.size(); ++n) {
          rng.reSeedShared(seed+layout[n]);
          seed=lrandom(rng);
        }
        rng.reSeedShared(seed);

        int totalNodes=0;
        int hi, lo;
        for (int n=0; n<_size; ++n) { 
          if (n==_rank)	lo=totalNodes;
          totalNodes+=lytc[n];
          if (n==_rank) hi=totalNodes;
        }
        double localMaxSA=0, globalMaxSA=0;
        assert(nds==surfaceAreas.size());
        for (int n=0; n<nds; ++n) 
          if (surfaceAreas[n]>localMaxSA) localMaxSA=surfaceAreas[n];
        MPI_Allreduce(&localMaxSA, &globalMaxSA, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);      
        //normalized all SurfaceArea (SA) [0..1]
        double* saR = new double[nds]();
        for (int n=0; n<nds; ++n) saR[n]=surfaceAreas[n]/globalMaxSA;

        //get all normalized SA
        double* saRT = new double[totalNodes]();
        int* rdispls = new int[_size]();
        rdispls[0]=0;
        for (int n=1; n<_size; ++n) rdispls[n]=rdispls[n-1]+lytc[n-1];
        MPI_Allgatherv(saR, nds, MPI_DOUBLE, saRT, lytc, rdispls, MPI_DOUBLE, MPI_COMM_WORLD);
        std::vector<NodeDescriptor*> survivors;
        int count=0;
        while (count<NumCptsToExtract) {
          std::map<double, int> shuffle;
          for (int n=0; n<totalNodes; ++n) 
            if (drandom(rng)<saRT[n]) shuffle[drandom(rng)]=n;
          std::map<double, int>::iterator miter, mend=shuffle.end();
          for (miter=shuffle.begin(); miter!=mend && count<NumCptsToExtract; ++miter) {
            ++count;
            int nidx=miter->second;
            if (nidx>=lo && nidx<hi) 
              survivors.push_back(nodeDescriptors[miter->second-lo]);
          }
        }
        delete [] saR;
        delete [] saRT;
        delete [] rdispls;

        nodeDescriptors=survivors;
        nds = nodeDescriptors.size();
        MPI_Allgather(&nds, 1, MPI_INT, lytc, 1, MPI_INT, MPI_COMM_WORLD);
      }

      for (int n=0; n<_size; ++n) 
        lytr.push_back(lytc[n]);
      delete [] lytc;
      _probedLayoutsMap[layout]=lytr;
      _probedNodesMap[layout][cattype]=std::pair<Grid*, std::vector<NodeDescriptor*> >
        (rval, nodeDescriptors);
    }
  }
  return rval;
}
Grid* TissueFunctor::doProbe_Region(LensContext* lc, std::vector<NodeDescriptor*>& nodeDescriptors,
    const std::string layout,
    NDPairList::iterator& ndpiter, NDPairList::iterator& ndpend_reverse,
    int& remaining,
    std::vector<float> vN)
{
  Grid* rval=0;
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  //first time use the PROBE-name or we use non-PROBE-layer
  std::pair<std::string, std::string> cattype=getCategoryTypePair(ndpiter, remaining);
  std::string category = cattype.first;
  std::string type = cattype.second;

  bool esyn=false;
  int typeIdx=getTypeLayerIdx(category, type, esyn);

  --ndpiter;
  --remaining;

  unsigned long long mask=0;
  key_size_t targetKey=0;

  //TUAN ADD PROBE SYNAPSERECEPTOR + CLEFT
  //if (category == "BRANCH" || category == "JUNCTION" || category == "CHANNEL"
  //    || category == "SYNAPSE" || category == "CLEFT"
  //   )
  if (category=="BRANCH" || category=="JUNCTION" || category=="CHANNEL") 
  {
    //find: mask + targetKey
    unsigned int* ids=new unsigned int[remaining];
    unsigned int idx=-1;
    for (; ndpiter!=ndpend_reverse; --ndpiter, --remaining) {
      NumericDataItem* ndi=dynamic_cast<NumericDataItem*>((*ndpiter)->getDataItem());
      if (ndi<=0) {
        std::cerr<<"TissueProbe parameter specification must comprise unsigned integers!"<<std::endl;
        exit(0);
      }
      maskVector.push_back(_segmentDescriptor.getSegmentKeyData((*ndpiter)->getName()));
      int val = ndi->getUnsignedInt();
      std::string fieldName ((*ndpiter)->getName());
      Params::reviseParamValue((unsigned int&)val, fieldName);
      if (val < 0)
      {
        std::cerr << "ERROR: The value of " << (*ndpiter)->getName() << " is in invalid range" << std::endl;
        assert(val >= 0);
      }
      ids[++idx] = val; 
    }

    mask=_segmentDescriptor.getMask(maskVector);
    targetKey=_segmentDescriptor.getSegmentKey(maskVector, ids);
    delete ids;
  }

  //SELECTION PROCESS
  // build vector -> nodeDescriptors
  //std::vector<double> surfaceAreas;
  GridLayerDescriptor* layer=0;
  std::map<ComputeBranch*, std::vector<int> >* indexMap;

  //Step 1 (case 1): any things associated with a given branch (e.g. CB or Junction)
  //             density assignment based on surfaceArea of all matched-compartment area
  if (category=="BRANCH") {
    layer=_compartmentVariableLayers[typeIdx];
    assert(layer);
    std::map<ComputeBranch*, std::vector<int> >::iterator mapiter, mapend=_branchIndexMap[type].end();
    for (mapiter=_branchIndexMap[type].begin(); mapiter!=mapend; ++mapiter) {
      key_size_t key=mapiter->first->_capsules->getKey();
      if ( (mapiter->second)[0]==_rank && _segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
        //LIMITATION: use the center capsule's coordinate to
        // determine if we should get that whole CB
        //Capsule& caps = (mapiter->first)->lastCapsule();
        int ncaps = mapiter->first->_nCapsules;
        Capsule& caps = mapiter->first->_capsules[ncaps/2];
        double length = vN[3]; //L0 in meter
        length *= 1e6; // convert to micrometer
        if (isInNVUGrid(caps.getBeginCoordinates(), 3, length, (int)vN[0], (int)vN[1], (int)vN[2]))
        {
          nodeDescriptors.push_back(layer->getNodeAccessor()->
              getNodeDescriptor((mapiter->second)[0], (mapiter->second)[1]));
        //if (NumCptsToExtract>=0) surfaceAreas.push_back(mapiter->first->getSurfaceArea());
        }
      }
    }
  }
  if (category=="JUNCTION") {
    layer=_junctionLayers[typeIdx];
    assert(layer);
    std::map<Capsule*, std::vector<int> >::iterator mapiter, mapend=_junctionIndexMap[type].end();
    for (mapiter=_junctionIndexMap[type].begin(); mapiter!=mapend; ++mapiter) {
      key_size_t key=mapiter->first->getKey();
      if ( (mapiter->second)[0]==_rank && _segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
        //LIMITATION: use the center capsule's coordinate to
        // determine if we should get that whole CB
        //Capsule& caps = (mapiter->first)->lastCapsule();
        Capsule& caps = mapiter->first[0];
        double length = vN[3]; //L0 in meter
        length *= 1e6; // convert to micrometer
        if (isInNVUGrid(caps.getBeginCoordinates(), 3, length, (int)vN[0], (int)vN[1], (int)vN[2]))
        {
          nodeDescriptors.push_back(layer->getNodeAccessor()->
              getNodeDescriptor((mapiter->second)[0], (mapiter->second)[1]));
        //if (NumCptsToExtract>=0) surfaceAreas.push_back(mapiter->first->getEndSphereSurfaceArea());

        }
      }
    }
  }
  if (category=="CHANNEL") {
    //NOTE: The density of channels come from  (1) those associated with CBs;
    //               and then (2) those associated with Junctions
    layer=_channelLayers[typeIdx];
    assert(layer);
    int density=layer->getDensity(_rank);
    int nChannelBranches=_channelBranchIndices1[typeIdx].size(); 
    key_size_t key;
    Capsule* caps;
    for (int i=0; i<density; ++i) {
      double surfaceArea=0;
      if (i < nChannelBranches) {
#ifdef MICRODOMAIN_CALCIUM
        std::tuple<int, int, std::string>& channelBranchIndexPair=_channelBranchIndices1[typeIdx][i][0];
        ComputeBranch* branch=findBranch(_rank, std::get<0>(channelBranchIndexPair), 
            _compartmentVariableTypes[std::get<1>(channelBranchIndexPair)]);
#else
        std::pair<int, int>& channelBranchIndexPair=_channelBranchIndices1[typeIdx][i][0];
        ComputeBranch* branch=findBranch(_rank, channelBranchIndexPair.first, _compartmentVariableTypes[channelBranchIndexPair.second]);
#endif
        key=branch->_capsules[0].getKey();
        int ncaps = branch->_nCapsules;
        caps = &(branch->_capsules[ncaps/2]);
        //if (NumCptsToExtract>=0) surfaceArea=branch->getSurfaceArea();
      }
      else {
#ifdef MICRODOMAIN_CALCIUM
        std::tuple<int, int, std::string>& channelJunctionIndexPair=_channelJunctionIndices1[typeIdx][i-nChannelBranches][0];
        Capsule* junction=findJunction(_rank, std::get<0>(channelJunctionIndexPair), 
            _compartmentVariableTypes[std::get<1>(channelJunctionIndexPair)]);
#else
        std::pair<int, int>& channelJunctionIndexPair=_channelJunctionIndices1[typeIdx][i-nChannelBranches][0];
        Capsule* junction=findJunction(_rank, channelJunctionIndexPair.first, _compartmentVariableTypes[channelJunctionIndexPair.second]);
#endif
        key=junction->getKey();
        caps = junction;
        //if (NumCptsToExtract>=0) surfaceArea=junction->getEndSphereSurfaceArea();
      }
      if (_segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
        double length = vN[3]; //L0 in meter
        length *= 1e6; // convert to micrometer
        if (isInNVUGrid(caps->getBeginCoordinates(), 3, length, (int)vN[0], (int)vN[1], (int)vN[2]))
        {
          nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
        //if (NumCptsToExtract>=0) surfaceAreas.push_back(surfaceArea);

        }
      }
    }
  }
  //Step 1 (case 2): any things independent from a neuron (e.g. synapse, cleft)
  //            density assignment based on counts (i.e. same surface Area)
  if (category=="SYNAPSE") {
    assert(0);
    //use Post-side
    layer = esyn ? _electricalSynapseLayers[typeIdx] : _chemicalSynapseLayers[typeIdx];
    assert(layer);
    int density=layer->getDensity(_rank);
    std::map<Touch*, int> mymap = _synapseReceptorMaps[typeIdx]; 
    for (auto const &entity : mymap )
    {
      Touch* touch = entity.first;
      int i = entity.second;
      key_size_t key = touch->getKey2();
      if (_segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
        assert(0); // not supported yet
        nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
        //if (NumCptsToExtract>=0) surfaceAreas.push_back(1.0);
      }
    }
    //for (int i=0; i<density; ++i) {
    //  nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
    //}
  }
  if (category=="CLEFT") {
    assert(0); // not supported yet
    //use Pre-side 
    layer = _synapticCleftLayers[typeIdx] ;
    assert(layer);
    int density=layer->getDensity(_rank);
    std::map<Touch*, int> mymap = _synapticCleftMaps[typeIdx]; 
    for (auto const &entity : mymap )
    {
      Touch* touch = entity.first;
      int i = entity.second;
      key_size_t key = touch->getKey1();
      if (_segmentDescriptor.getSegmentKey(key, mask)==targetKey) {
        assert(0); // not supported yet
        nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
        //if (NumCptsToExtract>=0) surfaceAreas.push_back(1.0);
      }
    }
    //for (int i=0; i<density; ++i) {
    //  nodeDescriptors.push_back(layer->getNodeAccessor()->getNodeDescriptor(_rank, i));
    //}
  }

  rval = layer->getGrid();
  int nds=nodeDescriptors.size();
//#define DEBUG
#ifdef DEBUG
  std::cerr << "Probe_region total " << nds << " elements" << std::endl;
#endif

  if (layout!="NO_LAYOUT_ID_SPECIFIED") {
    ShallowArray< int > lytr; //keep the density
    int* lytc = new int[_size]();
    MPI_Allgather(&nds, 1, MPI_INT, lytc, 1, MPI_INT, MPI_COMM_WORLD);
    for (int n=0; n<_size; ++n) 
      lytr.push_back(lytc[n]);
    delete [] lytc;
    _probedLayoutsMap[layout]=lytr;
    _probedNodesMap[layout][cattype]=std::pair<Grid*, std::vector<NodeDescriptor*> >
      (rval, nodeDescriptors);
  }
  return rval;
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

// GOAL: map the values from *.par file to the data member
//  ModelType can be CHANNEL | COMPARTMENT | SYNAPSE
//  Supported ModelType: CHANNEL | COMPARTMENT
// NOTE: only the first two ones have data inputable via *params.par file
//   As currently, Synapse Receptors are not supposed to
// have parameters whose values varies upon neuron types/branch location
// TUAN TODO: maybe consider this in the future
// Example:
//  <gbar={10.0}>
//  <float:gbar={10.0}>
//  <int:m=2>
//  <string:tag=abc>
void TissueFunctor::getModelParams(Params::ModelType modelType,
                                   NDPairList& paramsLocal,
                                   std::string& nodeType, key_size_t key)
{
  // NOTE: Currently for compartment data, e.g. Cm, gLeak
  // it is limited to have single value for all compartments in 1 branch
  std::list<std::pair<std::string, float> > compartmentParams;
  _tissueParams.getModelParams(modelType, nodeType, key, compartmentParams);
  std::list<std::pair<std::string, float> >::iterator
      cpiter = compartmentParams.begin(),
      cpend = compartmentParams.end();
  for (; cpiter != cpend; ++cpiter)
  {
    // NOTE: if we update the next 'for' code, update here too
    std::string mystring = (*cpiter).first;
    std::vector<std::string> tokens;
    std::string delimiters = ":";
    StringUtils::Tokenize(mystring, tokens, delimiters);
    if (tokens.size() != 1 && tokens.size() != 2)
      std::cerr << " ERROR at modelType =" << modelType << ": "<< mystring << " \n .. if you want explicit data type, use say 'float:gbar={10.0}' , i.e. maximum one semicolon (:) " << std::endl;
    assert(tokens.size() == 1 || tokens.size() == 2);
    std::string varName = tokens[tokens.size() - 1];

    if (tokens.size() == 1 || tokens[0] == "float")
    {
      // NEW CODE
      // first check if the tokens presence
      //
      NDPairList::iterator ndpiter, ndpend = paramsLocal.end();
      bool found = false;
      for (ndpiter = paramsLocal.begin(); ndpiter != ndpend; ++ndpiter)
      {
        if (varName.compare((*ndpiter)->getName()) == 0)
        // if ((*ndpiter)->getName() 	== varName)
        {
          found = true;
          FloatArrayDataItem* arrayDI =
              dynamic_cast<FloatArrayDataItem*>((*ndpiter)->getDataItem());
          if (arrayDI == 0)
          {  // handle the case for scalar data, e.g. Cm
            FloatDataItem* fltDI =
                dynamic_cast<FloatDataItem*>((*ndpiter)->getDataItem());
            if (fltDI == 0)
            {
              std::cerr << "TissueFunctor: " << varName
                        << " is being used but not Float" << std::endl;
              assert(0);
            }
            fltDI->setFloat((float)cpiter->second);
          }
          else
          {
            // handle the situation for array data, e.g. Vnew
            std::vector<int> coords;
            // NOTE: coords.push_back(index);
            // with 'index' is the value from 0 to #compartments-1
            //                                    in that branch
            coords.push_back(0);  // the first
            arrayDI->setFloat(coords, (float)cpiter->second);
          }
          break;
        }
      }
      //
      if (!found)
      {
        FloatDataItem* paramDI = new FloatDataItem((float)cpiter->second);
        std::auto_ptr<DataItem> paramDI_ap(paramDI);
        NDPair* ndp = new NDPair(varName, paramDI_ap);
        paramsLocal.push_back(ndp);
      }
      // END NEW CODE
      // FloatDataItem* paramDI = new FloatDataItem((float)cpiter->second);
      // std::auto_ptr<DataItem> paramDI_ap(paramDI);
      // NDPair* ndp = new NDPair(varName, paramDI_ap);
      // paramsLocal.push_back(ndp);
    }
    else if (tokens[0] == "int")
    {
      ////////////
      IntDataItem* paramDI = new IntDataItem((int)cpiter->second);
      std::auto_ptr<DataItem> paramDI_ap(paramDI);
      NDPair* ndp = new NDPair(varName, paramDI_ap);
      paramsLocal.push_back(ndp);
    }
		//TUAN TODO: update Params.cxx for this to work
    //else if (tokens[0] == "string")
    //{
    //  ////////////
		//	StringDataItem* paramDI = new StringDataItem(cpiter->second);
    //  std::auto_ptr<DataItem> paramDI_ap(paramDI);
    //  NDPair* ndp = new NDPair(varName, paramDI_ap);
    //  paramsLocal.push_back(ndp);
    //}
  }

  // NOTE: for channel data, e.g. gbar
  // we can have different values for different compatments in 1 branch
  // that's why we use std:vector<dyn_var_t>  here
  std::list<std::pair<std::string, std::vector<float> > >
      compartmentArrayParams;
  _tissueParams.getModelArrayParams(modelType, nodeType, key,
                                    compartmentArrayParams);
  std::list<std::pair<std::string, std::vector<float> > >::iterator
      capiter = compartmentArrayParams.begin(),
      capend = compartmentArrayParams.end();
  for (; capiter != capend; ++capiter)
  {
    std::string mystring = (*capiter).first;
    std::vector<std::string> tokens;
    std::string delimiters = ": ";
    StringUtils::Tokenize(mystring, tokens, delimiters);
    assert(tokens.size() == 1 || tokens.size() == 2);
    std::string varName = tokens[tokens.size() - 1];

    if (tokens.size() == 1 || tokens[0] == "float")
    {
      ShallowArray<float> farr;
      std::vector<float>::iterator viter = capiter->second.begin(),
                                       vend = capiter->second.end();
      for (; viter != vend; ++viter) farr.push_back(*viter);
      FloatArrayDataItem* paramDI = new FloatArrayDataItem(farr);
      std::auto_ptr<DataItem> paramDI_ap(paramDI);
      if (!paramsLocal.replace(capiter->first, paramDI_ap))
      {
        // NDPair* ndp = new NDPair(capiter->first, paramDI_ap);
        NDPair* ndp = new NDPair(varName, paramDI_ap);
        paramsLocal.push_back(ndp);
      }
    }
    else if (tokens[0] == "int")
    {
      ShallowArray<int> farr;
      std::vector<float>::iterator viter = capiter->second.begin(),
                                       vend = capiter->second.end();
      for (; viter != vend; ++viter) farr.push_back(*viter);
      IntArrayDataItem* paramDI = new IntArrayDataItem(farr);
      std::auto_ptr<DataItem> paramDI_ap(paramDI);
      if (!paramsLocal.replace(capiter->first, paramDI_ap))
      {
        // NDPair* ndp = new NDPair(capiter->first, paramDI_ap);
        NDPair* ndp = new NDPair(varName, paramDI_ap);
        paramsLocal.push_back(ndp);
      }
    }
		//TUAN TODO: update Params.cxx for this to work
    //else if (tokens[0] == "string")
    //{
    //  ////////////
		//	StringDataItem* paramDI = new StringDataItem(cpiter->second);
    //  std::auto_ptr<DataItem> paramDI_ap(paramDI);
    //  NDPair* ndp = new NDPair(varName, paramDI_ap);
    //  paramsLocal.push_back(ndp);
    //}
  }
}

// GOAL: check if a compartment (based on its 'key') has the given channel
// 'nodeType'
//     connect to it
bool TissueFunctor::isChannelTarget(key_size_t key, std::string nodeType)
{
  bool rval = false;
  std::list<Params::ChannelTarget>* channelTypes =
      _tissueParams.getChannelTargets(key);
  if (channelTypes)
  {
    std::list<Params::ChannelTarget>::iterator iiter = channelTypes->begin(),
                                               iend = channelTypes->end();
    for (; iiter != iend; ++iiter)
    {
      if (iiter->_type == nodeType)
      {
        rval = true;
        break;
      }
    }
  }
  return rval;
}

// GOAL: assign the probabilities (output-arg: probabilities)
//        for generating the electrical synapse
//     for a given key-pair (i.e. compartment-to-compartment as a touch via
//        input-arg: titer)
//     based on the values given in the last column in the SynParam file
//
void TissueFunctor::getElectricalSynapseProbabilities(
    std::vector<double>& probabilities, TouchVector::TouchIterator& titer,
    std::string nodeType)
{
  key_size_t key1 = titer->getKey1();
  key_size_t key2 = titer->getKey2();
  if (_tissueParams.electricalSynapses() &&
      (key1 < key2 ||
       !_tissueParams.symmetricElectricalSynapseTargets(key1, key2)))
  {
    std::list<Params::ElectricalSynapseTarget>* synapseTypes =
        _tissueParams.getElectricalSynapseTargets(key1, key2);
    if (synapseTypes)
    {
      std::list<Params::ElectricalSynapseTarget>::iterator
          iiter = synapseTypes->begin(),
          iend = synapseTypes->end();
      for (; iiter != iend; ++iiter)
      {
        if (iiter->_type == nodeType)
          probabilities.push_back(iiter->_parameter);
      }
    }
  }
}

void TissueFunctor::getChemicalSynapseProbabilities(
    std::vector<double>& probabilities, TouchVector::TouchIterator& titer,
    std::string nodeType)
{
  key_size_t key1 = titer->getKey1();
  key_size_t key2 = titer->getKey2();
  if (_tissueParams.chemicalSynapses())
  {
    std::list<Params::ChemicalSynapseTarget>* synapseTypes =
        _tissueParams.getChemicalSynapseTargets(key1, key2);
    if (synapseTypes)
    {
      std::vector<int> typeCounter;
      typeCounter.resize(_chemicalSynapseTypesMap.size(), 0);
      std::list<Params::ChemicalSynapseTarget>::iterator
          iiter = synapseTypes->begin(),
          iend = synapseTypes->end();
      for (; iiter != iend; ++iiter)
      {
        bool generated = false;
        bool nonGenerated = false;
        bool hit = false;
        bool mixedSynapse = (iiter->_targets.size() > 1) ? true : false;
        std::map<std::string, std::pair<std::list<std::string>,
                                        std::list<std::string> > >::iterator
            targetsIter,
            targetsEnd = iiter->_targets.end();
        int d = 0;
        for (targetsIter = iiter->_targets.begin(); targetsIter != targetsEnd;
             ++targetsIter, ++d)
        {
          std::map<std::string, int>::iterator miter =
              _chemicalSynapseTypesMap.find(targetsIter->first);
          if (miter != _chemicalSynapseTypesMap.end())
          {
            int type = miter->second;
            assert(type < typeCounter.size());
            if (mixedSynapse && !nonGenerated && !generated)
              nonGenerated =
                  isNonGenerated(titer, targetsIter->first, typeCounter[type]);
            if (!generated && !nonGenerated)
              generated = isGenerated(_generatedChemicalSynapses, titer, type,
                                      typeCounter[type]);
            if (targetsIter->first == nodeType) hit = true;
            typeCounter[type]++;
          }
        }
        if (hit)
        {
          if (generated)
            probabilities.push_back(1.0);
          else if (nonGenerated)
            probabilities.push_back(0.0);
          else
            probabilities.push_back(iiter->_parameter);
        }
      }
    }
  }
}

void TissueFunctor::getBidirectionalConnectionProbabilities(
    std::vector<double>& probabilities, TouchVector::TouchIterator& titer,
    std::string nodeType)
{
  key_size_t key1 = titer->getKey1();
  key_size_t key2 = titer->getKey2();
  if (_tissueParams.bidirectionalConnections() &&
      (key1 < key2 ||
       !_tissueParams.symmetricBidirectionalConnectionTargets(key1, key2)))
  {
    std::list<Params::BidirectionalConnectionTarget>* synapseTypes =
        _tissueParams.getBidirectionalConnectionTargets(key1, key2);
    if (synapseTypes)
    {
      std::list<Params::BidirectionalConnectionTarget>::iterator
          iiter = synapseTypes->begin(),
          iend = synapseTypes->end();
      for (; iiter != iend; ++iiter)
      {
        if (iiter->_type == nodeType)
          probabilities.push_back(iiter->_parameter);
      }
    }
  }
}

//GOAL for a given touch (with 2 Capsules) referenced by the iterator 'titer'
//  if the data for defining a chemical-synapse is provided (via SynParam.par file)
//given nodeType (e.g. AMPA)
// 
//  check if such nodeType (e.g. AMPA)
//  is defined as belonging to ChemicalSynapse for that touch 'titer'
bool TissueFunctor::isPointRequired(TouchVector::TouchIterator& titer,
                                    std::string nodeType)
{
  bool rval = false;
  key_size_t key1 = titer->getKey1();
  key_size_t key2 = titer->getKey2();
  if (_tissueParams.chemicalSynapses())
  {
    std::list<Params::ChemicalSynapseTarget>* synapseTypes =
        _tissueParams.getChemicalSynapseTargets(key1, key2);
    if (synapseTypes)
    {
      std::list<Params::ChemicalSynapseTarget>::iterator
          iiter = synapseTypes->begin(),
          iend = synapseTypes->end();
      for (; iiter != iend && !rval; ++iiter)
      {
        //<"AMPA", pair(list-of-nodeType-it-get-inputs,
        //              list-of-nodeType-it-perturb) >
        std::map<std::string, std::pair<std::list<std::string>,
                                        std::list<std::string> > >::iterator
            targetsIter,
            targetsEnd = iiter->_targets.end();
        for (targetsIter = iiter->_targets.begin();
             targetsIter != targetsEnd && !rval; ++targetsIter)
        {
          if (targetsIter->first == nodeType) rval = true;
        }
      }
    }
  }
  return rval;
}

// GOAL: the method setup the map 'smap'
//       during doLayout()
//       which is then used during doConnect()
//       via isGenerated() method to see if instances are created (e.g.
//       Connexon, SpineAttachment)
// PARAMS:
//   titer = the iterator pointing to the given Touch element in TouchVector
//                _tissueContext->_touchVector
//   type = the integer value indicating the type of touch (i.e. mapping to the nodekind value)
//           such as 'DenSpine' (if order=2), or 'Voltage' (if order=3)
//           NOTE: if order=1, we can have more than one case for type: 'AMPA', 'NMDA', 'GABAA'
//
//   order =  the integer value indicating what kind of touch (e.g. 0 = electrical synapse,
//                         1 = chemical synapse
//                         2 = spineneck-denshaft touch
//                         3 = synapticCleft or preSynapticPoint)
//          for one 'order' value, we can have different types of that kind of touch
//  RETURN:
//   true  = if a new one is added to the map
//   false = if not
bool TissueFunctor::setGenerated(
    std::map<Touch*, std::list<std::pair<int, int> > >& smap,
    TouchVector::TouchIterator& titer, int type, int order,
    std::string specialTreatment)
{
  std::map<Touch*, std::list<std::pair<int, int> > >::iterator miter =
      smap.find(&(*titer));
  bool newlyCreated = false;
  // (specialTreatment == "BidirectionalConnections")
  bool touchSpineNeckDenShaftIsUsed = false;
  bool newTouchIsNeckDenShaft = false;
  // (specialTreatment == "SynapticClefts" || specialTreatment == "PreSynapticPoints")
  bool touchChemSynapseIsUsed = false;//both Bouton-Spinehead and Bouton-DenShaft
  bool newTouchIsChemSynapse = false;
  bool newTouchIsSpinelessChemSynapse = false;
  // endspecialTreatment
  bool alreadyThere = false;
  key_size_t dummy_key;
  //ignore special treatment
  if (_tissueParams._use_biological_constraint == 0)
    specialTreatment = "";
  ///////
  if (miter == smap.end())
  {
    if (specialTreatment == "BidirectionalConnections")
    {  // detect if any capsule of the Touch
       // already involved in
      // another bidirectional-connection. If so, ignore it (or
      //...we can decide which one to keep)
      key_size_t key1 = (*titer).getKey1();
      key_size_t key2 = (*titer).getKey2();
      key_size_t key_spineneck = -1.0;
#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
      newTouchIsNeckDenShaft = (*titer).isSpineNeck_n_DenShaft(key_spineneck, _tissueParams);
#else
      newTouchIsNeckDenShaft = (*titer).hasSpineNeck(key_spineneck); //obsolete
#endif
      double propSpineNeck = 0;
      propSpineNeck =
          (key1 == key_spineneck) ? (*titer).getProp1() : (*titer).getProp2();

      // more constraint
      // newTouchIsNeckDenShaft = newTouchIsNeckDenShaft and (propSpineNeck >
      // 0.99);
      // Using propSpineNeck == 1.0 help to reduce the chance for the below bug
      // to occur
      // TUAN TODO: there is still potential bug when the spine neck is cut by
      // the
      //  TissueSlicer  and thus a SpineAttachment* is created but not sure on
      //  which
      //  MPIProcess
      // newTouchIsNeckDenShaft = newTouchIsNeckDenShaft and (propSpineNeck ==
      // 1.0);
      Capsule* neckCapsule =
          &_tissueContext
               ->_capsules[_tissueContext->getCapsuleIndex(key_spineneck)];
      ComputeBranch* branch = neckCapsule->getBranch();
      newTouchIsNeckDenShaft = newTouchIsNeckDenShaft and
                               //  (propSpineNeck > 0.98) and
                               branch->_daughters.size() == 0;
      if (newTouchIsNeckDenShaft)
      {
        bool isSide1 = (key_spineneck == key1);

#ifndef LTWT_TOUCH
        double dist = (*titer).getDistance();
#endif
        int count = 0;
        double propSpineNeck_inloop = 0.0;
        Touch* touchCanBeRemoved = 0;
        for (std::map<Touch*, std::list<std::pair<int, int> > >::const_iterator
                 it = smap.begin();
             it != smap.end(); ++it)
        {
          Touch* touch = it->first;
          key_size_t key1_inloop = (*touch).getKey1();
          key_size_t key2_inloop = (*touch).getKey2();
          key_size_t key_inloop_spineneck = +1.0;
#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
          (*touch).hasSpineNeck(key_inloop_spineneck, _tissueParams);
#else
          (*touch).hasSpineNeck(key_inloop_spineneck);//obsolete
#endif
          propSpineNeck_inloop = (key1_inloop == key_inloop_spineneck)
                                     ? (*touch).getProp1()
                                     : (*touch).getProp2();
          bool isSide1_inloop = (key_inloop_spineneck == key1_inloop);
#ifndef LTWT_TOUCH
          double dist_inloop = (*touch).getDistance();
#endif
          // equal and must be on the same side (i.e. either both first keys or
          //                                    both second keys)
          // if (key_spineneck == key_inloop_spineneck)
          if ((key_spineneck == key_inloop_spineneck)
              //  and (isSide1 == isSide1_inloop)
              )
          {
            if (0)
            {  // debug purpose
              std::cout << "found " << key1 << " " << key2 << " "
                        << key_spineneck << " " << key1_inloop << " "
                        << key2_inloop << " " << key_inloop_spineneck << " "
                        << isSide1 << isSide1_inloop << " " << propSpineNeck
                        << " " << propSpineNeck_inloop
#ifndef LTWT_TOUCH
                        << dist << " " << dist_inloop
#endif
                        << std::endl;
            }
            // count += 1;
            // if (count ==2)
            //{
            touchSpineNeckDenShaftIsUsed = true;
            touchCanBeRemoved = it->first;
            // TUAN TODO
            // we may use the comparison between propSpineNeck and
            // propSpineNeck_inloop
            // to determine which connection is the right one for the spine
            break;
            //}
          }
          // string value = it->second;
        }
        // if ((touchSpineNeckDenShaftIsUsed) and (propSpineNeck >
        // propSpineNeck_inloop))
        // {//the newer touch is considered better reflect the
        // spineneck-denshaft touch
        //     //remove the existing one
        //     assert(smap.erase(touchCanBeRemoved) ==1);
        //     //make sure the new one will be added
        //     touchSpineNeckDenShaftIsUsed=false;
        // }
        if (!touchSpineNeckDenShaftIsUsed)
        {
          smap[&(*titer)] = std::list<std::pair<int, int> >();
          miter = smap.find(&(*titer));
          newlyCreated = true;
        }
      }
    }
    else if (specialTreatment == "SynapticClefts" ||
             specialTreatment == "PreSynapticPoints")
    {  // first make sure this Touch is a
       // valid ChemicalSynapse touch
      if ((*titer).isSpineless(dummy_key))
      {  // spine-less touch
        key_size_t key1 = (*titer).getKey1();
        key_size_t key2 = (*titer).getKey2();
        key_size_t axon_key = -1.0;
        newTouchIsSpinelessChemSynapse = (*titer).isSpineless(axon_key);
        double propBouton =
            (key1 == axon_key) ? (*titer).getProp1() : (*titer).getProp2();

        // more constraint
        newTouchIsSpinelessChemSynapse =
            newTouchIsSpinelessChemSynapse && (propBouton > 0.99);
        // newTouchIsSpinelessChemSynapse = newTouchIsSpinelessChemSynapse &&
        // (propBouton == 1.0);
        if (newTouchIsSpinelessChemSynapse)
        {  // detect if any capsule of the
           // Touch already involved in
          // another chemical synapse-connection. If so, ignore it (or
          //...we can decide which one to keep)

          bool isSide1 = (axon_key == key1);
#ifndef LTWT_TOUCH
          double dist = (*titer).getDistance();
#endif
          int count = 0;
          double propBouton_inloop = 0.0;
          Touch* touchCanBeRemoved = 0;
          for (std::map<Touch*,
                        std::list<std::pair<int, int> > >::const_iterator it =
                   smap.begin();
               it != smap.end(); ++it)
          {
            Touch* touch = it->first;
            key_size_t key1_inloop = (*touch).getKey1();
            key_size_t key2_inloop = (*touch).getKey2();
            key_size_t key_inloop_axon = +1.0;
#ifdef TOUCHDETECT_SINGLENEURON_SPINES
          //Don't use in a tissue simulation scenario
          //as we want all possible 'GABAergic synapse' touches
          //are created - not to constraint that one 'axon' capsule can involve 
          //in only 1 'GABAergic synapse' touch
            if ((*touch).isSpineless(key_inloop_axon))
            {
              propBouton_inloop = (key1_inloop == key_inloop_axon)
                                      ? (*touch).getProp1()
                                      : (*touch).getProp2();
              bool isSide1_inloop = (key_inloop_axon == key1_inloop);
#ifndef LTWT_TOUCH
              double dist_inloop = (*touch).getDistance();
#endif
              // equal and must be on the same side 
              // (i.e. either both first keys
              // or
              // both second keys)
              if ((axon_key == key_inloop_axon)
                  //  and (isSide1 == isSide1_inloop)
                  )
              {
                if (0)
                {
                  std::cout << "found " << key1 << " " << key2 << " "
                            << axon_key << " " << key1_inloop << " "
                            << key2_inloop << " " << key_inloop_axon << " "
                            << isSide1 << isSide1_inloop << " " << propBouton
                            << " " << propBouton_inloop
#ifndef LTWT_TOUCH
                            << dist << " " << dist_inloop
#endif
                            << std::endl;
                }
                // count += 1;
                // if (count ==2)
                //{
                touchChemSynapseIsUsed = true;
                touchCanBeRemoved = it->first;
                // TUAN TODO
                // we may use the comparison between propBouton and
                // propBouton_inloop
                // to determine which connection is the right one for the spine
                break;
                //}
              }
            }
            // string value = it->second;
#endif
          }
          // TUAN TODO: try to think a way to do find the right touch
          // if ((touchChemSynapseIsUsed) and (propBouton > propBouton_inloop))
          // {//the newer touch is considered better reflect the
          // bouton-spineHead touch
          //     //remove the existing one
          //     assert(smap.erase(touchCanBeRemoved) ==1);
          //     //make sure the new one will be added
          //     touchChemSynapseIsUsed=false;
          // }
          if (!touchChemSynapseIsUsed)
          {
            smap[&(*titer)] = std::list<std::pair<int, int> >();
            miter = smap.find(&(*titer));
            newlyCreated = true;
          }
        }
      }
      else
      {  // spiny-touch
        key_size_t key1 = (*titer).getKey1();
        key_size_t key2 = (*titer).getKey2();
        key_size_t key_spinehead = -1.0;
#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
        newTouchIsChemSynapse = (*titer).hasSpineHead(key_spinehead, _tissueParams);
#else
        newTouchIsChemSynapse = (*titer).hasSpineHead(key_spinehead);//obsolete
#endif
        double propBouton =
            (key1 == key_spinehead) ? (*titer).getProp2() : (*titer).getProp1();

        // more constraint
        newTouchIsChemSynapse = newTouchIsChemSynapse && (propBouton > 0.99);
        if (newTouchIsChemSynapse)
        {  // detect if any capsule of the Touch
           // already involved in
          // another chemical synapse-connection. If so, ignore it (or
          //...we can decide which one to keep)

          bool isSide1 = (key_spinehead == key1);
#ifndef LTWT_TOUCH
          double dist = (*titer).getDistance();
#endif
          int count = 0;
          double propBouton_inloop = 0.0;
          Touch* touchCanBeRemoved = 0;
          for (std::map<Touch*,
                        std::list<std::pair<int, int> > >::const_iterator it =
                   smap.begin();
               it != smap.end(); ++it)
          {
            Touch* touch = it->first;
            key_size_t key1_inloop = (*touch).getKey1();
            key_size_t key2_inloop = (*touch).getKey2();
            key_size_t key_inloop_spinehead = +1.0;
#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
            (*touch).hasSpineHead(key_inloop_spinehead, _tissueParams);
#else
            (*touch).hasSpineHead(key_inloop_spinehead);//obsolete
#endif
            propBouton_inloop = (key1_inloop == key_inloop_spinehead)
                                    ? (*touch).getProp2()
                                    : (*touch).getProp1();
            bool isSide1_inloop = (key_inloop_spinehead == key1_inloop);
#ifndef LTWT_TOUCH
            double dist_inloop = (*touch).getDistance();
#endif
            // equal and must be on the same side (i.e. either both first keys
            // or
            //                                    both second keys)
            // if (key_spinehead == key_inloop_spinehead)
            if ((key_spinehead == key_inloop_spinehead)
                //  and (isSide1 == isSide1_inloop)
                )
            {
              if (0)
              {
                std::cout << "found " << key1 << " " << key2 << " "
                          << key_spinehead << " " << key1_inloop << " "
                          << key2_inloop << " " << key_inloop_spinehead << " "
                          << isSide1 << isSide1_inloop << " " << propBouton
                          << " " << propBouton_inloop
#ifndef LTWT_TOUCH
                          << dist << " " << dist_inloop
#endif
                          << std::endl;
              }
              // count += 1;
              // if (count ==2)
              //{
              touchChemSynapseIsUsed = true;
              touchCanBeRemoved = it->first;
              // TUAN TODO
              // we may use the comparison between propBouton and
              // propBouton_inloop
              // to determine which connection is the right one for the spine
              break;
              //}
            }
            // string value = it->second;
          }
          // TUAN TODO: try to think a way to do find the right touch
          // if ((touchChemSynapseIsUsed) and (propBouton > propBouton_inloop))
          // {//the newer touch is considered better reflect the
          // bouton-spineHead touch
          //     //remove the existing one
          //     assert(smap.erase(touchCanBeRemoved) ==1);
          //     //make sure the new one will be added
          //     touchChemSynapseIsUsed=false;
          // }
          if (!touchChemSynapseIsUsed)
          {
            smap[&(*titer)] = std::list<std::pair<int, int> >();
            miter = smap.find(&(*titer));
            newlyCreated = true;
          }
        }
      }
    }
    else if (specialTreatment == "ChemicalSynapses")
    {  // always add it here
      smap[&(*titer)] = std::list<std::pair<int, int> >();
      miter = smap.find(&(*titer));
      newlyCreated = true;
    }
    else
    {
      smap[&(*titer)] = std::list<std::pair<int, int> >();
      miter = smap.find(&(*titer));
      newlyCreated = true;
    }
  }
  else
    alreadyThere = true;

  bool rval = true;
  // Special treatment
  // for BidirectionalConnections, we only create for 1 touch
  if (specialTreatment == "BidirectionalConnections")
  {
    // if (newlyCreated or alreadyThere)
    if (newTouchIsNeckDenShaft and !touchSpineNeckDenShaftIsUsed)
      miter->second.push_back(std::pair<int, int>(type, order));
    else
      rval = false;
  }
  else if (specialTreatment == "SynapticClefts" ||
           specialTreatment == "PreSynapticPoints")
  {
    if ((newTouchIsChemSynapse || newTouchIsSpinelessChemSynapse) and
        !touchChemSynapseIsUsed)
      miter->second.push_back(std::pair<int, int>(type, order));
    else
      rval = false;
  }
  else
  {
    miter->second.push_back(std::pair<int, int>(type, order));
  }
  return rval;
}

std::list<Params::ChemicalSynapseTarget>::iterator
    TissueFunctor::getChemicalSynapseTargetFromOrder(
        TouchVector::TouchIterator& titer, std::string type, int order)
{
  std::list<Params::ChemicalSynapseTarget>::iterator rval, rend;
  key_size_t key1 = titer->getKey1();
  key_size_t key2 = titer->getKey2();
  assert(_tissueParams.chemicalSynapses());
  std::list<Params::ChemicalSynapseTarget>* synapseTypes =
      _tissueParams.getChemicalSynapseTargets(key1, key2);
  assert(synapseTypes);
  rval = synapseTypes->begin();
  rend = synapseTypes->end();
  while (order >= 0 && rval != rend)
  {
    std::map<std::string, std::pair<std::list<std::string>,
                                    std::list<std::string> > >::iterator
        targetsIter,
        targetsEnd = rval->_targets.end();
    int j = 0;
    for (targetsIter = rval->_targets.begin(); targetsIter != targetsEnd;
         ++targetsIter)
    {
      if (targetsIter->first == type) --order;
    }
    if (order >= 0) ++rval;
  }
  assert(rval != rend);
  return rval;
}

// check if the touch formed by two capsules
//  is 'generated' as the given 'order' or not
//   with order is an integer representing
//      'ChemicalSynapses' or 'ElectricalSynapses' or 'BidirectionalConnections'
//   and 'type' is an integer representing the name assigned for one 'order'
//      e.g. 'DenSpine' for ChemicalSynapses
//   (NOTE: one 'order' can have one or many 'name' defined via SynParam.par file)
//  PARAMS:
//   smap = the map holding what touch has been added,
//               and is used for what pair of (type,order)
//  NOTE: It is called during doConnect
//   and the maps 'smap' is setup (created) during
//    doLayout via setGenerated()  method
bool TissueFunctor::isGenerated(
    std::map<Touch*, std::list<std::pair<int, int> > >& smap,
    TouchVector::TouchIterator& titer, int type, int order)
{
  bool rval = false;
  std::map<Touch*, std::list<std::pair<int, int> > >::iterator miter =
      smap.find(&(*titer));
  if (miter != smap.end())
  {
    if (type < 0)
      rval = true;
    else
    {
      std::list<std::pair<int, int> >& l = miter->second;
      std::list<std::pair<int, int> >::iterator liter, lend = l.end();
      for (liter = l.begin(); liter != lend; ++liter)
      {
        if (liter->first == type && liter->second == order)
        {
          rval = true;
          break;
        }
      }
    }
  }
  return rval;
}

// GOAL
//  type = a string representing a nodetype can be "GABAA"
//  order = 
void TissueFunctor::setNonGenerated(TouchVector::TouchIterator& titer,
                                    std::string type, int order)
{
  std::list<Params::ChemicalSynapseTarget>::iterator iiter =
      getChemicalSynapseTargetFromOrder(titer, type, order);
  if (iiter->_targets.size() > 1)
  {
    _nonGeneratedMixedChemicalSynapses[&(*titer)].push_back(iiter);
  }
}

bool TissueFunctor::isNonGenerated(TouchVector::TouchIterator& titer,
                                   std::string nodeType, int order)
{
  bool rval = false;
  std::list<Params::ChemicalSynapseTarget>::iterator iiter =
      getChemicalSynapseTargetFromOrder(titer, nodeType, order);
  if (iiter->_targets.size() > 1)
  {
    std::map<Touch*,
             std::list<std::list<Params::ChemicalSynapseTarget>::iterator> >::
        iterator miter = _nonGeneratedMixedChemicalSynapses.find(&(*titer));
    if (miter != _nonGeneratedMixedChemicalSynapses.end())
    {
      if (find(miter->second.begin(), miter->second.end(), iiter) !=
          miter->second.end())
        rval = true;
    }
  }
  return rval;
}

RNG& TissueFunctor::findSynapseGenerator(int preRank, int postRank)
{
  if (preRank == postRank) return _tissueContext->_localSynapseGenerator;
  int rank1 = MIN(preRank, postRank);
  int rank2 = MAX(preRank, postRank);

  std::map<int, std::map<int, RNG> >::iterator miter =
      _synapseGeneratorMap.find(rank1);
  if (miter == _synapseGeneratorMap.end())
  {
    RNG rng;
    resetSynapseGenerator(rng, rank1, rank2);
    std::map<int, RNG> smap;
    smap[rank2] = rng;
    _synapseGeneratorMap[rank1] = smap;
    return _synapseGeneratorMap[rank1][rank2];
  }
  else
  {
    std::map<int, RNG>::iterator miter2 = miter->second.find(rank2);
    if (miter2 == miter->second.end())
    {
      RNG rng;
      resetSynapseGenerator(rng, rank1, rank2);
      miter->second[rank2] = rng;
      return miter->second[rank2];
    }
    else
    {
      return miter2->second;
    }
  }
}

void TissueFunctor::resetSynapseGenerator(RNG& rng, int rank1, int rank2)
{
  rng.reSeedShared(_tissueContext->_boundarySynapseGeneratorSeed);
  for (int i = 0; i < rank1; ++i)
  {
    long nextSeed = lrandom(rng);
    rng.reSeedShared(nextSeed);
  }
  rng.reSeedShared(lrandom(rng) + rank2);
}

TissueFunctor::~TissueFunctor()
{
#ifdef HAVE_MPI
  --_instanceCounter;
  if (_instanceCounter == 0)
  {
    std::map<ComputeBranch*, std::vector<CG_CompartmentDimension*> >::iterator
        miter,
        mend = _tissueContext->_branchDimensionsMap.end();
    for (miter = _tissueContext->_branchDimensionsMap.begin(); miter != mend;
         ++miter)
    {
      std::vector<CG_CompartmentDimension*>::iterator viter,
          vend = miter->second.end();
      for (viter = miter->second.begin(); viter != vend; ++viter)
      {
        delete *viter;
      }
    }
    std::map<Capsule*, CG_CompartmentDimension*>::iterator miter2,
        mend2 = _tissueContext->_junctionDimensionMap.end();
    for (miter2 = _tissueContext->_junctionDimensionMap.begin();
         miter2 != mend2; ++miter2)
    {
      delete miter2->second;
    }
    std::map<ComputeBranch*, CG_BranchData*>::iterator miter3,
        mend3 = _tissueContext->_branchBranchDataMap.end();
    for (miter3 = _tissueContext->_branchBranchDataMap.begin(); miter3 != mend3;
         ++miter3)
    {
      delete miter3->second;
    }
    std::map<Capsule*, CG_BranchData*>::iterator miter4,
        mend4 = _tissueContext->_junctionBranchDataMap.end();
    for (miter4 = _tissueContext->_junctionBranchDataMap.begin();
         miter4 != mend4; ++miter4)
    {
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
  int rval = 0;
  std::map<int, int>::iterator miter = cmap.find(index);
  if (miter == cmap.end())
    cmap[index] = 1;
  else
  {
    rval = miter->second;
    ++miter->second;
  }
  return rval;
}

// GOAL: for a given capsule (in the associated ComputeBranch),
// returns the index of the compartment it belongs to
//  NOTE:
//      The explicit branching junction and implicit branching junction
//      is not considered here
//  NOTE: If the capsule is at distal-end
//      check for implicit junction --> for that index should be zero
//   As the implicit branching junction has no real capsule matching
//    (we just assume ignore that capsule if it presents)
#ifndef IDEA1
int TissueFunctor::getCptIndex(Capsule* capsule)
{
  int cptIndex = 0;
  ComputeBranch* branch = capsule->getBranch();
  std::vector<int> cptsizes_in_branch;
  int ncpts = getNumCompartments(branch, cptsizes_in_branch);
  int cps_index =
      (capsule - capsule->getBranch()->_capsules);  // zero-based index
  //# capsules in that branch
  int ncaps = branch->_nCapsules;
  int cps_index_reverse = ncaps - cps_index - 1;  // from the distal-end
  std::vector<int>::iterator iter = cptsizes_in_branch.begin(),
                             iterend = cptsizes_in_branch.end();
  int count = 0;

  for (; iter < iterend; iter++)
  {
    count = count + *iter;
    if (count >= cps_index_reverse)
    {
      break;
    }
    cptIndex++;
  }
  // DEBUG PURPOSE
  // if (cptIndex >= cptsizes_in_branch.size())
  //{
  //    std::cerr << "Total compartments: " << cptsizes_in_branch.size()
  //  	     << "#capsules " << ncaps
  //  	     << "cps_index " << cps_index
  //          << "index to compart " << cps_index_reverse << std::endl;
  //    iter = cptsizes_in_branch.begin();
  //    count = 0;
  //    for (; iter < iterend; iter++)
  //    {
  //  	  std::cerr << count << " " << *iter << std::endl;
  //  	  count = count + *iter;
  //  	  /*if  (count >= cps_index_reverse){
  //  		break;
  //  		}*/
  //  	  cptIndex++;
  //    }
  //}//END DEBUG PURPOSE
  assert(cptIndex < cptsizes_in_branch.size());
  return cptIndex;
}
#endif

#ifdef IDEA1
// NOTE: This is a replacement for the previous function
// as it handle the case of rescaling branch-point junction based on '-r' option
//int TissueFunctor::getCptIndex(Capsule* capsule, Touch & touch)
//{
//  int cptIndex = 0;
//  //TUAN TODO NOW update
//  ComputeBranch* branch = capsule->getBranch();
////  std::vector<int>* cptsizes_in_branch = (_cptSizesForBranchMap[branch]);
//  std::map<ComputeBranch*, std::vector<int> >::iterator miter =
//      _cptSizesForBranchMap.find(branch);
//  //assert(miter != _cptSizesForBranchMap.end());
//  if (miter == _cptSizesForBranchMap.end())
//  {
//      std::vector<int> cptsizes_in_branch;
//      bool isDistalEndSeeImplicitBranchingPoint;
//      int ncpts = getNumCompartments(branch, cptsizes_in_branch,
//              isDistalEndSeeImplicitBranchingPoint);
//      miter =
//          _cptSizesForBranchMap.find(branch);
//
//  }
//  std::vector<int>* cptsizes_in_branch = &(miter->second); 
//  if (isPartofJunction(capsule, touch))
//  {
//    cptIndex = 0;
//  }
//  else{
//    assert(cptsizes_in_branch->size()>0);
//    int cps_index =
//      (capsule - capsule->getBranch()->_capsules);  // zero-based index
//    //# capsules in that branch
//    int ncaps = branch->_nCapsules;
//    int cps_index_reverse = ncaps - cps_index - 1;  // from the distal-end
//    std::vector<int>::iterator iter = cptsizes_in_branch->begin(),
//      iterend = cptsizes_in_branch->end();
//
//    int count = 0;
//    for (; iter < iterend; iter++)
//    {
//      count = count + *iter;
//      if (count >= cps_index_reverse)
//      {
//        break;
//      }
//      cptIndex++;
//    }
//
//  }
//  assert(cptIndex < cptsizes_in_branch->size());
//  return cptIndex;
//}
#endif

// GOAL: find the fractional volume of a single capsule on the proximal-side
// branch
//    connecting to the explicit [cut/branching] junction and/or implicit
//    branching junction
//    and reserve for the associated junction
dyn_var_t TissueFunctor::getFractionCapsuleVolumeFromPre(ComputeBranch* branch)
{
  dyn_var_t frac;
  // assert(branch->_parent);  // not soma
  Capsule caps = branch->_capsules[0];
  assert(_segmentDescriptor.getBranchType(caps.getKey()) !=
         Branch::_SOMA);  // not soma
  //if (branch->_nCapsules == 1)
  //  frac = 1.0 / 4.0;
  //else
  //  frac = 1.0 / 3.0;
  frac = 1.0 / 5.0;
  assert(frac > 0);
  assert(frac < 1);
  return frac;
}
// GOAL: find the fractional volume of a single capsule on the distal-side
// branch
//    connecting to the explicit [cut/branching] junction and/or implicit
//    branching junction
//    and reserve for the associated junction
dyn_var_t TissueFunctor::getFractionCapsuleVolumeFromPost(ComputeBranch* branch)
{
  dyn_var_t frac;
  // assert(branch->_parent);  // not soma
  Capsule caps = branch->_capsules[0];
  assert(_segmentDescriptor.getBranchType(caps.getKey()) !=
         Branch::_SOMA);  // not soma
  //if (branch->_nCapsules == 1)
  //{
  //  ComputeBranch* branch_parent = branch->_parent;
  //  assert(_segmentDescriptor.getBranchType(branch_parent->_capsules[0].getKey()) !=
  //      Branch::_SOMA);  // not soma
  //  // check if the branch is next to soma (as a major portion can be covered by
  //  // soma)
  //  // and thus we need to use a smaller fraction
  //  if (_segmentDescriptor.getBranchType(
  //          branch_parent->_capsules[0].getKey()) == Branch::_SOMA)
  //  {
  //    frac = (branch->_capsules[0].getLength() -
  //            branch_parent->_capsules[0].getRadius()) /
  //           branch->_capsules[0].getLength() / 4.0;
  //  }
  //  else
  //    frac = 1.0 / 4.0;
  //}
  //else
  //  frac = 1.0 / 3.0;
  frac = 1.0 / 5.0;

  assert(frac > 0);
  assert(frac < 1);
  return frac;
}

// REMARK: If upper ceiling is used, there is a chance that the last compartment
//     has only 1 capsule; while the other has 'ncaps'>1
// So the strategy:
//   1. use the floor instead of ceiling (but get 1 if the floor is 0)
//   2. distribute these remainder capsules to every compartments
//     a. if 1 remainder --> put to distal end
//     b. if 2 remainder --> put 1 to distal end, 1 to proximal end
//     c. repeat step a and b
//   from the distal-side
//   ncaps_branch = #caps on that branch
//   ncaps_cpt    = suggested #caps per compartment
//   The strategy make sure two compartments either having the same #capsules or
//   only 1 unit difference
//   NOTE: If the ComputeBranch face implicit branching, make one more
//   compartment which hold the implicit junction
// HISTORY:
//   v.1.1 : update the distribution in step 2
#ifdef IDEA1
//int TissueFunctor::getNumCompartments(ComputeBranch* branch)
//{
//    std::vector<int> cptsizes_in_branch;
//    return this->getNumCompartments(branch, cptsizes_in_branch);
//}
//int TissueFunctor::getNumCompartments(
//    ComputeBranch* branch, std::vector<int>& cptsizes_in_branch)
//{
//  if (branch->_configuredCompartment)
//  {//stop here to avoid recalculation
//      cptsizes_in_branch = branch->_cptSizesForBranch ;
//      return branch->_cptSizesForBranch.size();
//  }
//
//  int rval;
//  int ncpts;
//  //# capsules in that branch
//  int ncaps = branch->_nCapsules;
//  // we need this in case the ncaps is less than _compartmentSize
//  // e.g. soma has only 1 capsule
//  int cptSize = (ncaps > _compartmentSize) ? _compartmentSize : ncaps;
//  // Find: # compartments in the current branch
//  ncpts = (int(floor(double(ncaps) / double(cptSize))) > 0)
//              ? int(floor(double(ncaps) / double(cptSize)))
//              : 1;
//// suppose the branch is long enough, reverse some capsules at each end for
//// branchpoint
////  2. explicit slicing cut
////  3. explicit branchpoint
//  cptsizes_in_branch.clear();
//  Capsule* capPtr = &branch->_capsules[ncaps - 1];
//  key_size_t key = capPtr->getKey();
//  unsigned int computeOrder = _segmentDescriptor.getComputeOrder(key);
//  float reserved4proxend = 0.0;
//  float reserved4distend = 0.0;
//  //NOTE: '-r' #caps/cpt
//  if (ncaps == 1)
//  {// -r get any value 
//    cptSize = 1;
//    ncpts = 1;
//    //REGULAR treatment
//    if (computeOrder == 0 and computeOrder == MAX_COMPUTE_ORDER)
//    {
//        reserved4proxend = 0.25;
//        reserved4distend = 0.25;
//    }else if (computeOrder == 0)
//    {
//        reserved4proxend = 0.25;
//        reserved4distend = 0.0;
//    }else if (computeOrder == MAX_COMPUTE_ORDER)
//    {
//        reserved4proxend = 0.0;
//        reserved4distend = 0.25;
//    }
//    else{
//        reserved4proxend = 0.0;
//        reserved4distend = 0.0;
//    }
//    //SPECIAL treatment
//    if (branch->_parent)
//    {
//      Capsule& firstcaps = branch->_capsules[0];
//      Capsule& pcaps = branch->_parent->_capsules[0];
//      if (_segmentDescriptor.getBranchType(pcaps.getKey()) ==
//          Branch::_SOMA)  // the parent branch is soma
//      {
//          float length = firstcaps.getLength();
//          float somaR = pcaps.getLength(); //soma radius
//          if (length <= somaR)
//          {
//              std::cerr << "ERROR: There is 1-capsule branch from the soma, and the point falls within the soma'radius"
//                  << std::endl;
//              std::cerr << " ... Please make the capsule longer\n";
//              std::cerr << 
//                  "Neuron index: " << _segmentDescriptor.getNeuronIndex(pcaps.getKey()) 
//                  << std::endl;
//              double* coord = firstcaps.getBeginCoordinates();
//              std::cerr << "Coord: " << coord[0] << ", " << coord[1] << ", " << coord[2]
//              << std::endl;
//              assert(0);
//          }
//          else
//          {
//              reserved4proxend = somaR/length;
//              reserved4distend = 0.25 * (1.0-reserved4proxend); 
//          }
//      }
//    }
//    branch->_numCapsulesEachSideForBranchPoint = std::make_pair(reserved4proxend, reserved4distend);
//    cptsizes_in_branch.push_back(cptSize);
//  }
//  else if (ncaps == 2)
//  {// -r get any value
//    cptSize = 2;
//    ncpts = 1;
//    //REGULAR treatment
//    if (computeOrder == 0 and computeOrder == MAX_COMPUTE_ORDER)
//    {
//        reserved4proxend = 0.5;
//        reserved4distend = 0.5;
//    }else if (computeOrder == 0)
//    {
//        reserved4proxend = 0.5;
//        reserved4distend = 0.0;
//    }else if (computeOrder == MAX_COMPUTE_ORDER)
//    {
//        reserved4proxend = 0.0;
//        reserved4distend = 0.5;
//    }
//    else{
//        reserved4proxend = 0.0;
//        reserved4distend = 0.0;
//    }
//    //SPECIAL treatment
//    if (branch->_parent)
//    {
//      Capsule& firstcaps = branch->_capsules[0];
//      Capsule& pcaps = branch->_parent->_capsules[0];
//      if (_segmentDescriptor.getBranchType(firstcaps.getKey()) ==
//          Branch::_SOMA)  // the parent branch is soma
//      {//ignore the first capsule
//          float length = firstcaps.getLength();
//          float somaR = pcaps.getLength(); //soma radius
//          if (length <= somaR)
//          {//skip the first capsule 
//              reserved4proxend = 1.0;
//              reserved4distend = 0.25;
//          }else
//          {
//              reserved4proxend = somaR/length;
//              reserved4distend = 0.25;
//          } 
//      }
//    }
//    branch->_numCapsulesEachSideForBranchPoint = std::make_pair(reserved4proxend, reserved4distend);
//    cptsizes_in_branch.push_back(cptSize);
//  }
//  else if (ncaps >= 3)
//  {
//    float fcaps_loss = 0.0;
//    if (computeOrder == 0 and computeOrder == MAX_COMPUTE_ORDER)
//    {
//        if (_compartmentSize >= 2)
//        {
//            reserved4proxend = 0.75;
//            reserved4distend = 0.75;
//        }else{
//            reserved4proxend = 0.5;
//            reserved4distend = 0.5;
//        }
//    }else if (computeOrder == 0)
//    {
//        reserved4proxend = 0.5;
//        reserved4distend = 0.0;
//    }else if (computeOrder == MAX_COMPUTE_ORDER)
//    {
//        reserved4proxend = 0.0;
//        reserved4distend = 0.5;
//    }
//    else{
//        reserved4proxend = 0.0;
//        reserved4distend = 0.0;
//    }
//    //NOTE: adjust this we need to adjust "secA"
//    fcaps_loss = reserved4distend + reserved4proxend; //0.75 for proximal, 0.75 for distal end
//
//#define SMALL_FLT 0.00013
//    int caps_loss_prox = (reserved4proxend < SMALL_FLT) ? 0 : 1;
//    int caps_loss_dist = (reserved4distend < SMALL_FLT) ? 0 : 1;
//    int ncaps_loss =  caps_loss_prox + caps_loss_dist;
//    int tmpVal= int(floor(double(ncaps-ncaps_loss) / double(cptSize))); 
//    ncpts = (tmpVal > 0) ? tmpVal : 1;
//    cptsizes_in_branch.resize(ncpts);
//    std::fill(cptsizes_in_branch.begin(), cptsizes_in_branch.end(), 0);
//    //NOTE: reserve at each end
//    cptsizes_in_branch[0] = caps_loss_prox; //reserve 1 for proximal
//    cptsizes_in_branch[ncpts-1] = caps_loss_dist;//reserve 1 for distal
//    int caps_left = ncaps - ncaps_loss;
//
//    int count = 0;
//    do{
//      count++;
//      for (int ii = 0; ii < ncpts; ii++)
//      {
//        if (caps_left > 0)
//        {
//          cptsizes_in_branch[ii] += 1;
//          caps_left -= 1;
//        }
//        else
//          break;
//      }
//      if (count == 3)
//      {//every 3 capsules added to each cpt, there is 1 for prox.end branching point and 1 for dist.end branching point
//        if (caps_left > 0)
//        {
//          cptsizes_in_branch[ncpts-1] += 1;
//          caps_left -= 1;
//          reserved4distend += 1;
//        }
//        if (caps_left > 0)
//        {
//          cptsizes_in_branch[0] += 1;
//          caps_left -= 1;
//          reserved4proxend += 1;
//        }
//        count = 0;
//      }
//    }while (caps_left>0);
//
//    //SPECIAL treatment
//    if (branch->_parent)
//    {
//        Capsule& pcaps = branch->_parent->_capsules[0];
//        if (_segmentDescriptor.getBranchType(pcaps.getKey()) ==
//                Branch::_SOMA)  // the parent branch is soma
//        {//ignore the first capsule
//            reserved4proxend = (reserved4proxend >= 1.0) ? reserved4proxend : 1.0;
//        }
//    }
//    branch->_numCapsulesEachSideForBranchPoint = std::make_pair(reserved4proxend, reserved4distend);
//  }
//
//  branch->_cptSizesForBranch = cptsizes_in_branch;
//  branch->_configuredCompartment = true;
//  assert(cptsizes_in_branch.size() > 0);
//  //making sure no distal-end reserve for terminal branch
//  if (branch->_daughters.size() == 0)
//  {
//    branch->_numCapsulesEachSideForBranchPoint.second = 0.0;
//  }
//
//  //just for checking
//  int  sumEle = std::accumulate(cptsizes_in_branch.begin(), cptsizes_in_branch.end(), 0);
//  if (sumEle != ncaps)
//    std::cout << "numEle =" << sumEle << "; ncaps = " << ncaps << std::endl;
//  assert (sumEle == ncaps);
//
//  rval = ncpts;
//  return rval;
//}

#else
int TissueFunctor::getFirstIndexOfCapsuleSpanningSoma(ComputeBranch* branch)
{
  int firstIndexSpanningSoma = 0; //index of the first capsule that span the soma-membrane
  ComputeBranch* parentbranch = branch->_parent;
  if (parentbranch)
  {
    Capsule& firstcaps = parentbranch->_capsules[0];
    int ncaps = branch->_nCapsules;
    if (_segmentDescriptor.getBranchType(firstcaps.getKey()) ==
        Branch::_SOMA)
    {
      float somaR = firstcaps.getRadius();
      int i = 0;
      float dist2somaStart = branch->_capsules[i].getDist2Soma();
      float dist2somaEnd = dist2somaStart+branch->_capsules[i].getLength();
      if (branch->_capsules[ncaps-1].getDist2Soma() + branch->_capsules[ncaps-1].getLength() <= somaR)
      {
        std::cerr << "The first ComputeBranch falls within the soma on BRCHTYPE " << 
          _segmentDescriptor.getBranchType(branch->_capsules[ncaps-1].getKey())+ 1
          << " (NOTE: adjusted +1)"
          << std::endl;
        assert(branch->_capsules[ncaps-1].getDist2Soma() + branch->_capsules[ncaps-1].getLength() > somaR);
      }
      while (
          dist2somaStart < somaR and somaR > dist2somaEnd)
      {
        i++;
        dist2somaStart = branch->_capsules[i].getDist2Soma();
        dist2somaEnd = dist2somaStart+branch->_capsules[i].getLength();
        if (i > ncaps-1)
        {
          std::cerr << "ERROR: The branch stemming from soma is too short" << std::endl;
          assert(0);
        }
      }
      firstIndexSpanningSoma = i;
    }
  }
  return firstIndexSpanningSoma;
}
int TissueFunctor::getNumCompartments(
    ComputeBranch* branch, std::vector<int>& cptsizes_in_branch)
{
  int rval;
  int ncpts;
  //# capsules in that branch
  int ncaps = branch->_nCapsules;
  // we need this in case the ncaps is less than _compartmentSize
  // e.g. soma has only 1 capsule
  int cptSize = (ncaps > _compartmentSize) ? _compartmentSize : ncaps;


  int firstIndexSpanningSoma = 
    this->getFirstIndexOfCapsuleSpanningSoma(branch); //index of the first capsule that span the soma-membrane
  int extraCapsule4SomaAdjacentCpt = 0;
  ComputeBranch* parentbranch = branch->_parent;
  if (parentbranch)
  {
    Capsule& firstcaps = parentbranch->_capsules[0];
#define THRESHOLD_ACCEPTABLE_MIN_LENGTH_COMPARTMENT 3.0 //[um] micrometer
    if (_segmentDescriptor.getBranchType(firstcaps.getKey()) ==
        Branch::_SOMA)
    {
      float somaR = firstcaps.getRadius();
      if (branch->_capsules[firstIndexSpanningSoma].getDist2Soma()
          + branch->_capsules[firstIndexSpanningSoma].getLength() - somaR 
          <= THRESHOLD_ACCEPTABLE_MIN_LENGTH_COMPARTMENT and
          cptSize == 1)
      {
        extraCapsule4SomaAdjacentCpt = 1;
      }
    }
  }

  // Find: # compartments in the current branch
  ncpts = (int(floor(double(ncaps-firstIndexSpanningSoma-extraCapsule4SomaAdjacentCpt) / double(cptSize))) > 0)
              ? int(floor(double(ncaps-firstIndexSpanningSoma-extraCapsule4SomaAdjacentCpt) / double(cptSize)))
              : 1;
  int remainder_caps;
  remainder_caps = ncaps - firstIndexSpanningSoma - extraCapsule4SomaAdjacentCpt - cptSize * ncpts;
  cptsizes_in_branch.clear();
  //suppose 2cpts, each has 7caps, and remainder is 5
  //so increment = int(5/2)=2
  //to bring remainder smaller than #cpts
  //i.e. remainer =5-2*2=1
  //so cap1 = 7+increment+0
  //   cap2 = 7+increment+1
  int increment = int(floor(float(remainder_caps) / ncpts));
  remainder_caps = remainder_caps - ncpts * increment;
  /* RULE to reserve at each end for branch-point (suppose 3 branches)
    1caps --> branchpoint has <= 3 caps  
    1/2caps -->
    1/4caps -->  
   */
  for (int ii = 0; ii < ncpts; ii++)
  {
    // compartment indexing is distal to proximal, while 
    // capsule indexing is proximal to distal
    int ncapsule = cptSize;
    ncapsule += (ii < remainder_caps) ? increment + 1 : increment + 0;
    if (ii == ncpts-1)
      ncapsule += firstIndexSpanningSoma + extraCapsule4SomaAdjacentCpt;
    cptsizes_in_branch.push_back(ncapsule);
  }
  Capsule* capPtr = &branch->_capsules[ncaps - 1];
  key_size_t key = capPtr->getKey();
  unsigned int computeOrder = _segmentDescriptor.getComputeOrder(key);

  //just for checking
  int  sumEle = std::accumulate(cptsizes_in_branch.begin(), cptsizes_in_branch.end(), 0);
  if (sumEle != ncaps)
    std::cout << "numEle =" << sumEle << "; ncaps = " << ncaps << std::endl;
  assert (sumEle == ncaps);

  rval = ncpts;
  return rval;
}
int TissueFunctor::getNumCompartments(
    ComputeBranch* branch)
{
  std::vector<int> cptsizes_in_branch;
  int rval = getNumCompartments(branch, cptsizes_in_branch);
  return rval;
}
#endif

bool TissueFunctor::touchIsChemicalSynapse(
    std::map<Touch*, std::list<std::pair<int, int> > >& smap,
    TouchVector::TouchIterator& titer)
{
  bool rval = true;
  std::map<Touch*, std::list<std::pair<int, int> > >::iterator miter =
      smap.find(&(*titer));
  if (miter == smap.end())
  {
    rval = false;
  }
  return rval;
}

#ifdef DEBUG_CPTS
std::pair<float, float> TissueFunctor::getMeanSTD(int brType, 
    std::vector<std::pair<int, float> > cptData)
{
  float minVal, maxVal;
  getMeanSTD(brType, cptData, minVal, maxVal);
}
std::pair<float, float> TissueFunctor::getMeanSTD(int brType, 
    std::vector<std::pair<int, float> > cptData, float& minVal, float& maxVal)
{
  float sum = 0.0;
  float count = 0;
  minVal=FLT_MAX,maxVal=0.0;
  std::vector<std::pair<int, float> >::iterator iter = cptData.begin(),
    iend = cptData.end();
  for (; iter != iend; iter++)
  {
    if (iter->first == brType)
    {
      sum += iter->second;
      count++;
      minVal = (minVal > iter->second) ? iter->second : minVal;
      maxVal = (maxVal < iter->second) ? iter->second : maxVal;
    }
  }
  float mean = sum/count;
  float sumsq = 0.0;
  iter = cptData.begin();
  for (; iter != iend; iter++)
  {
    if (iter->first == brType)
    {
      sumsq += pow(iter->second-mean,2);
    }
  }
  float std = sqrt (sumsq/count);
  std::pair<float, float> result = std::make_pair(mean,std);
  return result;
}
#endif

#ifdef IDEA1
//GOAL: check if the capsule belong to the junction or not
//   The junction has to be explicit
//NOTE: This is a better strategy
// as the previous approach assumed that the junction takes the last capsule from 
//     the parent branch only, i.e. in a branchpoint, only the last capsule of the
//     parent branch is assigned with getFlag(key_thecapsule) == 1
//   However, this needs to be revised, as with '-r' option, the junction
//   may occupy more than one capsule at each side of the branchpoint
//bool TissueFunctor::isPartofJunction(Capsule* capsule, Touch &t)
//{
//  ComputeBranch* branch = capsule->getBranch();
//  float reserved4distend = _numCapsulesEachSideForBranchPointMap[branch].second;
//  float reserved4proxend = _numCapsulesEachSideForBranchPointMap[branch].first;
//  int cps_index =
//      (capsule - capsule->getBranch()->_capsules);  // zero-based index
//
//  //# capsules in that branch
//  int ncaps = branch->_nCapsules;
//  int cps_index_reverse = ncaps - cps_index - 1;  // from the distal-end
//  bool result = false;
//  key_size_t key = capsule->getKey();
//  unsigned int computeOrder =
//    _segmentDescriptor.getComputeOrder(key);
//  if (computeOrder == 0)
//  {
//    if (cps_index < reserved4proxend)
//      result = true;
//    if (cps_index == int(floor(reserved4proxend)))
//    {
//      result = (capsule->getEndProp() > t.getProp(capsule->getKey()));
//
//    }
//  }
//  else if (computeOrder == MAX_COMPUTE_ORDER)
//  {
//    if (cps_index_reverse < reserved4distend)
//      result = true;
//    if (cps_index_reverse == int(floor(reserved4distend)))
//    {
//      result = (capsule->getEndProp() <= t.getProp(capsule->getKey()));
//    }
//
//  }
//  if (_segmentDescriptor.getBranchType(capsule->getKey()) ==
//      Branch::_SOMA)  // the branch is a soma
//      result = true;
//  return result;
//}
#endif

//GOAL: return the name associated with an NTS-specific Layer defined via TissueFunctor
//  which include <category, name>
//Example: category = Compartment  |  Channel | ...
//         name     = Voltage 
std::pair<std::string, std::string> TissueFunctor::getCategoryTypePair(NDPairList::iterator& ndpiter, int& remaining)
{
  if ((*ndpiter)->getName()!="CATEGORY") {
    std::cerr<<"First parameter of TissueProbe must be PROBED or CATEGORY!"<<std::endl;
    exit(0);
  }
  StringDataItem* categoryDI = dynamic_cast<StringDataItem*>((*ndpiter)->getDataItem());
  if (categoryDI==0) {
    std::cerr<<"CATEGORY parameter of TissueProbe must be a string!"<<std::endl;
    exit(0);
  }
  std::string category=categoryDI->getString();
  std::string task(("TissueProbe"));
  assert(isValidCategoryString(category, task));

  int typeIdx;
  --ndpiter;
  --remaining;

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

  return std::pair<std::string, std::string>(category, type);
}

//GOAL: return the NTS-layer-index(category-dependent) based on the criteria passed in
//    which include CATEGORY, TYPE
// OUTPUT: esyn = True if electrical-synapse layer
int TissueFunctor::getTypeLayerIdx(std::string category, std::string type, bool& esyn)
{
  int typeIdx=-1;
  std::map<std::string, int>::iterator typeIter;
  esyn=false;
  if (category=="BRANCH") {
    typeIter=_compartmentVariableTypesMap.find(type);
    if (typeIter==_compartmentVariableTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  else if (category=="JUNCTION") {
    typeIter=_junctionTypesMap.find(type);
    if (typeIter==_junctionTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  else if (category=="CHANNEL") {
    typeIter=_channelTypesMap.find(type);
    if (typeIter==_channelTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  else if (category=="CHANNEL") {
    typeIter=_channelTypesMap.find(type);
    if (typeIter==_channelTypesMap.end()) {
      std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
      exit(0);
    }
    typeIdx=typeIter->second;
  }
  else if (category=="SYNAPSE") {
    typeIter=_chemicalSynapseTypesMap.find(type);
    if (typeIter==_chemicalSynapseTypesMap.end()) {
      typeIter=_electricalSynapseTypesMap.find(type);
      if (typeIter==_electricalSynapseTypesMap.end()) {
        std::cerr<<"Unrecognized TYPE during TissueProbe : "<<type<<" !"<<std::endl;
        exit(0);
      }
      else
        esyn=true;
    }
    typeIdx=typeIter->second;
  }
  else if (category == "CLEFT")
  {//Done
    typeIter = _synapticCleftTypesMap.find(type);
    //if (typeIter == _synapticCleftTypesMap.end())
    //{
    //  //typeIter = _preSynapticPointTypesMap.find(type);
    //  //if (typeIter == _preSynapticPointTypesMap.end())
    //  {
    //    std::cerr << "Unrecognized TYPE during TissueProbe : " << type << " !"
    //              << std::endl;
    //    preSynPoint = true;
    //    exit(EXIT_FAILURE);
    //  }
    //}
    typeIdx = typeIter->second;
  }
  else {
    std::cerr<<"Unrecognized CATEGORY on TissueFunctor : "<<category<<" !"<<std::endl;
    exit(0);
  }
  return typeIdx;
}

bool TissueFunctor::isValidCategoryString(std::string category, std::string task)
{
  bool result=true;
  if (category != "BRANCH" && category != "JUNCTION" && category != "CHANNEL" &&
      category != "SYNAPSE" && 
			category != "CLEFT")
    result = false;
  if (! result and _rank == 0)
  {
    std::cerr<<"Unrecognized CATEGORY during " << task <<": "<<category<<" !"<<std::endl;
  }
 return result; 
}

//GOAL: based on the category of layer, and index of layer in that category
//   return the Layer data (i.e. GridLayerDescriptor*)
GridLayerDescriptor* TissueFunctor::getGridLayerDescriptor(std::string category, int typeIdx, bool esyn)
{
  GridLayerDescriptor* layer;
  if (category=="BRANCH") layer=_compartmentVariableLayers[typeIdx];
  else if (category=="JUNCTION") layer=_junctionLayers[typeIdx];
  else if (category=="CHANNEL") layer=_channelLayers[typeIdx];
  else if (category=="SYNAPSE") layer = esyn ? _electricalSynapseLayers[typeIdx] : _chemicalSynapseLayers[typeIdx];
  else if (category == "CLEFT")
    layer = _synapticCleftLayers[typeIdx];

  return layer;
}


#ifdef MICRODOMAIN_CALCIUM
void TissueFunctor::checkValidUseMicrodomain(std::string compartmentNameOnly, std::string microdomainName)
{
  if (not microdomainName.empty())
  {//TUAN TODO: put this ever where in doLayout
    //if (compartmentNameOnly != "Calcium")
    if (not nodeTypeWithAllowedMicrodomain(compartmentNameOnly))
    {
      std::cerr <<  "IMPORTANT: microdomain is only supported for 'Calcium' compartment. You use it for "
        << compartmentNameOnly << std::endl;
      assert(0);
    }
  }

}
bool TissueFunctor::nodeTypeWithAllowedMicrodomain(std::string nodeType)
{
  bool result = true;
  if (nodeType!= "Calcium")
  {
    result = false;
  }
  return result;
}
#endif
