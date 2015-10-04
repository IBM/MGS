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
// ================================================================

#ifndef TISSUECONTEXT_H
#define TISSUECONTEXT_H

#define TISSUE_CONTEXT_SEND_RECEIVE_PHASES 4

#define SIG_HIST 1000

#include <mpi.h>

#include "Sender.h"
#include "Receiver.h"
#include "NeuroDevCommandLine.h"
#include "Capsule.h"
#include "NeuronPartitioner.h"
#include "Decomposition.h"
#include "Tissue.h"
#include "Params.h"
#include "ComputeBranch.h"
#include "TouchVector.h"
#include "SegmentDescriptor.h"
#include "rndm.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <cassert>

class CG_CompartmentDimension;
class CG_BranchData;

class TissueContext
{
public:
  TissueContext();
  ~TissueContext();

  typedef enum{NOT_SET=0, FIRST_PASS, SECOND_PASS, ADDITIONAL_PASS} DetectionPass;
  NeuroDevCommandLine _commandLine;

  int _nCapsules;
  Capsule* _capsules;
  Capsule* _origin;
  TouchVector _touchVector;

  NeuronPartitioner* _neuronPartitioner;
  Decomposition* _decomposition;
  Tissue* _tissue;
  RNG _touchSampler, _localSynapseGenerator;
  long _boundarySynapseGeneratorSeed, _localSynapseGeneratorSeed;
  
  std::map<unsigned int, std::vector<ComputeBranch*> > _neurons;
  
  int getRank() {return _rank;}
  int getMpiSize() {return _mpiSize;}
  void readFromFile(FILE* dataFile, int size, int rank);
  void writeToFile(int size, int rank);
  void writeData(FILE* data);
  int setUpCapsules(int nCapsules, DetectionPass detectionPass, int rank, int maxComputeOrder);
  void setUpBranches(int rank, int maxComputeOrder);
  bool isInitialized() {return _initialized;}
  void setInitialized() {_initialized=true;}
   void resetBranches();

  unsigned int getRankOfBeginPoint(ComputeBranch*);
  unsigned int getRankOfEndPoint(ComputeBranch*);
  bool isTouchToEnd(Capsule& c, Touch& t);
  bool isMappedTouch(Touch& t, std::map<double, int>::iterator& iter1, std::map<double, int>::iterator& iter2);
  bool isLensTouch(Touch& t, int rank);
  int getCapsuleIndex(double key);
  void correctTouchKeys(int rank);
  DetectionPass getPass(double key);
  DetectionPass addToCapsuleMap(double key, int index, DetectionPass);
  void clearCapsuleMaps();
  void seed(int rank);

  void getCapsuleMaps(std::map<double, int>& firstPassCapsuleMap, std::map<double,int>& secondPassCapsuleMap);
  void resetCapsuleMaps(std::map<double, int>& firstPassCapsuleMap, std::map<double,int>& secondPassCapsuleMap);
  std::map<ComputeBranch*, std::vector<CG_CompartmentDimension*> > _branchDimensionsMap;
  std::map<ComputeBranch*, CG_BranchData* > _branchBranchDataMap;
  std::map<Capsule*, CG_CompartmentDimension*> _junctionDimensionMap;
  std::map<Capsule*, CG_BranchData* > _junctionBranchDataMap;
  std::map<double, int> _firstPassCapsuleMap, _secondPassCapsuleMap; // key to capsule index 

  void rebalance(Params* params, TouchVector* touchVector);

 private:
  bool isSpanning(Capsule&);             // first coord of capsule is in a different volume from second
  bool sameBranch(Capsule&, unsigned int, unsigned int, unsigned int, DetectionPass);      
  // capsule is in the branch identified by neuron, bran0ch indices
  bool isGoing(Capsule&, int);              // first coord of capsule is in this volume, while second is in another
  bool isComing(Capsule&, int);             // first coord of capsule is another volume, while second is in this
  bool isOutside(ComputeBranch*, int);
  bool isConsecutiveCapsule(int index);

  bool _initialized;
  SegmentDescriptor _segmentDescriptor;
  bool _seeded;
  int _rank, _mpiSize;
};
#endif

