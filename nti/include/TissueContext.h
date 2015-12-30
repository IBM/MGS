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

  typedef enum
  {
    NOT_SET = 0,
    FIRST_PASS,
    SECOND_PASS,
    ADDITIONAL_PASS
  } DetectionPass;
  NeuroDevCommandLine _commandLine;

  int _nCapsules; // total # capsules in the tissue partitioned to the current MPI process
  Capsule* _capsules; // array of all capsules, starting from soma
              //NOTE: A soma is a capsule with 2 ends at the same coordinate, same radius
  Capsule* _origin;
  TouchVector _touchVector;

  NeuronPartitioner* _neuronPartitioner;
  Decomposition* _decomposition;
  Tissue* _tissue;
  RNG _touchSampler, _localSynapseGenerator;
  long _boundarySynapseGeneratorSeed, _localSynapseGeneratorSeed;

  //define the mapping from a neuron (based on its index)
  //       to the list of all branches for that neuron 
  std::map<unsigned int, std::vector<ComputeBranch*> > _neurons;

  int getRank() { return _rank; }
  int getMpiSize() { return _mpiSize; }
  void readFromFile(FILE* dataFile, int size, int rank);
  void writeToFile(int size, int rank);
  void writeData(FILE* data);
  int setUpCapsules(int nCapsules, DetectionPass detectionPass, int rank,
                    int maxComputeOrder);
  void setUpBranches(int rank, int maxComputeOrder);
  bool isInitialized() { return _initialized; }
  void setInitialized() { _initialized = true; }
  void resetBranches();

  unsigned int getRankOfBeginPoint(ComputeBranch*);
  unsigned int getRankOfEndPoint(ComputeBranch*);
  bool isTouchToEnd(Capsule& c, Touch& t);
  bool isMappedTouch(Touch& t, std::map<key_size_t, int>::iterator& iter1,
                     std::map<key_size_t, int>::iterator& iter2);
  bool isLensTouch(Touch& t, int rank);
  int getCapsuleIndex(key_size_t key);
  void correctTouchKeys(int rank);
  DetectionPass getPass(key_size_t key);
  DetectionPass addToCapsuleMap(key_size_t key, int index, DetectionPass);
  void clearCapsuleMaps();
  void seed(int rank);

  void getCapsuleMaps(std::map<key_size_t, int>& firstPassCapsuleMap,
                      std::map<key_size_t, int>& secondPassCapsuleMap);
  void resetCapsuleMaps(std::map<key_size_t, int>& firstPassCapsuleMap,
                        std::map<key_size_t, int>& secondPassCapsuleMap);
  //define the mapping from a branch (represented by ComputeBranch)
  //    to the vector of all points in that branch
  //            each point is represented by CG_CompartmentDimension class
  //             which is (x,y,z,r,dist2soma)
  std::map<ComputeBranch*, std::vector<CG_CompartmentDimension*> >
      _branchDimensionsMap;
  //define the mapping from a branch
  //    to the data uniquely identified the branch (CG_BranchData*)
  //              which is (key,size)
  //      key  = a key formed by multiple fields that uniquely identify that branch
  //      size = # of compartments in that branch
  std::map<ComputeBranch*, CG_BranchData*> _branchBranchDataMap;
  //define the linkage between a single compartment (as a junction point)
  //         (from which the branching occurs)
  //    to a list of other compartments (that joint to the the same junction point)
  std::map<Capsule*, CG_CompartmentDimension*> _junctionDimensionMap;
  //define the linkage between a single compartment (as a junction point)
  //    to the branch to which it belongs 
  std::map<Capsule*, CG_BranchData*> _junctionBranchDataMap;
  //define the mapping from a key (representing a branch)
  //    to the index of a capsule
  std::map<key_size_t, int> _firstPassCapsuleMap,
      _secondPassCapsuleMap;  // key to capsule index

  void rebalance(Params* params, TouchVector* touchVector);

  private:
  bool isSpanning(Capsule&);  // first coord of capsule is in a different volume
                              // from second
  bool sameBranch(Capsule&, unsigned int, unsigned int, unsigned int,
                  DetectionPass);
  // capsule is in the branch identified by neuron, bran0ch indices
  bool isGoing(Capsule&, int);   // first coord of capsule is in this volume,
                                 // while second is in another
  bool isComing(Capsule&, int);  // first coord of capsule is another volume,
                                 // while second is in this
  bool isOutside(ComputeBranch*, int);
  bool isConsecutiveCapsule(int index);

  bool _initialized;
  SegmentDescriptor _segmentDescriptor;
  bool _seeded;
  int _rank, _mpiSize;
};
#endif
