// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

#include "MaxComputeOrder.h"

class CG_CompartmentDimension;
class CG_BranchData;

class TissueContext
{
  public:
  TissueContext();
  ~TissueContext();

  typedef enum 
  {
      UNDEFINED = 0,
      PROXIMAL = 1,
      DISTAL = 2,
      SOMA = 3
  } CapsuleAtBranchStatus;

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
  // keeps track of all neuron in this computing node
  // neuron is uniquely identified via its index - an integer
  //    index is defined based on the order in tissue.txt file
  // each neuron has a list of branches (ComputeBranch)
  std::map<unsigned int, std::vector<ComputeBranch*> > _neurons;

  //define the mapping from a branch (represented by ComputeBranch)
  //    to the vector of all compartments in that branch
	// a compartment is the representation of a number of Capsules
	// and comprises
  //   (1) a centroid-point  
  //    that has the same value of the compartmental variable in the solver
  // a compartment is represented by CG_CompartmentDimension class
  //           VER.1.0:  which is (x,y,z,r,dist2soma)
  //           VER.2.0:  which is (r,dist2soma, surface_area)
  std::map<ComputeBranch*, std::vector<CG_CompartmentDimension*> >
      _branchDimensionsMap;
  //define the mapping from a branch
  //    to the data uniquely identified the branch (CG_BranchData*)
  //              which is (key,size)
  //      key  = a key formed by multiple fields that uniquely identify that branch
  //      size = # of compartments in that branch
  std::map<ComputeBranch*, CG_BranchData*> _branchBranchDataMap;
  //define the mapping between a capsule (the last capsule
	//        on the branch where junction start)
  //    to junction-compartment (that joint to the the same junction point)
  std::map<Capsule*, CG_CompartmentDimension*> _junctionDimensionMap;
  //define the linkage between a capsule (as a junction point)
  //    to the branch to which it belongs 
  std::map<Capsule*, CG_BranchData*> _junctionBranchDataMap;
  //define the mapping from a key (representing a branch)
  //    to the index of a capsule
  std::map<key_size_t, int> _firstPassCapsuleMap,
      _secondPassCapsuleMap;  // key to capsule index
#ifdef IDEA1
  // for ComputeBranch that are 'improper', then it tells how many capsules in the 
  //   right ComputeBranch, which resides on a different MPI process
  std::map<ComputeBranch*, int> _improperComputeBranchCorrectedCapsuleCountsMap; // improper: CBs that start in another rank, end either (in this rank or in another rank)
  Params* _params;
  void makeProperComputeBranch();
#endif

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

  void rebalance(Params* params, TouchVector* touchVector);

#ifdef IDEA1
  //bool isPartOfJunction(Capsule& capsule, Touch &t);
  //bool isPartOfJunction(Capsule& capsule, Touch &t, int& rank, Capsule* junctionCapsule);
  //bool isPartOfJunction(Capsule& capsule, Touch &t, int &rank);
  //bool isPartOfImplicitJunction(Capsule& capsule, Touch &t, int &status, int &rank);
  //int getJunctionMPIRank(Capsule &caps);
  bool isPartOfExplicitJunction(Capsule& capsule, Touch &t, CapsuleAtBranchStatus& status, int &rank);
  bool isPartOfExplicitJunction(Capsule& capsule, Touch &t, int& rank, Capsule** junctionCapsule);
  bool isPartOfExplicitJunction(Capsule& capsule, Touch &t, int& rank);
  bool isPartOfExplicitJunction(Capsule& capsule, Touch &t);
  bool isPartOfExplicitJunction(Capsule& capsule, Touch &t, 
        CapsuleAtBranchStatus& status, int& rank, Decomposition* decomposition);
  int getCptIndex(Capsule& caps, Touch & t);
  int getCptIndex(Capsule* caps, Touch & touch);
  int getNumCompartments(ComputeBranch* branch);
  int getNumCompartments(
    ComputeBranch* branch, std::vector<int>& cptsizes_in_branch);
#endif

  private:
  bool isSpanning(Capsule&);  // check if a Capsule span the slicing boundary
  bool sameBranch(Capsule&, unsigned int, unsigned int, unsigned int,
                  DetectionPass);
  // capsule is in the branch identified by neuron, bran0ch indices
  bool isGoing(Capsule&, int);   // first coord of capsule is in this volume,
                                 // while second is in another
  bool isComing(Capsule&, int);  // first coord of capsule is another volume,
                                 // while second is in this
  bool isOutside(ComputeBranch*, int);
  bool isConsecutiveCapsule(int index);
#ifdef IDEA1
  bool isProperSpanning(ComputeBranch* branch, ShallowArray<int, MAXRETURNRANKS, 100>& endRanks);
  bool isImproperSpanning(ComputeBranch* branch, int& beginRank);
#endif

  bool _initialized;
  SegmentDescriptor _segmentDescriptor;
  bool _seeded;
  int _rank, _mpiSize;
};
#endif
