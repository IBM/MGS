// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SEGMENTFORCEAGGREGATOR_H
#define SEGMENTFORCEAGGREGATOR_H
#define SEGMENTFORCEAGGREGATOR_RECEIVE_PHASES 3

#include <mpi.h>
#include "Receiver.h"
#include "SegmentDescriptor.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <list>


class Tissue;
class SegmentForce;
class SegmentForceAnalyzer;

class SegmentForceAggregator : public Receiver 

{
 public:
  SegmentForceAggregator(int rank, int nSlicers, int nSegmentForceDetectors, Tissue* tissue);
  ~SegmentForceAggregator();
  
  double aggregate(int frontNumber);
  double* getSegmentForces() {return _segmentForces;}
  int getNumberOfSegmentForces() {return _segmentForceDataSize;}
  void setNumberOfSegmentForces(int nSegmentForces) {_segmentForceDataSize=nSegmentForces;}
			
  int getRank() {return _rank;}

  void prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef);
  void* getRecvbuf(int receiveCycle, int receivePhase);
  int* getRecvcounts(int receiveCycle, int receivePhase);
  int* getRdispls(int receiveCycle, int receivePhase);
  MPI_Datatype* getRecvtypes(int receiveCycle, int receivePhase);
  int* getRecvSizes(int receiveCycle, int receivePhase);
  int getNumberOfReceiveCycles() {return 1;}
  int getNumberOfReceivePhasesPerCycle(int receiveCycle);
  void finalizeReceive(int receiveCycle, int receivePhase) {}
  int getNumberOfSenders() {return _numberOfSenders;}

 private:
  void initializePhase1Receive();

  int _rank;
  int _nSlicers;
  int _nSegmentForceDetectors;
  int _numberOfSenders;

  Tissue* _tissue;
  SegmentForceAnalyzer* _segmentForceAnalyzer;
	
  bool _writeToFile;

  // Receive Phase 0: ALLTOALL
  int *_segmentForcesPerSender; 		// recvbuf
  int _numberOfInts;                // recvcount
  MPI_Datatype _typeInt;            // recvtype
        
  // Receive Phase 1: ALLTOALLV
  double* _segmentForces; 			// recvbuf
  int* _segmentForceDispls;         // recvDispls
  int _segmentForceDataSize;
  int _previousSegmentForceDataSize;
  MPI_Datatype _typeSegmentForceData;    // recvtype

  // Receive Phase 2: ALLREDUCESUM
  double _E; 			// recvbuf

  SegmentDescriptor _segmentDescriptor;
};

#endif

