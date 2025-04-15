// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TOUCHAGGREGATOR_H
#define TOUCHAGGREGATOR_H
#define TOUCHAGGREGATOR_RECEIVE_PHASES 2

#include <mpi.h>
#include "Receiver.h"
#include "TouchVector.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <list>


class Touch;
class TissueContext;

class TouchAggregator : public Receiver 

{
 public:
  TouchAggregator(int rank, int nSenders, TissueContext* tissueContext);
  ~TouchAggregator();

  Touch* getTouches() {return _touches;}
  int getNumberOfTouches() {return _touchDataSize;}
			
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
  void reinstateTouches();

 private:
  void initializePhase1Receive();

  int _rank;
  int _numberOfSenders;

  bool _writeToFile;

  // Receive Phase 0: ALLTOALL
  int *_touchesPerSender; 							// recvbuf
  int _numberOfInts;                     // recvcount
  MPI_Datatype _typeInt;                 // recvtype
        
  // Receive Phase 1: ALLTOALLW
  Touch*_touches;
  Touch* _recvbuf;
  int* _touchDispls;         // recvDispls
  MPI_Datatype* _typeTouches;  // recvtype
  int _touchDataSize;
  int _previousTouchDataSize;

  TissueContext* _tissueContext;
};

#endif

