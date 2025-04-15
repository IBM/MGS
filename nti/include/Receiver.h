// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RECEIVER_H
#define RECEIVER_H

#include <mpi.h>

#include "Communicator.h"

class Receiver
{
   public:
     virtual ~Receiver() {}
     virtual int getRank() =0;
     virtual void prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef) =0;
     virtual void* getRecvbuf(int receiveCycle, int receivePhase) =0;
     virtual int* getRecvcounts(int receiveCycle, int receivePhase) =0;
     virtual int* getRdispls(int receiveCycle, int receivePhase) =0;
     virtual MPI_Datatype* getRecvtypes(int receiveCycle, int receivePhase) =0;
     virtual int getNumberOfReceiveCycles() =0;
     virtual int getNumberOfReceivePhasesPerCycle(int cycleNumber) =0;
     virtual int getNumberOfSenders() =0;
     virtual void finalizeReceive(int receiveCycle, int receivePhase) =0;
};

#endif

