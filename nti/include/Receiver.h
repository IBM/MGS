// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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

