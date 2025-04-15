// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SENDER_H
#define SENDER_H

#define MERGE_BUFF_SIZE 100000

#include <mpi.h>

#include "TableEntry.h"
#include "Communicator.h"

class Sender
{
   public:
     virtual ~Sender() {}
     virtual int getRank() =0;
     virtual void prepareToSend(int sendCycle, int sendPhase, CommunicatorFunction& funPtrRef) =0;
     virtual void* getSendbuf(int sendCycle, int sendPhase) =0;
     virtual int* getSendcounts(int sendCycle, int sendPhase) =0;
     virtual int* getSdispls(int sendCycle, int sendPhase) =0;
     virtual MPI_Datatype* getSendtypes(int sendCycle, int sendPhase) =0;
     virtual int getNumberOfSendCycles() =0;
     virtual int getNumberOfSendPhasesPerCycle(int sendCycle) =0;
     virtual int getNumberOfReceivers() =0;
     virtual void mergeWithSendBuf(int index, int count, int sendCycle, int sendPhase) =0;

};

#endif

