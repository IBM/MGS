// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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

