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
// =================================================================

#ifndef MpiAsynchSender_H
#define MpiAsynchSender_H
#include "Copyright.h"
#include <mpi.h>

#include "AsynchSender.h"

class Simulation;

class MpiAsynchSender : public AsynchSender
{
   public:
      MpiAsynchSender(char* buffer, int bufferSize, int peer, int tag, Simulation* sim);
 
      // Checks if the transaction is complete.
      virtual bool check();
#ifdef VERBOSE
      static double sendElapsed;
#endif
   protected:
      virtual void internalSendRequest(int size);
      virtual void internalComplete();

      int _tag;
      MPI_Request _request;
      Simulation* _sim;
};

#endif
