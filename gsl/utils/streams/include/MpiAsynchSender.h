// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
