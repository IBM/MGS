// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MpiAsynchReceiver_H
#define MpiAsynchReceiver_H
#include "Copyright.h"
#include <mpi.h>

#include "AsynchReceiver.h"

class MpiAsynchReceiver : public AsynchReceiver
{
   public:
      MpiAsynchReceiver(char* buffer, int bufferSize, int peer, int tag);
 
      // Checks if the transaction is complete.
      virtual bool check();

   protected:
      virtual void internalReceiveRequest();
      virtual int internalComplete();
 
      int _tag;
      MPI_Request _request;
};

#endif
