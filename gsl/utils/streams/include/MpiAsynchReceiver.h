// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
