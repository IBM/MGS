// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
//#include <mpi.h>
#include "MpiAsynchReceiver.h"
#include <cassert>

MpiAsynchReceiver::MpiAsynchReceiver(char* buffer, int bufferSize, 
				     int peer, int tag)
   : AsynchReceiver(buffer, bufferSize, peer), _tag(tag), _request(0)
{
  assert(0);
}
 
bool MpiAsynchReceiver::check() 
{
   int flag;
   MPI_Iprobe(_peer, _tag, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
   return flag == 1;
}

void MpiAsynchReceiver::internalReceiveRequest() 
{
   MPI_Irecv(_buffer, _bufferSize, MPI_CHAR, _peer, _tag,
	     MPI_COMM_WORLD, &_request);
}

int MpiAsynchReceiver::internalComplete() 
{
   MPI_Status status;
   int count;
   MPI_Wait(&_request, &status);
   MPI_Get_count(&status, MPI_CHAR, &count);
   return count;
}
