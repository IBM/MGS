// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
// 
// =================================================================

//#include <mpi.h>
#include "MpiAsynchSender.h"
#include "Simulation.h"

MpiAsynchSender::MpiAsynchSender(char* buffer, int bufferSize, 
				 int peer, int tag, Simulation* sim)
  : AsynchSender(buffer, bufferSize, peer), _tag(tag), _request(0), _sim(sim)
{}
 
bool MpiAsynchSender::check() {
   int flag;
   MPI_Iprobe(_peer, _tag, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
   return flag == 1;
}

void MpiAsynchSender::internalSendRequest(int size) {
#ifdef VERBOSE
  double now, then;
  /* MPI_W begin: measure MPI send communiation */
  now=MPI_Wtime();
#endif      
  MPI_Isend(_buffer, size, MPI_CHAR, _peer, _tag, MPI_COMM_WORLD, &_request);
#ifdef VERBOSE
  then=MPI_Wtime();
  sendElapsed+=(then-now);
  /* MPI_W begin: measure MPI send communiation */
#endif
}

void MpiAsynchSender::internalComplete() {
   MPI_Wait(&_request, MPI_STATUS_IGNORE);
}
