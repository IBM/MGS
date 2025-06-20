// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Communicator.h"
#include "Receiver.h"
#include "Sender.h"
#include "BG_AvailableMemory.h"
#include <mpi.h>
#include <cassert>
#include <vector>

#define MAX_N_SEND_BUFFS 2

Communicator::Communicator() 
{
}

Communicator::~Communicator()
{
}

void Communicator::allToAll(Sender* s, Receiver* r, int cycle, int phase)
{
  MPI_Alltoall(s->getSendbuf(cycle, phase), *s->getSendcounts(cycle, phase), *s->getSendtypes(cycle, phase), 
               r->getRecvbuf(cycle, phase), *r->getRecvcounts(cycle, phase), *r->getRecvtypes(cycle, phase), MPI_COMM_WORLD);

}

void Communicator::allToAllV(Sender* s, Receiver* r, int cycle, int phase)
{
  MPI_Alltoallv(s->getSendbuf(cycle, phase), s->getSendcounts(cycle, phase),  s->getSdispls(cycle, phase), *s->getSendtypes(cycle, phase), 
		r->getRecvbuf(cycle, phase), r->getRecvcounts(cycle, phase),  r->getRdispls(cycle, phase), *r->getRecvtypes(cycle, phase), MPI_COMM_WORLD);
}


void Communicator::allToAllW(Sender* s, Receiver* r, int cycle, int phase)
{
   MPI_Alltoallw(s->getSendbuf(cycle, phase), s->getSendcounts(cycle, phase), s->getSdispls(cycle, phase), s->getSendtypes(cycle, phase), 
                 r->getRecvbuf(cycle, phase), r->getRecvcounts(cycle, phase), r->getRdispls(cycle, phase), r->getRecvtypes(cycle, phase), MPI_COMM_WORLD);

}

void Communicator::allGather(Sender* s, Receiver* r, int cycle, int phase)
{
  MPI_Allgather(s->getSendbuf(cycle, phase), *s->getSendcounts(cycle, phase), *s->getSendtypes(cycle, phase), 
                r->getRecvbuf(cycle, phase), *r->getRecvcounts(cycle, phase),  *r->getRecvtypes(cycle, phase), MPI_COMM_WORLD);
}

void Communicator::allGatherV(Sender* s, Receiver* r, int cycle, int phase)
{
  MPI_Allgatherv(s->getSendbuf(cycle, phase), *s->getSendcounts(cycle, phase), *s->getSendtypes(cycle, phase), 
                 r->getRecvbuf(cycle, phase), r->getRecvcounts(cycle, phase),  r->getRdispls(cycle, phase), *r->getRecvtypes(cycle, phase), MPI_COMM_WORLD);
}

void Communicator::allReduceSum(Sender* s, Receiver* r, int cycle, int phase)
{
  MPI_Allreduce(s->getSendbuf(cycle, phase), r->getRecvbuf(cycle, phase), 
		*s->getSendcounts(cycle, phase), *s->getSendtypes(cycle, phase), 
		MPI_SUM, MPI_COMM_WORLD);
}

void Communicator::bcast(Sender* s, Receiver* r, int cycle, int phase)
{
  if (s->getRank()==0) MPI_Bcast(s->getSendbuf(cycle, phase), *s->getSendcounts(cycle, phase), *s->getSendtypes(cycle, phase), 0, MPI_COMM_WORLD);
  else MPI_Bcast(r->getRecvbuf(cycle, phase), *r->getRecvcounts(cycle, phase), *r->getRecvtypes(cycle, phase), 0, MPI_COMM_WORLD);
}

void Communicator::tableMerge(Sender* s, Receiver* r, int cycle, int phase) 
{
  int rank=s->getRank();
  int size=0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int recvRank[2] = {rank*2+1, rank*2+2};
  if (recvRank[0]>=size) recvRank[0]=-1;
  if (recvRank[1]>=size) recvRank[1]=-1;

  int sendRank=-1;
  if (rank>0) sendRank=int(floor(double(rank-1)/2.0));

  if (recvRank[0]!=-1) {
    void* recvbuf[2] = {0, 0};
    MPI_Datatype recvtype = *(r->getRecvtypes(cycle, phase));    
    MPI_Request request[2];
    recvbuf[0] = r->getRecvbuf(cycle, phase);
    MPI_Irecv(recvbuf[0], MERGE_BUFF_SIZE, recvtype, recvRank[0], MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
    if (recvRank[1]!=-1) {
      recvbuf[1] = r->getRecvbuf(cycle, phase);
      MPI_Irecv(recvbuf[1], MERGE_BUFF_SIZE, recvtype, recvRank[1], MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
    }
    bool flip=false;
    int recvcount=0;
    int tag=0;
    int more[2] = {1, 1};
    if (recvRank[1]==-1) more[1]=0;
    int i=0;
    while (more[0] || more[1]) {
      MPI_Status status;
      for (int flag=0; flag==0; flip=!flip) {
        i = flip ? 1 : 0;
        flag=request[i].Test(status);
        if (more[i]==0) i = flip ? 0 : 1;
        if (flag) more[i] = status.Get_tag();	  
      }
      MPI_Get_count(&status, recvtype, &recvcount);
      s->mergeWithSendBuf(i, recvcount, cycle, phase);
      if (more[i]) MPI_Irecv(recvbuf[i], MERGE_BUFF_SIZE, recvtype, recvRank[i], MPI_ANY_TAG, MPI_COMM_WORLD, &request[i]);
    }
  }

  if (sendRank!=-1) {
    MPI_Datatype sendtype = *(s->getSendtypes(cycle, phase));
    int sendcount = *(s->getSendcounts(cycle, phase));
    int sendbufsize = MERGE_BUFF_SIZE;
    int tag = 1;
    std::vector<void*> sendbufs;
    sendbufs.push_back(s->getSendbuf(cycle, 0));
    if (sendcount<=sendbufsize) {
      sendbufsize = sendcount;
      tag = 0;
    }
    std::vector<MPI_Request> requestVector;
    MPI_Request request;
    MPI_Isend(sendbufs[0], sendbufsize, sendtype, sendRank, tag, MPI_COMM_WORLD, &request);
    requestVector.push_back(request);
    while (tag) {
      bool reused=false;
      for (int i=0; i<requestVector.size(); ++i) {
        int flag=0;
	      MPI_Status status;
        MPI_Test(&requestVector[i], &flag, &status);
        if (flag) {
	        reused=true;
	        s->getSendbuf(cycle, i);
	        sendcount-=sendbufsize;
	        if (sendcount<=sendbufsize) {
	          sendbufsize = sendcount;
	          tag=0;
	        }
	        MPI_Isend(sendbufs[i], sendbufsize, sendtype, sendRank, tag, MPI_COMM_WORLD, &requestVector[i]);
	      }
      }
      if (!reused && requestVector.size()<MAX_N_SEND_BUFFS) {
        int idx = requestVector.size();
        sendbufs.push_back(s->getSendbuf(cycle, idx));
        sendcount-=sendbufsize;
        if (sendcount<=sendbufsize) {
          sendbufsize = sendcount;
          tag=0;
        }
        MPI_Request request;
        MPI_Isend(sendbufs[idx], sendbufsize, sendtype, sendRank, tag, MPI_COMM_WORLD, &request);
        requestVector.push_back(request);
      }
    }
  }
}
