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

#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <mpi.h>

#include <iostream>
#include <math.h>

class Sender;
class Receiver;

class Communicator
{
   public:
     Communicator();      
     ~Communicator();
     
     void allToAll(Sender* s, Receiver* r, int cycle, int phase);   
     void allToAllV(Sender* s, Receiver* r, int cycle, int phase);   
     void allToAllW(Sender* s, Receiver* r, int cycle, int phase);   
     void allGather(Sender* s, Receiver* r, int cycle, int phase);   
     void allGatherV(Sender* s, Receiver* r, int cycle, int phase);   
     void allReduceSum(Sender* s, Receiver* r, int cycle, int phase);
     void bcast(Sender* s, Receiver* r, int cycle, int phase);
     void tableMerge(Sender* s, Receiver* r, int cycle, int phase);
};

typedef void(Communicator::*CommunicatorFunction)(Sender*, Receiver*, int, int);

#endif
