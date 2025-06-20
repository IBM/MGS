// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
