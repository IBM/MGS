// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMPCATEGORY_H
#define COMPCATEGORY_H
#include <map>
#include "Copyright.h"

#include "rndm.h"

class WorkUnit;
class Simulation;
class NodeDescriptor;
class ConnectionIncrement;

class CompCategory
{

   public:
      // Used to push workunits to simulation.
      virtual void getWorkUnits() = 0;

      // Initializes partitions, pushes work units to the simulation.
      virtual void initPartitions(int numCpuWorkUnits, int numGpuWorkUnits) = 0;
#if 0
#ifdef HAVE_MPI
      virtual void resetSendProcessIdIterators() = 0;
      virtual int getSendNextProcessId() = 0;
      virtual bool atSendProcessIdEnd() = 0;
      virtual void resetReceiveProcessIdIterators() = 0;
      virtual int getReceiveNextProcessId() = 0;
      virtual bool atReceiveProcessIdEnd() = 0;
      virtual void send(int) = 0;
      virtual void receiveReset(int pid) = 0;
      virtual int receive(int pid, const char* buffer, int count) = 0; //returns offset of last byte used
      virtual bool receiveDone(int pid) = 0;
#endif
#endif
//      virtual std::map<std::string, ConnectionIncrement>* getComputeCost() = 0;
//      virtual ConnectionIncrement* getComputeCost() = 0;        commented out by Jizhu Lu on 02/20/2006
      virtual ~CompCategory() {}
      virtual Simulation& getSimulation() = 0;
      bool _doneflag;
};

#endif
