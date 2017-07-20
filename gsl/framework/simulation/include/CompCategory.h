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
      virtual int initPartitions(int num) = 0;
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
