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

#ifndef _SimInitializer_H
#define _SimInitializer_H
#include "Copyright.h"
#include "RunTimeTopology.h"

#include <memory>
#include <string>

class CommandLine;
class Simulation;
class GraphicalUserInterface;
class TextUserInterface;

class SimInitializer
{
   public:
      SimInitializer();
      bool execute(int* argc, char*** argv);
   private:
      bool internalExecute(int argc, char** argv);
      bool runSimulationAndUI(
	 CommandLine& commandLine, std::unique_ptr<Simulation>& sim);
      int _rank;
      int _size;
      static RunTimeTopology _topology;
};
#endif // SimInitializer
