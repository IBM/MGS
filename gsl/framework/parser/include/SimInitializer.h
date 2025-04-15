// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      bool runSimulationAndUI(CommandLine& commandLine, std::unique_ptr<Simulation>& sim);
      int _rank;
      int _size;
      static RunTimeTopology _topology;
      std::string preprocessMGSROOT(const std::string& inputFile);
};
#endif // SimInitializer
