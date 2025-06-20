// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "LoadPhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

LoadPhase::LoadPhase(const std::string& name, machineType mType)
   : Phase(name, mType)
{
}

LoadPhase::~LoadPhase()
{
}

std::string LoadPhase::getType() const
{
   return "Load";
}

void LoadPhase::duplicate(std::unique_ptr<Phase>& rv) const
{
   rv.reset(new LoadPhase(*this));
}

void LoadPhase::addToSimulation(Simulation* sim) const
{
  sim->addLoadPhase(_name, _machineType);
}
