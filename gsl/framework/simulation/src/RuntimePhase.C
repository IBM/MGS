// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "RuntimePhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

RuntimePhase::RuntimePhase(const std::string& name, machineType mType)
  : Phase(name, mType)
{
}

RuntimePhase::~RuntimePhase()
{
}

std::string RuntimePhase::getType() const
{
   return "Runtime";
}

void RuntimePhase::duplicate(std::unique_ptr<Phase>& rv) const
{
   rv.reset(new RuntimePhase(*this));
}

void RuntimePhase::addToSimulation(Simulation* sim) const
{
   sim->addRuntimePhase(_name, _machineType);
}
