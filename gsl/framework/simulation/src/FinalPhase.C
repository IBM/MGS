// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FinalPhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

FinalPhase::FinalPhase(const std::string& name, machineType mType)
  : Phase(name, mType)
{
}

FinalPhase::~FinalPhase()
{
}

std::string FinalPhase::getType() const
{
   return "Final";
}

void FinalPhase::duplicate(std::unique_ptr<Phase>& rv) const
{
   rv.reset(new FinalPhase(*this));
}

void FinalPhase::addToSimulation(Simulation* sim) const
{
   sim->addFinalPhase(_name, _machineType);
}
