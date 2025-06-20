// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "InitPhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

InitPhase::InitPhase(const std::string& name, machineType mType)
   : Phase(name, mType)
{
}

InitPhase::~InitPhase()
{
}

std::string InitPhase::getType() const
{
   return "Init";
}

void InitPhase::duplicate(std::unique_ptr<Phase>& rv) const
{
   rv.reset(new InitPhase(*this));
}

void InitPhase::addToSimulation(Simulation* sim) const
{
   sim->addInitPhase(_name, _machineType);
}
