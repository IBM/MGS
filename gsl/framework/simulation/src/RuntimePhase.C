// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "RuntimePhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

RuntimePhase::RuntimePhase(const std::string& name)
   : Phase(name)
{
}

RuntimePhase::~RuntimePhase()
{
}

std::string RuntimePhase::getType() const
{
   return "Runtime";
}

void RuntimePhase::duplicate(std::auto_ptr<Phase>& rv) const
{
   rv.reset(new RuntimePhase(*this));
}

void RuntimePhase::addToSimulation(Simulation* sim) const
{
   sim->addRuntimePhase(_name);
}
