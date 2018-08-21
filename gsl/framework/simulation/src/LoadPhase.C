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

#include "LoadPhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

LoadPhase::LoadPhase(const std::string& name)
   : Phase(name)
{
}

LoadPhase::~LoadPhase()
{
}

std::string LoadPhase::getType() const
{
   return "Load";
}

void LoadPhase::duplicate(std::auto_ptr<Phase>& rv) const
{
   rv.reset(new LoadPhase(*this));
}

void LoadPhase::addToSimulation(Simulation* sim) const
{
   sim->addLoadPhase(_name);
}
