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

#include "InitPhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

InitPhase::InitPhase(const std::string& name, PhaseElement::machineType mType)
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
