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

#include "FinalPhase.h"
#include <memory>
#include <string>
#include "Simulation.h"

FinalPhase::FinalPhase(const std::string& name)
   : Phase(name)
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
   sim->addFinalPhase(_name);
}
