// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005, 2006  All rights reserved
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

void FinalPhase::duplicate(std::auto_ptr<Phase>& rv) const
{
   rv.reset(new FinalPhase(*this));
}

void FinalPhase::addToSimulation(Simulation* sim) const
{
   sim->addFinalPhase(_name);
}
