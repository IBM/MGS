// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "SimulationSetter.h"
#include "CG_SimulationSetter.h"
#include <memory>

void SimulationSetter::initialize(RNG& rng) 
{
}

void SimulationSetter::propagateToggles(RNG& rng) 
{
}

void SimulationSetter::switchPlasticityOnOff(Trigger* trigger, NDPairList* ndPairList) 
{
}

SimulationSetter::SimulationSetter() 
   : CG_SimulationSetter()
{
}

SimulationSetter::~SimulationSetter() 
{
}

void SimulationSetter::duplicate(std::auto_ptr<SimulationSetter>& dup) const
{
   dup.reset(new SimulationSetter(*this));
}

void SimulationSetter::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new SimulationSetter(*this));
}

void SimulationSetter::duplicate(std::auto_ptr<CG_SimulationSetter>& dup) const
{
   dup.reset(new SimulationSetter(*this));
}

