// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

void SimulationSetter::duplicate(std::unique_ptr<SimulationSetter>&& dup) const
{
   dup.reset(new SimulationSetter(*this));
}

void SimulationSetter::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new SimulationSetter(*this));
}

void SimulationSetter::duplicate(std::unique_ptr<CG_SimulationSetter>&& dup) const
{
   dup.reset(new SimulationSetter(*this));
}

