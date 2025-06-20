// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "SimulationInfoNew.h"
#include "CG_SimulationInfoNew.h"
#include <memory>

void SimulationInfoNew::initialize(RNG& rng) 
{
}

void SimulationInfoNew::calculateInfo(RNG& rng) 
{
}

SimulationInfoNew::SimulationInfoNew() 
   : CG_SimulationInfoNew(){
}

SimulationInfoNew::~SimulationInfoNew() 
{
}

void SimulationInfoNew::duplicate(std::unique_ptr<SimulationInfoNew>&& dup) const
{
   dup.reset(new SimulationInfoNew(*this));
}

void SimulationInfoNew::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new SimulationInfoNew(*this));
}

void SimulationInfoNew::duplicate(std::unique_ptr<CG_SimulationInfoNew>&& dup) const
{
   dup.reset(new SimulationInfoNew(*this));
}

