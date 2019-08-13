#include "Lens.h"
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

void SimulationInfoNew::duplicate(std::unique_ptr<SimulationInfoNew>& dup) const
{
   dup.reset(new SimulationInfoNew(*this));
}

void SimulationInfoNew::duplicate(std::unique_ptr<Variable>& dup) const
{
   dup.reset(new SimulationInfoNew(*this));
}

void SimulationInfoNew::duplicate(std::unique_ptr<CG_SimulationInfoNew>& dup) const
{
   dup.reset(new SimulationInfoNew(*this));
}

