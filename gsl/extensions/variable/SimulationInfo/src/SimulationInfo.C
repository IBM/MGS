#include "Lens.h"
#include "SimulationInfo.h"
#include "CG_SimulationInfo.h"
#include <memory>

void SimulationInfo::initialize(RNG& rng) 
{
	assert(deltaT);
	assert(recordIntervalInTime > 0 || recordIntervalInIterations > 0);
	if (recordIntervalInTime > 0)
		recordIntervalInIterations = (int)(recordIntervalInTime / (*deltaT));
	if (recordIntervalInIterations == 0)
	{
		std::cerr << "Record interval cannot be zero. Make sure recordIntervalInTimes > deltaT\n";
		assert(0);
	}
}

void SimulationInfo::calculateInfo(RNG& rng) 
{
	currentTime = (*deltaT) * getSimulation().getIteration();
}

SimulationInfo::SimulationInfo() 
   : CG_SimulationInfo()
{
}

SimulationInfo::~SimulationInfo() 
{
}

void SimulationInfo::duplicate(std::auto_ptr<SimulationInfo>& dup) const
{
   dup.reset(new SimulationInfo(*this));
}

void SimulationInfo::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new SimulationInfo(*this));
}

void SimulationInfo::duplicate(std::auto_ptr<CG_SimulationInfo>& dup) const
{
   dup.reset(new SimulationInfo(*this));
}

