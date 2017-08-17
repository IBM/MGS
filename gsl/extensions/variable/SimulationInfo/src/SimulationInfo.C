#include "Lens.h"
#include "SimulationInfo.h"
#include "CG_SimulationInfo.h"
#include <memory>

void SimulationInfo::initialize(RNG& rng) 
{
   if (not deltaT)
   {
      std::cerr << "ERROR: Please connect deltaT to " << typeid(*this).name() << std::endl;
   }
   assert(deltaT);
   if (recordIntervalInTime <= 0 and recordIntervalInIterations <= 0)
   {
      std::cerr << "ERROR: Please define recordIntervalInTime > 0  or recordIntervalInIterations > 0 for " << typeid(*this).name() << std::endl;
   }
   //at least one must be used
   assert(recordIntervalInTime > 0 || recordIntervalInIterations > 0);

   if (recordIntervalInTime > 0)
      recordIntervalInIterations = (int)(recordIntervalInTime / (*deltaT));
   else if (recordIntervalInIterations > 0)
      recordIntervalInTime = (float)(recordIntervalInIterations * (*deltaT));

   if (recordIntervalInIterations == 0)
   {
      std::cerr << "Record interval cannot be zero. Make sure recordIntervalInTimes > deltaT\n";
      assert(0);
   }
   iterationCount = 0;
}

void SimulationInfo::calculateInfo(RNG& rng) 
{
   //currentTime = (*deltaT) * getSimulation().getIteration();
   currentTime += (*deltaT); //by doing this - we can configure a negative time and use this to 
                             // set-up a way to run-until-steady-state before any data I/O
                             // is performed which use 0-based time as criteria
                             // NOTE: Not all system has steady-states (e.g. those with oscillation)
                             // so this is not always useful
   // * getSimulation().getIteration();
   iterationCount  = getSimulation().getIteration() - 1; //zero-based for RuntimePhase
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

