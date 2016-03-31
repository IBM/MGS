#include "Lens.h"
#include "CurrentPulseGenerator.h"
#include "CG_CurrentPulseGenerator.h"
#include "rndm.h"
#include <memory>
#include <math.h>
#include <iostream>

void CurrentPulseGenerator::initialize(RNG& rng)
{
  assert(deltaT);
  if (pattern == "periodic")
    nextPulse = delay;
  else if (pattern == "poisson")
    nextPulse = delay - log(drandom(rng)) * period;
  peakInc = peak;
  if (duration > period)
  {
    std::cerr << "The duration of the stimulus should not be greater than the "
                 "period\n";
    assert(duration < period);
  }
}

void CurrentPulseGenerator::update(RNG& rng)
{
  I = 0.0;
  float currentTime = (getSimulation().getIteration() * (*deltaT));
  if (currentTime >= (nextPulse + duration) && currentTime <= last)
  {
    I = 0.0;
    peakInc += inc;
    if (pattern == "periodic")
      nextPulse += period;
    else if (pattern == "poisson")
      nextPulse -= log(drandom(rng)) * period;
  }
  else if (currentTime >= nextPulse && currentTime <= last)
  {
    I = peakInc;
  }
}

CurrentPulseGenerator::CurrentPulseGenerator() : CG_CurrentPulseGenerator() {}

CurrentPulseGenerator::~CurrentPulseGenerator() {}

void CurrentPulseGenerator::duplicate(
    std::auto_ptr<CurrentPulseGenerator>& dup) const
{
  dup.reset(new CurrentPulseGenerator(*this));
}

void CurrentPulseGenerator::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new CurrentPulseGenerator(*this));
}

void CurrentPulseGenerator::duplicate(
    std::auto_ptr<CG_CurrentPulseGenerator>& dup) const
{
  dup.reset(new CurrentPulseGenerator(*this));
}
