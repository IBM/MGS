#include "Lens.h"
#include "CurrentPulseGenerator.h"
#include "CG_CurrentPulseGenerator.h"
#include "rndm.h"
#include <memory>
#include <math.h>
#include <iostream>
#include "MaxComputeOrder.h"

/**
 * periodic   = 
 * poisson    =
 * dualexp    = f(t) = (1 - e^(-t/tauRise)) * e^(-t/tauDecay)
 *        tauRise, tauDecay = in [ms]
 *        NOTE: EPSP-like current injection
 *        Larkum - Zhu - Sakmann (2001) chose tauDecay=4*tauRise
 *                         tauRise = 2, 5, 10, or 50 [ms]
 **/ 
void CurrentPulseGenerator::initialize(RNG& rng)
{
#ifdef WAIT_FOR_REST
  if (delay < NOGATING_TIME)
  {
    std::cerr << "ERROR : delay is smaller than NOGATING_TIME" << std::endl;
    std::cerr << "Either disable WAIT_FOR_REST or make delay longer" << std::endl;
    assert(0);
  }
#endif
  assert(deltaT);
  if (pattern == "periodic")
    nextPulse = delay;
  else if (pattern == "poisson")
    nextPulse = delay - log(drandom(rng)) * period;
	else if (pattern == "dualexp")
	{
    nextPulse = delay;
		assert(riseTime > 0 );
		assert(decayTime > 0);
	}
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
  {//no pulse
    I = 0.0;
    peakInc += inc;
    if (pattern == "periodic")
      nextPulse += period;
    else if (pattern == "poisson")
      nextPulse -= log(drandom(rng)) * period;
		else if (pattern == "dualexp")
      nextPulse += period;
  }
  else if (currentTime >= nextPulse && currentTime <= last)
  {//having pulse
    float dt = currentTime - nextPulse;
    if ((pattern == "periodic") || (pattern == "poisson"))
      I = peakInc;
    else if (pattern == "dualexp")
      I = peakInc * (1- exp(-dt/riseTime)) * (exp(-dt/decayTime));
  }
}

CurrentPulseGenerator::CurrentPulseGenerator() : CG_CurrentPulseGenerator() {
}

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
