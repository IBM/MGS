#include "CG_CurrentPulseGenerator.h"
#include "CurrentPulseGenerator.h"
#include "Lens.h"
#include "MaxComputeOrder.h"
#include "rndm.h"
#include <iostream>
#include <math.h>
#include <cmath>
#include <memory>
#include <typeinfo>

#define DEFAULT_IO_INTERVAL 0.1  //[ms]
#define decimal_places 5
#define fieldDelimiter "\t"

/**
 * periodic   =
 * poisson    =
 * dualexp    = f(t) = (1 - e^(-t/tauRise)) * e^(-t/tauDecay)
 *        tauRise, tauDecay = in [ms]
 *        NOTE: EPSP-like current injection
 *        Larkum - Zhu - Sakmann (2001) chose tauDecay=4*tauRise
 *                         tauRise = 2, 5, 10, or 50 [ms]
 *  whitenoise = with [mean, SD]
 **/
void CurrentPulseGenerator::initialize(RNG& rng)
{
#ifdef WAIT_FOR_REST
  if (delay < NOGATING_TIME)
  {
    std::cerr << "ERROR : delay is smaller than NOGATING_TIME" << std::endl;
    std::cerr << "Either disable WAIT_FOR_REST or make delay longer"
              << std::endl;
    assert(0);
  }
#endif
  if (not deltaT)
  {
    std::cerr << typeid(*this).name() << " needs time-step connected\n";
    assert(deltaT);
  }
  if (pattern == "periodic")
  {
    nextPulse = delay;
    fpt_update = &CurrentPulseGenerator::update_PeriodicProtocol;
  }
  else if (pattern == "poisson")
  {
    nextPulse = delay - log(drandom(rng)) * period;
    fpt_update = &CurrentPulseGenerator::update_PoissonProtocol;
  }
  else if (pattern == "dualexp")
  {
    nextPulse = delay;
    assert(riseTime > 0);
    assert(decayTime > 0);
    fpt_update = &CurrentPulseGenerator::update_DualExpProtocol;
  }
  else if (pattern == "periodic_train")
  {
    nextPulse = delay;
    time_start_train = delay;
    fpt_update = &CurrentPulseGenerator::update_PeriodicTrainProtocol;
    num_completed_trains = 0;
    num_completed_pulses_per_train = 0;
    if (num_trains == 0)
      num_trains = (int)std::ceil(last/period);
  }
  else if (pattern == "whitenoise")
  {
    //need 'mean' and 'sd' values (in unit: pA)
    //nextPulse = delay - log(drandom(rng)) * period;
    fpt_update = &CurrentPulseGenerator::update_WhiteNoiseProtocol;
  }
  else 
  {
    std::cerr << typeid(*this).name() << " do not support this \"" << pattern << "\" protocol\n";
    assert(0);
  }

  peakInc = peak;
  if (duration > period)
  {
    std::cerr << "The duration of the stimulus should not be greater than the "
                 "period\n";
    assert(duration < period);
  }

  if (write_to_file == 1)
  {
    std::ostringstream os;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << "Iinj :" << "\n";
    if (io_interval < 0.000001)
      io_interval = DEFAULT_IO_INTERVAL;
    time_write_data = 0.0 ; 
  }
}
void CurrentPulseGenerator::dataCollection(float currentTime)
{
    (*outFile) <<  currentTime;
    (*outFile) << std::fixed << fieldDelimiter << I;
    (*outFile) << "\n";
}

void CurrentPulseGenerator::update(RNG& rng)
{
  float currentTime = (getSimulation().getIteration() * (*deltaT));
  (*this.*fpt_update)(rng, currentTime);
  if (write_to_file and currentTime > time_write_data)
  {
    this->dataCollection(currentTime);
    time_write_data += io_interval;
  }
}

void CurrentPulseGenerator::finalize(RNG& rng)
{
  if (write_to_file == 1)
    outFile->close();
}
void CurrentPulseGenerator::update_PeriodicProtocol(RNG& rng, float currentTime)
{
  I = 0.0;
  if (currentTime >= (nextPulse + duration) && currentTime <= last)
  {  // no pulse
    I = 0.0;
    peakInc += inc;
    nextPulse += period;
  }
  else if (currentTime >= nextPulse && currentTime <= last)
  {  // having pulse
    I = peakInc;
  }
}
void CurrentPulseGenerator::update_PoissonProtocol(RNG& rng, float currentTime)
{
  I = 0.0;
  if (currentTime >= (nextPulse + duration) && currentTime <= last)
  {  // no pulse
    I = 0.0;
    peakInc += inc;
    nextPulse += log(drandom(rng)) * period;
  }
  else if (currentTime >= nextPulse && currentTime <= last)
  {  // having pulse
    I = peakInc;
  }
}
void CurrentPulseGenerator::update_DualExpProtocol(RNG& rng, float currentTime)
{
  I = 0.0;
  if (currentTime >= (nextPulse + duration) && currentTime <= last)
  {  // no pulse
    I = 0.0;
    peakInc += inc;
    nextPulse += period;
  }
  else if (currentTime >= nextPulse && currentTime <= last)
  {  // having pulse
    float dt = currentTime - nextPulse;
    I = peakInc * (1 - exp(-dt / riseTime)) * (exp(-dt / decayTime));
  }
}

void CurrentPulseGenerator::update_PeriodicTrainProtocol(RNG& rng, float currentTime)
{
  I = 0.0;
  if (currentTime > last or currentTime < time_start_train or 
      currentTime > num_trains * period)
  {
    //do nothing
  }
  else
  {
    if (currentTime >= (time_start_train + period) )
    {//reset new  train
      peakInc = peak;
      num_completed_trains += 1;
      num_completed_pulses_per_train = 0;
      time_start_train += period;
      nextPulse = time_start_train;
    }

    if (currentTime >= (nextPulse + duration) )
    {  // no pulse
      I = 0.0;
      peakInc += inc;
      nextPulse += intra_train_gap;
      num_completed_pulses_per_train += 1;
    }
    else if (currentTime >= nextPulse && currentTime <= nextPulse + duration &&
        num_completed_pulses_per_train < num_pulses_per_train)
    {  // having pulse
      I = peakInc;
    }
  }
  //NOTE: 
  //___[ ]_____[ ]______[ ]___________[  ]
  //  delay
  //   dura
  //    intertrian
}

void CurrentPulseGenerator::update_WhiteNoiseProtocol(RNG& rng, float currentTime)
{
  I = 0.0;
  if (currentTime >= (nextPulse + duration) && currentTime <= last)
  {  // no pulse
    I = 0.0;
    peakInc += inc;
    nextPulse += log(drandom(rng)) * period;
  }
  else if (currentTime >= nextPulse && currentTime <= last)
  {  // having pulse
    I = gaussian(mean, sd, rng);
  }
}
CurrentPulseGenerator::CurrentPulseGenerator() : CG_CurrentPulseGenerator(), outFile(0) {}

CurrentPulseGenerator::~CurrentPulseGenerator() { 
  if (write_to_file) 
    delete outFile; 
}

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
