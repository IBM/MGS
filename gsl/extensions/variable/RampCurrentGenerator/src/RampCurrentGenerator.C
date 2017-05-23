#include "Lens.h"
#include "RampCurrentGenerator.h"
#include "CG_RampCurrentGenerator.h"
#include "rndm.h"
#include <memory>
#include <iostream>
#include <math.h>
#include <cmath>
#include <memory>
#include <typeinfo>
#define DEFAULT_IO_INTERVAL 0.1  //[ms]
#define decimal_places 5
#define fieldDelimiter "\t"

void RampCurrentGenerator::initialize(RNG& rng) 
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
  assert(tstart < tend);
  first_enter_pulse = true;
  if (pattern == "ramp")
  {
    tstart = delay;
    tend = delay + duration;
    nextPulse = delay;
    fpt_update = &RampCurrentGenerator::update_RampProtocol;
  }
  else{
    std::cerr << "Unsupported pattern: Use 'ramp'\n";
    assert(0);
  }
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

void RampCurrentGenerator::update(RNG& rng) 
{
  I = 0.0;
  float currentTime = (getSimulation().getIteration() * (*deltaT));
  (*this.*fpt_update)(rng, currentTime);
  if (write_to_file and currentTime > time_write_data)
  {
    this->dataCollection(currentTime);
    time_write_data += io_interval;
  }
}

void RampCurrentGenerator::update_RampProtocol(RNG& rng, float currentTime)
{
  if (currentTime >= (nextPulse + duration) && currentTime <= last)
  {//no pulse
    I = 0.0;
    first_enter_pulse = true;
    //peakInc += inc;
    if (pattern == "ramp")
      nextPulse += period;

  }
  else if (currentTime >= nextPulse && currentTime <= last)
  {//having pulse
    //float dt = currentTime - nextPulse;
    if (first_enter_pulse)
    {
      tstart = nextPulse;
      tend = tstart+duration;
      first_enter_pulse = false;
    }
    float time_ofset = currentTime - tstart;
    if ((pattern == "ramp"))
    {
      I = Istart + Iend * (time_ofset/(tend-tstart));
    }
  }
}

void RampCurrentGenerator::finalize(RNG& rng)
{
  if (write_to_file == 1)
    outFile->close();
}

RampCurrentGenerator::RampCurrentGenerator() 
   : CG_RampCurrentGenerator(), outFile(0)
{
}

RampCurrentGenerator::~RampCurrentGenerator() 
{
  if (write_to_file) 
    delete outFile; 
}

void RampCurrentGenerator::duplicate(std::auto_ptr<RampCurrentGenerator>& dup) const
{
   dup.reset(new RampCurrentGenerator(*this));
}

void RampCurrentGenerator::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new RampCurrentGenerator(*this));
}

void RampCurrentGenerator::duplicate(std::auto_ptr<CG_RampCurrentGenerator>& dup) const
{
   dup.reset(new RampCurrentGenerator(*this));
}

