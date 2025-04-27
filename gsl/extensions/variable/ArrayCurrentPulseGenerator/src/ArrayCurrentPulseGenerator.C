#include "Lens.h"
#include "ArrayCurrentPulseGenerator.h"
#include "CG_ArrayCurrentPulseGenerator.h"
#include "rndm.h"
#include <iostream>
#include <math.h>
#include <cmath>
#include <memory>
#include <typeinfo>

#define DEBUG

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
 *  ramp      = gradually increase until rheobase is detected
 **/
void ArrayCurrentPulseGenerator::initialize(RNG& rng) 
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

  if (I.size() != num_streams) I.increaseSizeTo(num_streams);
  if (streams_duration.size() != num_streams) streams_duration.resize(num_streams);
  if (streams_delay.size() != num_streams) streams_delay.resize(num_streams);
  if (streams_peakInc.size() != num_streams) streams_peakInc.resize(num_streams);
  if (nextPulses.size() != num_streams) nextPulses.resize(num_streams);
  for (int _i = 0; _i < num_streams; _i++)
  {
    streams_duration[_i] = duration;
    streams_delay[_i] = delay;
    streams_peakInc[_i] = peak;

  }
  rerand_streams_info(rng);
  if (pattern == "periodic")
  {
    nextPulses = streams_delay;
    fpt_update = &ArrayCurrentPulseGenerator::update_PeriodicProtocol;
  }
  else if (pattern == "poisson")
  {
    for (int _i = 0; _i < num_streams; _i++)
      nextPulses[_i] = streams_delay[_i] - log(drandom(rng)) * period;
    fpt_update = &ArrayCurrentPulseGenerator::update_PoissonProtocol;
  }
  else if (pattern == "dualexp")
  {
    nextPulses = streams_delay;
    assert(riseTime > 0);
    assert(decayTime > 0);
    fpt_update = &ArrayCurrentPulseGenerator::update_DualExpProtocol;
  }
  else if (pattern == "periodic_train")
  {
    nextPulses = streams_delay;
    time_start_train = delay;
    fpt_update = &ArrayCurrentPulseGenerator::update_PeriodicTrainProtocol;
    num_completed_trains = 0;
    num_completed_pulses_per_train = 0;
    if (num_trains == 0)
      num_trains = (int)std::ceil(last/period);
  }
  else if (pattern == "whitenoise")
  {
    //need 'mean' and 'sd' values (in unit: pA)
    //nextPulse = delay - log(drandom(rng)) * period;
    fpt_update = &ArrayCurrentPulseGenerator::update_WhiteNoiseProtocol;
  }
  else if (pattern == "ramp")
  {
    nextPulses = streams_delay;
    fpt_update = &ArrayCurrentPulseGenerator::update_RampProtocol;
    last = delay + duration; // only 1 repeat is allowed here
  }
  else 
  {
    std::cerr << typeid(*this).name() << " do not support this \"" << pattern << "\" protocol\n";
    std::cerr << "Use either [periodic_train, periodic, poisson, dualexp, ramp]" << std::endl;
    assert(0);
  }

  peakInc = peak;
  for (int _i = 0; _i < num_streams; _i++)
    nextPulses[_i] += init_duration;  // the time during which assumed for system settle to equilibrium
  last += init_duration;  // adjust the ending time
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
void ArrayCurrentPulseGenerator::dataCollection(float currentTime)
{
    (*outFile) <<  currentTime;
    for (int _i = 0; _i < num_streams; _i++)
      (*outFile) << std::fixed << fieldDelimiter << I[_i];
    (*outFile) << "\n";
}

void ArrayCurrentPulseGenerator::update(RNG& rng) 
{
  float currentTime = (getSimulation().getIteration() * (*deltaT));
  (*this.*fpt_update)(rng, currentTime);
  if (write_to_file and currentTime > time_write_data)
  {
    this->dataCollection(currentTime);
    time_write_data += io_interval;
  }
}

void ArrayCurrentPulseGenerator::finalize(RNG& rng) 
{
  if (write_to_file == 1)
    outFile->close();
}
/*
 * Sequence:
 *     init_off_on(peak)_off_on(peak+inc)_off_on(peak+2*inc)_... until 'last'
 *  init = init_duration period
 *  off  = delay 
 *  on   = duration 
 * I(t) = peak + inc * (iteration-1)
 */
void ArrayCurrentPulseGenerator::update_PeriodicProtocol(RNG& rng, float currentTime)
{
  for (int _i = 0; _i < num_streams; _i++)
  {
    I[_i] = 0.0;
    if (currentTime >= (nextPulses[_i] + streams_duration[_i]) && currentTime <= last)
    {  // no pulse
      I[_i] = 0.0;
      streams_peakInc[_i] += inc;
      nextPulses[_i] += period;
    }
    else if (currentTime >= nextPulses[_i] && currentTime <= last)
    {  // having pulse
      I[_i] = streams_peakInc[_i];
    }
  }
}
/*
 * same as 'periodic', except the 'on' time is random
 */
void ArrayCurrentPulseGenerator::update_PoissonProtocol(RNG& rng, float currentTime)
{
  for (int _i = 0; _i < num_streams; _i++)
  {
//#ifdef DEBUG
//    std::cerr << _i << "  " << num_streams << ", " << currentTime << ", " << nextPulses[_i] 
//      << "," << streams_duration[_i] << std::endl;
//#endif
    I[_i] = 0.0;
    if (currentTime >= (nextPulses[_i] + streams_duration[_i]) && currentTime <= last)
    {  // no pulse
      I[_i] = 0.0;
      streams_peakInc[_i] += inc;
      nextPulses[_i] -= log(drandom(rng)) * period;
    }
    else if (currentTime >= nextPulses[_i] && currentTime <= last)
    {  // having pulse
      I[_i] = streams_peakInc[_i];
//#ifdef DEBUG
//      std::cerr << "I[" << _i << "] = " << I[_i] << std::endl;
//#endif
    }
  }
}
/*
 * Sequence:
 *     off_on(I(t))_off_on(I(t+1))_off_on(I(t+2))_...
 * I(t+iteration) = (peak + inc * (iteration-1)) * dual_exp
 */
void ArrayCurrentPulseGenerator::update_DualExpProtocol(RNG& rng, float currentTime)
{
  for (int _i = 0; _i < num_streams; _i++)
  {
    I[_i] = 0.0;
    if (currentTime >= (nextPulses[_i] + streams_duration[_i]) && currentTime <= last)
    {  // no pulse
      I[_i] = 0.0;
      streams_peakInc[_i] += inc;
      nextPulses[_i] += period;
    }
    else if (currentTime >= nextPulses[_i] && currentTime <= last)
    {  // having pulse
      float time_offset = currentTime - nextPulses[_i];
      I[_i] = streams_peakInc[_i] * (1 - exp(-time_offset / riseTime)) * (exp(-time_offset / decayTime));
    }
  }
}
/*
 * No repeat:
 *     _(delay)_0--(increase linearly)------------maxRamp
 *              |                                 |
 *            time_start                        time_ramp_end
 *  delay = time until time_start
 *  peak = maxRamp (Iend ~ peak ~ peakInc)
 *  duration = total_ramp_time = time_ramp_end - time_start
 *  ASSUMPTION: Istart = 0 
 *              no repetition (i.e. single cycle), i.e. last=delay+duration
 */
void ArrayCurrentPulseGenerator::update_RampProtocol(RNG& rng, float currentTime)
{
  for (int _i = 0; _i < num_streams; _i++)
  {
    I[_i] = 0.0;
    if (currentTime >= (nextPulses[_i] + streams_duration[_i]) && currentTime <= last)
    {  // no pulse
      I[_i] = 0.0;
      streams_peakInc[_i] += inc;
      nextPulses[_i] += period;
    }
    else if (currentTime >= nextPulses[_i] && currentTime <= last)
    {  // having pulse
      float time_offset = currentTime - nextPulses[_i];
      I[_i] = streams_peakInc[_i] * time_offset / duration;
    }
  }
}


/*
 * 1 train = _|||_
 *      peak=200pA
 *      duration=2ms
 *      intratrain_gap=20ms (i.e. 50Hz)
 *      num_pulses_per_train=3    (i.e. triplet)
 * num_trains = 4
 *      inter_train_gap=period=200ms    (i.e. 5Hz)
 *      num_trains=4  repeat 4 times and time must be < last
 *  _|||____|||___|||___|||
 *NOTE: 
 *___[ ]______[ ]______[ ]______[  ]
 *delay
 *   dura
 *    intertrian
 */
void ArrayCurrentPulseGenerator::update_PeriodicTrainProtocol(RNG& rng, float currentTime)
{
  for (int _i = 0; _i < num_streams; _i++)
  {
    I[_i] = 0.0;
    if (currentTime > last or currentTime < time_start_train or 
        currentTime > num_trains * period)
    {
      //do nothing
    }
    else
    {
      if (currentTime >= (time_start_train + period) )
      {//reset new  train
        streams_peakInc[_i] = peak;
        num_completed_trains += 1;
        num_completed_pulses_per_train = 0;
        time_start_train += period;
        nextPulses[_i] = time_start_train;
      }

      if (currentTime >= (nextPulses[_i] + duration) )
      {  // no pulse
        I[_i] = 0.0;
        streams_peakInc[_i] += inc;
        nextPulses[_i] += intra_train_gap;
        num_completed_pulses_per_train += 1;
      }
      else if (currentTime >= nextPulses[_i] && currentTime <= nextPulses[_i] + duration &&
          num_completed_pulses_per_train < num_pulses_per_train)
      {  // having pulse
        I[_i] = streams_peakInc[_i];
      }
    }
  }
}

/*
 * the time of input is random 
 * and once it is triggered, the current amplitude is also random as a function of 
 *      Gaussian (mean, sd)
 */
void ArrayCurrentPulseGenerator::update_WhiteNoiseProtocol(RNG& rng, float currentTime)
{
  for (int _i = 0; _i < num_streams; _i++)
  {
    I[_i] = 0.0;
    if (currentTime >= (nextPulses[_i] + duration) && currentTime <= last)
    {  // no pulse
      I[_i] = 0.0;
      nextPulses[_i] -= log(drandom(rng)) * period;
    }
    else if (currentTime >= nextPulses[_i] && currentTime <= last)
    {  // having pulse
      I[_i] = gaussian(mean, sd, rng);
    }
  }
}

ArrayCurrentPulseGenerator::ArrayCurrentPulseGenerator() 
   : CG_ArrayCurrentPulseGenerator(), outFile(0)
{
}

ArrayCurrentPulseGenerator::~ArrayCurrentPulseGenerator() 
{
  if (write_to_file) 
    delete outFile; 
}

void ArrayCurrentPulseGenerator::duplicate(std::unique_ptr<ArrayCurrentPulseGenerator>&& dup) const
{
   dup.reset(new ArrayCurrentPulseGenerator(*this));
}

void ArrayCurrentPulseGenerator::duplicate(std::unique_ptr<Variable>duplicate(std::unique_ptr<Variable>& dup)duplicate(std::unique_ptr<Variable>& dup) dup) const
{
   dup.reset(new ArrayCurrentPulseGenerator(*this));
}

void ArrayCurrentPulseGenerator::duplicate(std::unique_ptr<CG_ArrayCurrentPulseGenerator>&& dup) const
{
   dup.reset(new ArrayCurrentPulseGenerator(*this));
}
void ArrayCurrentPulseGenerator::reset_current()
{
    for (int _i = 0; _i < num_streams; _i++)
      I[_i] = 0.0;
}
void ArrayCurrentPulseGenerator::rerand_streams_info(RNG& rng)
{
  for (int _i = 0; _i < num_streams; _i++)
  {
    streams_duration[_i] = duration + drandom(-duration*0.2,+duration*0.2, rng);
    streams_delay[_i] = delay + drandom(-delay*0.2,+delay*0.2, rng);
  }
}
