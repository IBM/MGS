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

/* NOTE:
 * In Ramp we mainly use
 *   _(delay)_Iramp[0]--(inc. linearly)---Iramp[1]
 *              |                          ||
 *            tpoints[0]                   tpoints[1] 
 *   or
 *   _(delay)_Iramp[0]--(inc. linearly)---Iramp[1]---(dec. linearly)--Iramp[2]
 *              |                                 |                    |
 *            tpoints[0]                        tpoints[1]          tpoints[2]
 * So, a single period is most likely the case
 *
 */
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
  if (not deltaT)
  {
    std::cerr << "ERROR: " << typeid(*this).name() << " needs time-step connected\n";
    assert(deltaT);
  }
  first_enter_pulse = true;
  I = 0.0; // [pA]
  if (time_points.size() <= 1)
  {
    std::cerr << "ERROR: " << typeid(*this).name() 
      << " needs time_points.size() >= 2\n";
    assert(time_points.size() > 1);
  }
  if (time_points[0] < 0.0 or time_points[0] > 0.0)
  {
    std::cerr << "ERROR: " << typeid(*this).name() << " needs \n";
    std::cerr << "   time_points[0] = 0.0; \n";
    std::cerr << " .. and you can adjust 'delay' or 'init_duration' if needed " << std::endl;
    assert(0);
  }

  n_timepoints = time_points.size();
  for (int ii = 0; ii < n_timepoints-1; ii++)
    assert(time_points[ii] < time_points[ii+1]);

  current_index = 0;
  float duration = time_points[current_index+1] - time_points[current_index];
  //if (n_timepoints == 3 and Iramp[0] = 0.0 
  //    and Iramp[1] > 0 and Iramp[3] = 0.0)
  //{
  //  // inc is used for the case inc. and dec. ramps
  //  //with increasing peak - if multiple periods are used
  //}else {
  //  std::cerr << "WARNING: " << typeid(*this).name() << ":\n";
  //  std::cerr << " ... 'inc' is used only when time_points.size() != 3, " 
  //    << "Iramp[0] = 0.0 and ";
  //    << "Iramp[1] > 0 and Iramp[3] = 0.0 \n";
  //  inc = 0.0;
  //} 
  
  if (pattern == "ramp")
  {
    tstart = delay;
    tend = delay + time_points[current_index+1];
    duration = tend - tstart;
    nextPulse = delay;
    fpt_update = &RampCurrentGenerator::update_RampProtocol;
    Istart = Iramp[0];
    Iend = Iramp[1];
  }
  else{
    std::cerr << "Unsupported pattern: Use 'ramp'\n";
    assert(0);
  }
  tstart += init_duration;
  tend += init_duration;
  last += init_duration;
  nextPulse += init_duration;
  if (duration > period)
  {
    std::cerr << "The duration of the RampCurrentGenerator stimulus should not be greater than the "
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
void RampCurrentGenerator::dataCollection(float currentTime)
{
    (*outFile) <<  currentTime;
    (*outFile) << std::fixed << fieldDelimiter << I;
    (*outFile) << "\n";
}

void RampCurrentGenerator::update(RNG& rng) 
{
  float currentTime = (getSimulation().getIteration() * (*deltaT));
  (*this.*fpt_update)(rng, currentTime);
  if (write_to_file and currentTime > time_write_data)
  {
    this->dataCollection(currentTime);
    time_write_data += io_interval;
  }
}

/*
 * No repeat:
 *   _(delay)_Istart--(increase linearly)------------Iend
 *              |                                 |
 *            time_start                        time_ramp_end
 *   _(delay)_Iramp[0]--(inc. linearly)---Iramp[1]---(dec. linearly)--Iramp[2]
 *              |                                 |                    |
 *            tpoints[0]                        tpoints[1]          tpoints[2]
 *  delay = time until time_start
 *  peak = maxRamp (Iend)
 *  NOTE: Generic version of CurrentPulseGenerator("ramp")
 *     - Istart, Iend
 *     - repetition is allowed, with 'inc' increase in Iend after each cycle
 */
void RampCurrentGenerator::update_RampProtocol(RNG& rng, float currentTime)
{
  I = 0.0;
  if (currentTime <= last)
  {
    if (currentTime >= (nextPulse + time_points[n_timepoints-1]-time_points[0]) )
    {//no pulse
      I = 0.0;
      first_enter_pulse = true;
      //if (n_timepoints == 3 and current_index == 1)
      //  Iend += inc;
      nextPulse += period;
      current_index = 0;
    }
    else if (currentTime >= nextPulse){
      if (current_index < n_timepoints - 1)
      {
        if (currentTime >= nextPulse )
        {//having pulse
          //float dt = currentTime - nextPulse;
          if (first_enter_pulse)
          {
            tstart = time_points[current_index];
            tend = tstart + duration; 
            first_enter_pulse = false;
            Iend = Iramp[current_index+1];
            Istart = Iramp[current_index];
          }
          float time_offset = currentTime - tstart;
          I = Istart + (Iend-Istart) * (time_offset/(duration));
        }
        if (currentTime + (*deltaT) >= nextPulse + time_points[current_index+1])
        {
          current_index += 1;
          if (current_index < n_timepoints - 1)
            duration = (time_points[current_index+1]-time_points[current_index]);
          first_enter_pulse = true;
        }
      }else
      {
        //do nothing
      }
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

void RampCurrentGenerator::duplicate(std::unique_ptr<RampCurrentGenerator>&& dup) const
{
   dup.reset(new RampCurrentGenerator(*this));
}

void RampCurrentGenerator::duplicate(std::unique_ptr<Variable>duplicate(std::unique_ptr<Variable>& dup)duplicate(std::unique_ptr<Variable>& dup) dup) const
{
   dup.reset(new RampCurrentGenerator(*this));
}

void RampCurrentGenerator::duplicate(std::unique_ptr<CG_RampCurrentGenerator>&& dup) const
{
   dup.reset(new RampCurrentGenerator(*this));
}

