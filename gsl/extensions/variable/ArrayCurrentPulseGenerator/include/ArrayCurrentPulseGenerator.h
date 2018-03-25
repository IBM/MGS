#ifndef ArrayCurrentPulseGenerator_H
#define ArrayCurrentPulseGenerator_H

#include "Lens.h"
#include "CG_ArrayCurrentPulseGenerator.h"
#include <memory>
#include <fstream>

class ArrayCurrentPulseGenerator : public CG_ArrayCurrentPulseGenerator
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void finalize(RNG& rng);
      ArrayCurrentPulseGenerator();
      virtual ~ArrayCurrentPulseGenerator();
      virtual void duplicate(std::auto_ptr<ArrayCurrentPulseGenerator>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ArrayCurrentPulseGenerator>& dup) const;
   private:
      void update_PeriodicProtocol(RNG& , float currentTime);
      void update_DualExpProtocol(RNG& , float currentTime);
      void update_PoissonProtocol(RNG& , float currentTime);
      void update_PeriodicTrainProtocol(RNG& , float currentTime);
      void update_WhiteNoiseProtocol(RNG& rng, float currentTime);
      void update_RampProtocol(RNG& , float currentTime);

      std::vector<float> nextPulses; //[ms]
      float peakInc; //[pA]
      int   num_completed_trains; //[number of completed trains]
      int   num_completed_pulses_per_train; //[number of pulses complated in 1 pulse-train]
      void (ArrayCurrentPulseGenerator::*fpt_update)(RNG& rng, float currentTime) = NULL;
      void dataCollection(float currentTime);
      float time_start_train;  //[ms]
      std::ofstream* outFile = 0;
      float time_write_data; // [ms]
      std::vector<float> streams_duration;
      std::vector<float> streams_delay;
      std::vector<float> streams_peakInc;
      //dyn_var_t[] streams_duration;
      //dyn_var_t[] streams_delay;
      ////dyn_var_t[] checked_time; // msec (time-point reference to calculate delay_until_adjust_noise)
      void reset_current();
      void rerand_streams_info(RNG& rng);
};

#endif
