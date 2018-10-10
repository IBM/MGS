#ifndef CurrentPulseGenerator_H
#define CurrentPulseGenerator_H

#include "Lens.h"
#include "CG_CurrentPulseGenerator.h"
#include <memory>
#include <fstream>

class CurrentPulseGenerator : public CG_CurrentPulseGenerator
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void finalize(RNG& rng);
      CurrentPulseGenerator();
      virtual ~CurrentPulseGenerator();
      virtual void duplicate(std::unique_ptr<CurrentPulseGenerator>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_CurrentPulseGenerator>& dup) const;
   private:
      void update_PeriodicProtocol(RNG& , float currentTime);
      void update_DualExpProtocol(RNG& , float currentTime);
      void update_PoissonProtocol(RNG& , float currentTime);
      void update_PeriodicTrainProtocol(RNG& , float currentTime);
      void update_WhiteNoiseProtocol(RNG& rng, float currentTime);
      void update_RampProtocol(RNG& , float currentTime);

      float nextPulse; //[ms]
      float peakInc; //[pA]
      int   num_completed_trains; //[number of completed trains]
      int   num_completed_pulses_per_train; //[number of pulses complated in 1 pulse-train]
      void (CurrentPulseGenerator::*fpt_update)(RNG& rng, float currentTime) = NULL;
      void dataCollection(float currentTime);
      float time_start_train;  //[ms]
      std::ofstream* outFile = 0;
      float time_write_data; // [ms]
};

#endif
