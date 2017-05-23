#ifndef RampCurrentGenerator_H
#define RampCurrentGenerator_H

#include "Lens.h"
#include "CG_RampCurrentGenerator.h"
#include <memory>

class RampCurrentGenerator : public CG_RampCurrentGenerator
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      RampCurrentGenerator();
      virtual ~RampCurrentGenerator();
      virtual void duplicate(std::auto_ptr<RampCurrentGenerator>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_RampCurrentGenerator>& dup) const;
   private:
      void update_RampProtocol(RNG& , float currentTime);
      float tstart, tend; //[ms]
      float nextPulse; //[ms]
      bool first_enter_pulse;
      std::ofstream* outFile;
      float time_write_data; // [ms]
      void (CurrentPulseGenerator::*fpt_update)(RNG& rng, float currentTime) = NULL;
      void dataCollection(float currentTime);
};

#endif
