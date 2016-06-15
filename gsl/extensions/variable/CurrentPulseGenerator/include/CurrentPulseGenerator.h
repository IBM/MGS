#ifndef CurrentPulseGenerator_H
#define CurrentPulseGenerator_H

#include "Lens.h"
#include "CG_CurrentPulseGenerator.h"
#include <memory>

class CurrentPulseGenerator : public CG_CurrentPulseGenerator
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      CurrentPulseGenerator();
      virtual ~CurrentPulseGenerator();
      virtual void duplicate(std::auto_ptr<CurrentPulseGenerator>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_CurrentPulseGenerator>& dup) const;
   private:
      float nextPulse; //[ms]
      float peakInc; //[pA]
};

#endif
