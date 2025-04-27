#ifndef SimulationInfoNew_H
#define SimulationInfoNew_H

#include "Lens.h"
#include "CG_SimulationInfoNew.h"
#include <memory>

class SimulationInfoNew : public CG_SimulationInfoNew
{
   public:
      void initialize(RNG& rng);
      void calculateInfo(RNG& rng);
      SimulationInfoNew();
      virtual ~SimulationInfoNew();
      virtual void duplicate(std::unique_ptr<SimulationInfoNew>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SimulationInfoNew>&& dup) const;
};

#endif
