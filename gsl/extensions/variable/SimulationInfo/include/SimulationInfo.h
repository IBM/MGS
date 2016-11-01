#ifndef SimulationInfo_H
#define SimulationInfo_H

#include "Lens.h"
#include "CG_SimulationInfo.h"
#include <memory>

class SimulationInfo : public CG_SimulationInfo
{
   public:
      void initialize(RNG& rng);
      void calculateInfo(RNG& rng);
      SimulationInfo();
      virtual ~SimulationInfo();
      virtual void duplicate(std::auto_ptr<SimulationInfo>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_SimulationInfo>& dup) const;
};

#endif
