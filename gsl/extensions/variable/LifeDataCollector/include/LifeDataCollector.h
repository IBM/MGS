#ifndef LifeDataCollector_H
#define LifeDataCollector_H

#include "Lens.h"
#include "CG_LifeDataCollector.h"
#include <memory>

class LifeDataCollector : public CG_LifeDataCollector
{
   public:
      void initialize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      LifeDataCollector();
      virtual ~LifeDataCollector();
      virtual void duplicate(std::auto_ptr<LifeDataCollector>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_LifeDataCollector>& dup) const;
};

#endif
