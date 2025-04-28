#ifndef AtasoyNFUnitDataCollector_H
#define AtasoyNFUnitDataCollector_H

#include "Lens.h"
#include "CG_AtasoyNFUnitDataCollector.h"
#include <memory>

class AtasoyNFUnitDataCollector : public CG_AtasoyNFUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      AtasoyNFUnitDataCollector();
      virtual ~AtasoyNFUnitDataCollector();
      virtual void duplicate(std::unique_ptr<AtasoyNFUnitDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_AtasoyNFUnitDataCollector>&& dup) const;
};

#endif
