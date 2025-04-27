#include "Lens.h"
#include "AtasoyNFUnitDataCollector.h"
#include "CG_AtasoyNFUnitDataCollector.h"
#include <memory>

void AtasoyNFUnitDataCollector::initialize(RNG& rng) 
{
}

void AtasoyNFUnitDataCollector::finalize(RNG& rng) 
{
}

void AtasoyNFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
}

AtasoyNFUnitDataCollector::AtasoyNFUnitDataCollector() 
   : CG_AtasoyNFUnitDataCollector()
{
}

AtasoyNFUnitDataCollector::~AtasoyNFUnitDataCollector() 
{
}

void AtasoyNFUnitDataCollector::duplicate(std::unique_ptr<AtasoyNFUnitDataCollector>&& dup) const
{
   dup.reset(new AtasoyNFUnitDataCollector(*this));
}

void AtasoyNFUnitDataCollector::duplicate(std::unique_ptr<Variable>duplicate(std::unique_ptr<Variable>& dup)duplicate(std::unique_ptr<Variable>& dup) dup) const
{
   dup.reset(new AtasoyNFUnitDataCollector(*this));
}

void AtasoyNFUnitDataCollector::duplicate(std::unique_ptr<CG_AtasoyNFUnitDataCollector>&& dup) const
{
   dup.reset(new AtasoyNFUnitDataCollector(*this));
}

