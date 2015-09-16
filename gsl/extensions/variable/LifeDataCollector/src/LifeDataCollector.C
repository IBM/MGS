#include "Lens.h"
#include "LifeDataCollector.h"
#include "CG_LifeDataCollector.h"
#include <memory>

void LifeDataCollector::initialize(RNG& rng) 
{
}

void LifeDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
}

LifeDataCollector::LifeDataCollector() 
   : CG_LifeDataCollector()
{
}

LifeDataCollector::~LifeDataCollector() 
{
}

void LifeDataCollector::duplicate(std::auto_ptr<LifeDataCollector>& dup) const
{
   dup.reset(new LifeDataCollector(*this));
}

void LifeDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new LifeDataCollector(*this));
}

void LifeDataCollector::duplicate(std::auto_ptr<CG_LifeDataCollector>& dup) const
{
   dup.reset(new LifeDataCollector(*this));
}

