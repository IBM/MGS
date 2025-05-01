// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "PointCalciumSource.h"
#include "CG_PointCalciumSource.h"
#include <memory>

void PointCalciumSource::stimulate(RNG& rng) 
{
}

void PointCalciumSource::setCaCurrent(Trigger* trigger, NDPairList* ndPairList) 
{
  NDPairList::iterator iter=ndPairList->begin();
  NDPairList::iterator end=ndPairList->end();
  for (; iter!=end; ++iter) {
    if ( (*iter)->getName() == "I_Ca" ) {
      I_Ca=static_cast<NumericDataItem*>((*iter)->getDataItem())->getFloat();
    }
  }
}

PointCalciumSource::PointCalciumSource() 
   : CG_PointCalciumSource()
{
}

PointCalciumSource::~PointCalciumSource() 
{
}

void PointCalciumSource::duplicate(std::unique_ptr<PointCalciumSource>&& dup) const
{
   dup.reset(new PointCalciumSource(*this));
}

void PointCalciumSource::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new PointCalciumSource(*this));
}

void PointCalciumSource::duplicate(std::unique_ptr<CG_PointCalciumSource>&& dup) const
{
   dup.reset(new PointCalciumSource(*this));
}

