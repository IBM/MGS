// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PointCurrentSource.h"
#include "Simulation.h"
#include "CG_PointCurrentSource.h"
#include <memory>

void PointCurrentSource::stimulate(RNG& rng) 
{
}

void PointCurrentSource::setCurrent(Trigger* trigger, NDPairList* ndPairList) 
{
  NDPairList::iterator iter=ndPairList->begin();
  NDPairList::iterator end=ndPairList->end();
  for (; iter!=end; ++iter) {
    if ( (*iter)->getName() == "I" ) {
      I=static_cast<NumericDataItem*>((*iter)->getDataItem())->getFloat();
    }
  }
}

PointCurrentSource::PointCurrentSource() 
   : CG_PointCurrentSource()
{
}

PointCurrentSource::~PointCurrentSource() 
{
}

void PointCurrentSource::duplicate(std::unique_ptr<PointCurrentSource>&& dup) const
{
   dup.reset(new PointCurrentSource(*this));
}

void PointCurrentSource::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new PointCurrentSource(*this));
}

void PointCurrentSource::duplicate(std::unique_ptr<CG_PointCurrentSource>&& dup) const
{
   dup.reset(new PointCurrentSource(*this));
}

