// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
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

void PointCalciumSource::duplicate(std::auto_ptr<PointCalciumSource>& dup) const
{
   dup.reset(new PointCalciumSource(*this));
}

void PointCalciumSource::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new PointCalciumSource(*this));
}

void PointCalciumSource::duplicate(std::auto_ptr<CG_PointCalciumSource>& dup) const
{
   dup.reset(new PointCalciumSource(*this));
}

