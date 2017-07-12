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

void PointCurrentSource::duplicate(std::auto_ptr<PointCurrentSource>& dup) const
{
   dup.reset(new PointCurrentSource(*this));
}

void PointCurrentSource::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new PointCurrentSource(*this));
}

void PointCurrentSource::duplicate(std::auto_ptr<CG_PointCurrentSource>& dup) const
{
   dup.reset(new PointCurrentSource(*this));
}

