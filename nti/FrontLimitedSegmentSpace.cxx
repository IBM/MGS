// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "FrontLimitedSegmentSpace.h"

FrontLimitedSegmentSpace::FrontLimitedSegmentSpace(TissueGrowthSimulator& sim)
	: _sim(sim)
{
}

FrontLimitedSegmentSpace::FrontLimitedSegmentSpace(FrontLimitedSegmentSpace& frontLimitedSegmentSpace)
	: _sim(frontLimitedSegmentSpace._sim)
{
}

SegmentSpace* FrontLimitedSegmentSpace::duplicate()
{
  return new FrontLimitedSegmentSpace(*this);
}

FrontLimitedSegmentSpace::~FrontLimitedSegmentSpace()
{
}
