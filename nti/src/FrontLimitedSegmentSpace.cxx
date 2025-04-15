// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
