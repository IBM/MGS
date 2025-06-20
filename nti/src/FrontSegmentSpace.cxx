// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FrontSegmentSpace.h"

FrontSegmentSpace::FrontSegmentSpace(TissueGrowthSimulator& sim)
	: _sim(sim)
{
}

FrontSegmentSpace::FrontSegmentSpace(FrontSegmentSpace& frontSegmentSpace)
	: _sim(frontSegmentSpace._sim)
{
}

FrontSegmentSpace::~FrontSegmentSpace()
{
}

SegmentSpace* FrontSegmentSpace::duplicate()
{
  return new FrontSegmentSpace(*this);
}
