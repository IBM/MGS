// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
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
