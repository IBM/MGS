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
