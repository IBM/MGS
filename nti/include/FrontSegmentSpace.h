// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FRONTSEGMENTSPACE_H
#define FRONTSEGMENTSPACE_H

#include <mpi.h>
#include "SegmentSpace.h"
#include "TissueGrowthSimulator.hpp"
class FrontSegmentSpace : public SegmentSpace
{
 public:
  FrontSegmentSpace(TissueGrowthSimulator& sim);
  FrontSegmentSpace(FrontSegmentSpace& frontSegmentSpace);
  ~FrontSegmentSpace();
  inline bool isInSpace(Segment* seg) {
    return (seg->getFrontLevel()==_sim.getFrontNumber())?true:false;
  }
  SegmentSpace* duplicate();
  
 private:
  TissueGrowthSimulator& _sim;
};
 
#endif

