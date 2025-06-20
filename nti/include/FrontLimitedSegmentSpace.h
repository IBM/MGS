// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FRONTLIMITEDSEGMENTSPACE_H
#define FRONTLIMITEDSEGMENTSPACE_H

#include <mpi.h>
#include "SegmentSpace.h"
#include "TissueGrowthSimulator.hpp"

class FrontLimitedSegmentSpace : public SegmentSpace
{
 public:
  FrontLimitedSegmentSpace(TissueGrowthSimulator& sim);
  FrontLimitedSegmentSpace(FrontLimitedSegmentSpace&);
  ~FrontLimitedSegmentSpace();
  inline bool isInSpace(Segment* seg) {
    return (seg->getFrontLevel()<=_sim.getFrontNumber())?true:false;
  }
  SegmentSpace* duplicate();
  
 private:
  TissueGrowthSimulator& _sim;
};
 
#endif

