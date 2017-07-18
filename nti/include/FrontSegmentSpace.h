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

