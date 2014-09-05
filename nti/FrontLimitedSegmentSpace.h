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

