// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-06-21-2012-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef SEGMENTKEYSEGMENTSPACE_H
#define SEGMENTKEYSEGMENTSPACE_H

#include <mpi.h>
#include "SegmentSpace.h"
#include "SegmentDescriptor.h"
#include "TissueGrowthSimulator.hpp"

#include <vector>
#include <utility>
#include <string>

class SegmentKeySegmentSpace : public SegmentSpace
{
 public:
  SegmentKeySegmentSpace(std::vector<std::pair<std::string, unsigned int> > probeKey);
  SegmentKeySegmentSpace(SegmentKeySegmentSpace& segmentKeySegmentSpace);
  ~SegmentKeySegmentSpace();
  
  bool isInSpace(Segment* seg);  
  SegmentSpace* duplicate();

 private:
  std::vector<std::pair<std::string, unsigned int> > _probeKey;
  SegmentDescriptor _segmentDescriptor;
};
 
#endif

