// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

