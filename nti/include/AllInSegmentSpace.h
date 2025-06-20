// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ALLINSEGMENTSPACE_H
#define ALLINSEGMENTSPACE_H

#include <mpi.h>
#include "SegmentSpace.h"

class AllInSegmentSpace : public SegmentSpace
{
 public:
  AllInSegmentSpace() {}
  AllInSegmentSpace(AllInSegmentSpace& allInSpegmentSpace) {}
  ~AllInSegmentSpace() {}
  inline bool isInSpace(Segment* seg) {return true;}
  SegmentSpace* duplicate() {return new AllInSegmentSpace(*this);}
};
 
#endif

