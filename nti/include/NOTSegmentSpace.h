// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NOTSEGMENTSPACE_H
#define NOTSEGMENTSPACE_H

#include <mpi.h>
#include "SegmentSpace.h"

class NOTSegmentSpace : public SegmentSpace
{
 public:
  NOTSegmentSpace(SegmentSpace* segmentSpace) : _segmentSpace(segmentSpace) {}
  NOTSegmentSpace(NOTSegmentSpace& notSegmentSpace) : _segmentSpace(notSegmentSpace._segmentSpace) {}
  ~NOTSegmentSpace() {}
  inline bool isInSpace(Segment* seg) {return !_segmentSpace->isInSpace(seg);}
  SegmentSpace* duplicate() {return new NOTSegmentSpace(*this);}
 private:
  SegmentSpace* _segmentSpace;
};
 
#endif

