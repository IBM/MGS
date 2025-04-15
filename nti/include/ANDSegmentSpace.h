// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ANDSEGMENTSPACE_H
#define ANDSEGMENTSPACE_H

#include <mpi.h>
#include "SegmentSpace.h"

class SegmentSpace;

class ANDSegmentSpace : public SegmentSpace
{
 public:
  ANDSegmentSpace(SegmentSpace* segmentSpace1, SegmentSpace* segmentSpace2) : _segmentSpace1(segmentSpace1), _segmentSpace2(segmentSpace2) {}
  ANDSegmentSpace(ANDSegmentSpace& andSegmentSpace) : _segmentSpace1(andSegmentSpace._segmentSpace1), _segmentSpace2(andSegmentSpace._segmentSpace2) {}
  ~ANDSegmentSpace() {}
  inline bool isInSpace(Segment* seg) {return _segmentSpace1->isInSpace(seg) && _segmentSpace2->isInSpace(seg);}
  SegmentSpace* duplicate() {return new ANDSegmentSpace(*this);}
 private:
  SegmentSpace *_segmentSpace1, *_segmentSpace2;
};
 
#endif

