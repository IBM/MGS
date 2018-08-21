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

#ifndef ORSEGMENTSPACE_H
#define ORSEGMENTSPACE_H

#include <mpi.h>
#include "SegmentSpace.h"

class SegmentSpace;

class ORSegmentSpace : public SegmentSpace
{
 public:
  ORSegmentSpace(SegmentSpace* segmentSpace1, SegmentSpace* segmentSpace2) : _segmentSpace1(segmentSpace1), _segmentSpace2(segmentSpace2) {}
  ORSegmentSpace(ORSegmentSpace& andSegmentSpace) : _segmentSpace1(andSegmentSpace._segmentSpace1), _segmentSpace2(andSegmentSpace._segmentSpace2) {}
  ~ORSegmentSpace() {}
  inline bool isInSpace(Segment* seg) {return _segmentSpace1->isInSpace(seg) || _segmentSpace2->isInSpace(seg);}
  SegmentSpace* duplicate() {return new ORSegmentSpace(*this);}
 private:
  SegmentSpace *_segmentSpace1, *_segmentSpace2;
};
 
#endif

