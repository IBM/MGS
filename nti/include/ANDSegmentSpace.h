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

