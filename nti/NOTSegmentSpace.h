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

