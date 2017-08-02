// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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

