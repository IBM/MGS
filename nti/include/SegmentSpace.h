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

#ifndef SEGMENTSPACE_H
#define SEGMENTSPACE_H

#include <mpi.h>
#include "Segment.h"

class SegmentSpace
{
   public:
    virtual ~SegmentSpace() {}
    virtual bool isInSpace(Segment* Segment)=0;
    virtual SegmentSpace* duplicate()=0;
};
 
#endif

