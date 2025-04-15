// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

