// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef DENDRITETOUCHSPACE_H
#define DENDRITETOUCHSPACE_H
#include "TouchSpace.h"
#include "SegmentDescriptor.h"

#include <mpi.h>

class DendriteTouchSpace : public TouchSpace
{
 public:
   DendriteTouchSpace();
   DendriteTouchSpace(DendriteTouchSpace& dendriteTouchSpace);
   ~DendriteTouchSpace();
   bool isInSpace(double segKey1);
   bool areInSpace(double segKey1, double segKey2);
   TouchSpace* duplicate();
 private:
   SegmentDescriptor _segmentDescriptor;
};
 
#endif

