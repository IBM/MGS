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

#ifndef NONAUTAPTICAXODENDRITICTOUCHSPACE_H
#define NONAUTAPTICAXODENDRITICTOUCHSPACE_H
#include "TouchSpace.h"
#include "SegmentDescriptor.h"

#include <mpi.h>

class NonautapticAxodendriticTouchSpace : public TouchSpace
{
 public:
   NonautapticAxodendriticTouchSpace();
   NonautapticAxodendriticTouchSpace(NonautapticAxodendriticTouchSpace& nonautapticAxodendriticTouchSpace);
   ~NonautapticAxodendriticTouchSpace();
   bool isInSpace(key_size_t segKey1);
   bool areInSpace(key_size_t segKey1, key_size_t segKey2);
   TouchSpace* duplicate();
 private:
   SegmentDescriptor _segmentDescriptor;
};
 
#endif

