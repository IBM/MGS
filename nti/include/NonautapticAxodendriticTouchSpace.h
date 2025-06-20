// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

