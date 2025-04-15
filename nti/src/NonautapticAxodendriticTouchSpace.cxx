// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NonautapticAxodendriticTouchSpace.h"
#include "Branch.h"

NonautapticAxodendriticTouchSpace::NonautapticAxodendriticTouchSpace() {}

NonautapticAxodendriticTouchSpace::NonautapticAxodendriticTouchSpace(
    NonautapticAxodendriticTouchSpace& nonautapticAxodendriticTouchSpace)
    : _segmentDescriptor(nonautapticAxodendriticTouchSpace._segmentDescriptor)
{
}

bool NonautapticAxodendriticTouchSpace::isInSpace(key_size_t segKey1)
{
  return (_segmentDescriptor.getBranchType(segKey1) == Branch::_AXON);
}

bool NonautapticAxodendriticTouchSpace::areInSpace(key_size_t segKey1,
                                                   key_size_t segKey2)
{
  return ((_segmentDescriptor.getNeuronIndex(segKey1) !=
           _segmentDescriptor.getNeuronIndex(segKey2)) &&

          (_segmentDescriptor.getBranchType(segKey1) == Branch::_AXON) &&

          (_segmentDescriptor.getBranchType(segKey2) == Branch::_BASALDEN ||
           _segmentDescriptor.getBranchType(segKey2) == Branch::_APICALDEN));
}

TouchSpace* NonautapticAxodendriticTouchSpace::duplicate()
{
  return new NonautapticAxodendriticTouchSpace(*this);
}

NonautapticAxodendriticTouchSpace::~NonautapticAxodendriticTouchSpace() {}
