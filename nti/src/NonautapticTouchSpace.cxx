// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NonautapticTouchSpace.h"
#include "Branch.h"

NonautapticTouchSpace::NonautapticTouchSpace()
{
}

NonautapticTouchSpace::NonautapticTouchSpace(NonautapticTouchSpace& nonautapticTouchSpace) :
  _segmentDescriptor(nonautapticTouchSpace._segmentDescriptor)
{
}

bool NonautapticTouchSpace::isInSpace(key_size_t segKey1)
{
  return true;
}

bool NonautapticTouchSpace::areInSpace(key_size_t segKey1, key_size_t segKey2)
{
  return ( _segmentDescriptor.getNeuronIndex(segKey1)!= 
	   _segmentDescriptor.getNeuronIndex(segKey2) );
}

TouchSpace* NonautapticTouchSpace::duplicate()
{
  return new NonautapticTouchSpace(*this);
}

NonautapticTouchSpace::~NonautapticTouchSpace()
{
}
