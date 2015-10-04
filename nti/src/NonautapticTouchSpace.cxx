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

#include "NonautapticTouchSpace.h"

NonautapticTouchSpace::NonautapticTouchSpace()
{
}

NonautapticTouchSpace::NonautapticTouchSpace(NonautapticTouchSpace& nonautapticTouchSpace) :
  _segmentDescriptor(nonautapticTouchSpace._segmentDescriptor)
{
}

bool NonautapticTouchSpace::isInSpace(double segKey1)
{
  return true;
}

bool NonautapticTouchSpace::areInSpace(double segKey1, double segKey2)
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
