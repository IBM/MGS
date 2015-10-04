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

#include "NonautapticAxodendriticTouchSpace.h"

NonautapticAxodendriticTouchSpace::NonautapticAxodendriticTouchSpace()
{
}

NonautapticAxodendriticTouchSpace::NonautapticAxodendriticTouchSpace(NonautapticAxodendriticTouchSpace& nonautapticAxodendriticTouchSpace) :
  _segmentDescriptor(nonautapticAxodendriticTouchSpace._segmentDescriptor)
{
}

bool NonautapticAxodendriticTouchSpace::isInSpace(double segKey1)
{
  return (_segmentDescriptor.getBranchType(segKey1) == 1);
}

bool NonautapticAxodendriticTouchSpace::areInSpace(double segKey1, double segKey2)
{
  return (	(_segmentDescriptor.getNeuronIndex(segKey1)!= 
		 _segmentDescriptor.getNeuronIndex(segKey2)) &&

		(_segmentDescriptor.getBranchType(segKey1) == 1) &&
		
		(_segmentDescriptor.getBranchType(segKey2) == 2 || 
		 _segmentDescriptor.getBranchType(segKey2) == 3 ) );
}

TouchSpace* NonautapticAxodendriticTouchSpace::duplicate()
{
  return new NonautapticAxodendriticTouchSpace(*this);
}

NonautapticAxodendriticTouchSpace::~NonautapticAxodendriticTouchSpace()
{
}
