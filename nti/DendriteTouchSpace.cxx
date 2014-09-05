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

#include "DendriteTouchSpace.h"

DendriteTouchSpace::DendriteTouchSpace()
{
}


DendriteTouchSpace::DendriteTouchSpace(DendriteTouchSpace& dendriteTouchSpace) :
  _segmentDescriptor(dendriteTouchSpace._segmentDescriptor)
{
}

bool DendriteTouchSpace::isInSpace(double segKey1)
{
  return (_segmentDescriptor.getBranchType(segKey1) == 0 ||
	  _segmentDescriptor.getBranchType(segKey1) == 2 || 
	  _segmentDescriptor.getBranchType(segKey1) == 3 );
}

bool DendriteTouchSpace::areInSpace(double segKey1, double segKey2)
{
  return (	(_segmentDescriptor.getBranchType(segKey1) == 0 ||
		 _segmentDescriptor.getBranchType(segKey1) == 2 || 
		 _segmentDescriptor.getBranchType(segKey1) == 3 ) &&

		 (_segmentDescriptor.getBranchType(segKey1) == 0 ||
		  _segmentDescriptor.getBranchType(segKey2) == 2 || 
		  _segmentDescriptor.getBranchType(segKey2) == 3 ) );
}

DendriteTouchSpace::~DendriteTouchSpace()
{
}

TouchSpace* DendriteTouchSpace::duplicate()
{
  return new DendriteTouchSpace(*this);
}
