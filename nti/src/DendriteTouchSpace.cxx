// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "DendriteTouchSpace.h"
#include "Branch.h"

DendriteTouchSpace::DendriteTouchSpace()
{
}


DendriteTouchSpace::DendriteTouchSpace(DendriteTouchSpace& dendriteTouchSpace) :
  _segmentDescriptor(dendriteTouchSpace._segmentDescriptor)
{
}

bool DendriteTouchSpace::isInSpace(key_size_t segKey1)
{
  return (_segmentDescriptor.getBranchType(segKey1) == Branch::_SOMA ||
	  _segmentDescriptor.getBranchType(segKey1) == Branch::_BASALDEN || 
	  _segmentDescriptor.getBranchType(segKey1) == Branch::_APICALDEN );
}

bool DendriteTouchSpace::areInSpace(key_size_t segKey1, key_size_t segKey2)
{
  return (	(_segmentDescriptor.getBranchType(segKey1) == Branch::_SOMA ||
		 _segmentDescriptor.getBranchType(segKey1) == Branch::_BASALDEN || 
		 _segmentDescriptor.getBranchType(segKey1) == Branch::_APICALDEN ) &&

		 (_segmentDescriptor.getBranchType(segKey1) == Branch::_SOMA ||
		  _segmentDescriptor.getBranchType(segKey2) == Branch::_BASALDEN || 
		  _segmentDescriptor.getBranchType(segKey2) == Branch::_APICALDEN ) );
}

DendriteTouchSpace::~DendriteTouchSpace()
{
}

TouchSpace* DendriteTouchSpace::duplicate()
{
  return new DendriteTouchSpace(*this);
}
