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
