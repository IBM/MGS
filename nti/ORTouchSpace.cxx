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

#include "ORTouchSpace.h"

ORTouchSpace::ORTouchSpace(TouchSpace& touchSpace1,
			   TouchSpace& touchSpace2)
{
  _touchSpace1=touchSpace1.duplicate();
  _touchSpace2=touchSpace2.duplicate();
}

ORTouchSpace::ORTouchSpace(ORTouchSpace& orTouchSpace)
{
  _touchSpace1=orTouchSpace._touchSpace1->duplicate();
  _touchSpace2=orTouchSpace._touchSpace2->duplicate();
}

bool ORTouchSpace::isInSpace(double key)
{ 
  return (_touchSpace1->isInSpace(key) ||
	  _touchSpace2->isInSpace(key) );
}

bool ORTouchSpace::areInSpace(double key1, double key2)
{
  return (_touchSpace1->areInSpace(key1, key2) ||
	  _touchSpace2->areInSpace(key1, key2) );
}

TouchSpace* ORTouchSpace::duplicate() 
{
  return new ORTouchSpace(*this);
}

ORTouchSpace::~ORTouchSpace()
{
  delete _touchSpace1;
  delete _touchSpace2;
}
