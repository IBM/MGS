// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

bool ORTouchSpace::isInSpace(key_size_t key)
{ 
  return (_touchSpace1->isInSpace(key) ||
	  _touchSpace2->isInSpace(key) );
}

bool ORTouchSpace::areInSpace(key_size_t key1, key_size_t key2)
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
