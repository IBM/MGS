// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "AllInTouchSpace.h"

AllInTouchSpace::AllInTouchSpace()
{
}

AllInTouchSpace::AllInTouchSpace(AllInTouchSpace& allInTouchSpace)
{
}

bool AllInTouchSpace::isInSpace(key_size_t key)
{ 
  return true;
}

bool AllInTouchSpace::areInSpace(key_size_t key1, key_size_t key2)
{
  return true;
}

TouchSpace* AllInTouchSpace::duplicate() 
{
  return new AllInTouchSpace(*this);
}

AllInTouchSpace::~AllInTouchSpace()
{
}
