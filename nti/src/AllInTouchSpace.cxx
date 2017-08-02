// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
