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
