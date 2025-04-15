// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ORTOUCHSPACE_H
#define ORTOUCHSPACE_H
#include "TouchSpace.h"
#include <map>
#include <list>

#include <mpi.h>

class ORTouchSpace : public TouchSpace
{
 public:
   ORTouchSpace(TouchSpace& touchSpace1,
		TouchSpace& touchSpace2);
   ORTouchSpace(ORTouchSpace& orTouchSpace);
   ~ORTouchSpace();
   bool isInSpace(key_size_t key);
   bool areInSpace(key_size_t key1, key_size_t key2);
   TouchSpace* duplicate();
 private:
   TouchSpace* _touchSpace1;
   TouchSpace* _touchSpace2;
};
 
#endif

