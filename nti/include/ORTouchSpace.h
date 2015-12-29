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

