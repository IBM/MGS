// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TOUCHSPACE_H
#define TOUCHSPACE_H

#include<cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "MaxComputeOrder.h"
class TouchSpace
{
   public:
     virtual ~TouchSpace() {}
     virtual bool isInSpace(key_size_t segKey)=0;
     virtual bool areInSpace(key_size_t segKey1, key_size_t segKey2)=0;
     virtual TouchSpace* duplicate()=0;
};
 
#endif

