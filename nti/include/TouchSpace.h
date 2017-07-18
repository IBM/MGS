// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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

