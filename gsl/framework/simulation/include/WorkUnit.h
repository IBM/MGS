// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef WORKUNIT_H
#define WORKUNIT_H
#include "Copyright.h"
#include "RNG.h"

// Definition of a work unit class


class WorkUnit
{
   public:
      virtual void execute() = 0;
      virtual ~WorkUnit() {}
};
#endif
