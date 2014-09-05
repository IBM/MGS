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
