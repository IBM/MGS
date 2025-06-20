// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
