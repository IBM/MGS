// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PAUSEACTIONABLE_H
#define PAUSEACTIONABLE_H
#include "Copyright.h"


class PauseActionable
{
   public:
      virtual void action() =0;
      virtual ~PauseActionable() {}
};
#endif
