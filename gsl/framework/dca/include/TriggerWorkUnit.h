// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TriggerWorkUnit_H
#define TriggerWorkUnit_H
#include "Copyright.h"

#include "WorkUnit.h"
#include <vector>
#include "RNG.h"

class TriggerableCaller;
class Trigger;

class TriggerWorkUnit : public WorkUnit
{
   public:
      TriggerWorkUnit(Trigger* trigger,
		      std::vector<TriggerableCaller*>::iterator begin,
		      std::vector<TriggerableCaller*>::iterator end);
      virtual void execute();
      void resetLimits(std::vector<TriggerableCaller*>::iterator begin,
		       std::vector<TriggerableCaller*>::iterator end);
   private:
      Trigger* _trigger;
      std::vector<TriggerableCaller*>::iterator _begin;
      std::vector<TriggerableCaller*>::iterator _end;
};

#endif
