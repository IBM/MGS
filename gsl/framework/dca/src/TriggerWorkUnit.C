// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TriggerWorkUnit.h"
#include "Trigger.h"
#include "TriggerableCaller.h"
#include <vector>
#include <iostream>

TriggerWorkUnit::TriggerWorkUnit(
   Trigger* trigger, std::vector<TriggerableCaller*>::iterator begin,
   std::vector<TriggerableCaller*>::iterator end)
   : _trigger(trigger), _begin(begin), _end(end)
{
}

void TriggerWorkUnit::execute()
{
   std::vector<TriggerableCaller*>::iterator it;
   for(it = _begin; it != _end; ++it) {
      (*it)->event(_trigger);
   }
}

void TriggerWorkUnit::resetLimits(
   std::vector<TriggerableCaller*>::iterator begin,
   std::vector<TriggerableCaller*>::iterator end)
{
   _begin = begin;
   _end = end;
}
