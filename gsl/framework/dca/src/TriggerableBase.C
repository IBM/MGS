// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TriggerableBase.h"
#include "TriggerableCaller.h"
#include "Trigger.h"
#include "DuplicatePointerArray.h"
#include <vector>
#include <cassert>

void TriggerableBase::addTrigger(
   Trigger* trigger, const std::string& functionName, std::unique_ptr<NDPairList>& ndpList)
{
   NDPairList* ndp = ndpList.release();
   if (ndp) {
      _ndPairLists.push_back(ndp);
   }
   std::unique_ptr<TriggerableCaller> cup;
   EventType type = createTriggerableCaller(functionName, ndp, std::move(cup));
   assert(type != _UNALTERED);
   if (type == _SERIAL) {
      trigger->addSerialTriggerableCaller(cup);
   } else { // _PARALLEL
      trigger->addParallelTriggerableCaller(cup);
   }
   _triggers.push_back(trigger);
}
