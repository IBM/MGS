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
