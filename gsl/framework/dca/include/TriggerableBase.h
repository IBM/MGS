// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TriggerableBase_H
#define TriggerableBase_H
#include "Copyright.h"

#include "Triggerable.h"
#include "TriggerableCaller.h"
#include "DuplicatePointerArray.h"
#include <vector>

#include <memory>

class Trigger;

class TriggerableBase : public Triggerable
{
   public:      
      enum EventType {_UNALTERED, _SERIAL, _PARALLEL};

      virtual void addTrigger(
	 Trigger* trigger, const std::string& functionName, 
	 std::unique_ptr<NDPairList>& ndpList);
      virtual ~TriggerableBase() {}

   protected:
      virtual EventType createTriggerableCaller(
	 const std::string& functionName, NDPairList* ndpList, 
	 std::unique_ptr<TriggerableCaller>&& triggerableCaller) = 0;
   private:
      DuplicatePointerArray<NDPairList> _ndPairLists;
      std::vector<Trigger*> _triggers;
      
};
#endif
