// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TriggerableCaller_H
#define TriggerableCaller_H
#include "Copyright.h"

#include "Trigger.h"
#include "Triggerable.h"
#include "NDPairList.h"
#include <vector>

#include <memory>

class Trigger;

class TriggerableCaller
{

   public:      
      TriggerableCaller()
	 : _ndPairList(0) {}
      TriggerableCaller(NDPairList* ndPairList)
	 : _ndPairList(ndPairList) {}
      virtual void event(Trigger* trigger) = 0;
      virtual Triggerable* getTriggerable() = 0;
      virtual void duplicate(std::unique_ptr<TriggerableCaller>&& dup) const = 0;
      virtual ~TriggerableCaller() {};
   protected:
      NDPairList* _ndPairList;
};
#endif
