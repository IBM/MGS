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
	 std::auto_ptr<NDPairList>& ndpList);
      virtual ~TriggerableBase() {}

   protected:
      virtual EventType createTriggerableCaller(
	 const std::string& functionName, NDPairList* ndpList, 
	 std::auto_ptr<TriggerableCaller>& triggerableCaller) = 0;
   private:
      DuplicatePointerArray<NDPairList> _ndPairLists;
      std::vector<Trigger*> _triggers;
      
};
#endif
