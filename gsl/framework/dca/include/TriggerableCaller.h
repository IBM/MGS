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
