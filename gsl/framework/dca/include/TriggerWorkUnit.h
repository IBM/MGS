// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
