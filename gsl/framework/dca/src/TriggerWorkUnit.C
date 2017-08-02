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
