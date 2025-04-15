// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMPOSITETRIGGERSERVICETRIGGERDESCRIPTOR_H
#define COMPOSITETRIGGERSERVICETRIGGERDESCRIPTOR_H
#include "Copyright.h"

#include "TriggerType.h"

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <memory>

class BoolDataItem;
class Simulation;
class Trigger;
class DataItem;
class Trigger;
class NDPairList;

class CompositeTriggerServiceTriggerDescriptor : public TriggerType
{
   public:
      CompositeTriggerServiceTriggerDescriptor(Simulation& s);
      Trigger* getTrigger(NDPairList& ndp);
      Trigger* getTrigger(std::vector<DataItem*> const & args);
      virtual void duplicate(std::unique_ptr<TriggerType>& dup) const;
      virtual ~CompositeTriggerServiceTriggerDescriptor();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
