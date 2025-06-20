// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SEMAPHORETRIGGERDESCRIPTOR_H
#define SEMAPHORETRIGGERDESCRIPTOR_H
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

class SemaphoreTriggerDescriptor : public TriggerType
{
   public:
      SemaphoreTriggerDescriptor(Simulation& s);
      Trigger* getTrigger(NDPairList& ndp);
      Trigger* getTrigger(std::vector<DataItem*> const & args);
      virtual void duplicate(std::unique_ptr<TriggerType>& dup) const;
      virtual ~SemaphoreTriggerDescriptor();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
