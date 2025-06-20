// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DOUBLETRIGGERDESCRIPTOR_H
#define DOUBLETRIGGERDESCRIPTOR_H
#include "Copyright.h"

#include "TriggerType.h"

#include <vector>

class Simulation;
class Trigger;
class DataItem;
class NDPairList;

class DoubleTriggerDescriptor : public TriggerType
{
   public:
      DoubleTriggerDescriptor(Simulation& s);
      Trigger* getTrigger(NDPairList& ndp);
      Trigger* getTrigger(std::vector<DataItem*> const & args);
      virtual void duplicate(std::unique_ptr<TriggerType>& dup) const;
      virtual ~DoubleTriggerDescriptor();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
