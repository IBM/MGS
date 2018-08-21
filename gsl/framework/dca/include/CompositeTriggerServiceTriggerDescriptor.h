// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void duplicate(std::auto_ptr<TriggerType>& dup) const;
      virtual ~CompositeTriggerServiceTriggerDescriptor();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
