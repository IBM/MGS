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

#ifndef FLOATTRIGGERDESCRIPTOR_H
#define FLOATTRIGGERDESCRIPTOR_H
#include "Copyright.h"

#include "TriggerType.h"

#include <vector>

class Simulation;
class Trigger;
class DataItem;
class NDPairList;

class FloatTriggerDescriptor : public TriggerType
{
   public:
      FloatTriggerDescriptor(Simulation& s);
      Trigger* getTrigger(NDPairList& ndp);
      Trigger* getTrigger(std::vector<DataItem*> const & args);
      virtual void duplicate(std::unique_ptr<TriggerType>& dup) const;
      virtual ~FloatTriggerDescriptor();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
