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
      virtual void duplicate(std::auto_ptr<TriggerType>& dup) const;
      virtual ~DoubleTriggerDescriptor();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
