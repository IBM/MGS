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

#ifndef UNSIGNEDSERVICETRIGGERDESCRIPTOR_H
#define UNSIGNEDSERVICETRIGGERDESCRIPTOR_H
#include "Copyright.h"

#include "TriggerType.h"
#include "NDPairList.h"

#include <vector>

class Simulation;
class Trigger;
class DataItem;

class UnsignedServiceTriggerDescriptor : public TriggerType
{
   public:
      UnsignedServiceTriggerDescriptor(Simulation& s);
      Trigger* getTrigger(NDPairList& ndp);
      Trigger* getTrigger(const std::vector<DataItem*>& args);
      virtual void duplicate(std::unique_ptr<TriggerType>& dup) const;
      virtual ~UnsignedServiceTriggerDescriptor();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
