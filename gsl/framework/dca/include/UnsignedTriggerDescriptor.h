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

#ifndef UNSIGNEDTRIGGERDESCRIPTOR_H
#define UNSIGNEDTRIGGERDESCRIPTOR_H
#include "Copyright.h"

#include "TriggerType.h"
#include "NDPairList.h"

#include <vector>

class Simulation;
class Trigger;
class DataItem;

class UnsignedTriggerDescriptor : public TriggerType
{
   public:
      UnsignedTriggerDescriptor(Simulation& s);
      Trigger* getTrigger(NDPairList& ndp);
      Trigger* getTrigger(const std::vector<DataItem*>& args);
      virtual void duplicate(std::auto_ptr<TriggerType>& dup) const;
      virtual ~UnsignedTriggerDescriptor();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
   private:
      Simulation& _sim;
};
#endif
