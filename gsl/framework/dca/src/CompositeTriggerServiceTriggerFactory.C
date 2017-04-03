// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "CompositeTriggerServiceTriggerFactory.h"
#include "NDPairList.h"
#include "CompositeTriggerServiceTriggerDescriptor.h"

extern "C"
{
   TriggerType* CompositeTriggerServiceTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new CompositeTriggerServiceTriggerDescriptor(sim));
   }
}

#include "TypeClassifier.h"

CompositeTriggerServiceTriggerFactory _CompositeTriggerServiceTriggerFactory;
#include "FactoryMap.h"
#include "TriggerType.h"
#include <list>



CompositeTriggerServiceTriggerFactory::CompositeTriggerServiceTriggerFactory()
{
   //   std::cout<<"CompositeTriggerServiceTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("CompositeTriggerServiceTrigger", CompositeTriggerServiceTriggerFactoryFunction);
}

