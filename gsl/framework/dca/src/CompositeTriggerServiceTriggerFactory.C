// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

