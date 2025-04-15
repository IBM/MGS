// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BoolTriggerFactory.h"

BoolTriggerFactory _BoolTriggerFactory;

#include "BoolTriggerDescriptor.h"
#include "FactoryMap.h"
#include "TriggerType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   TriggerType* BoolTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new BoolTriggerDescriptor(sim));
   }
}


BoolTriggerFactory::BoolTriggerFactory()
{
   //   std::cout<<"BoolTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("BoolTrigger", BoolTriggerFactoryFunction);
}

