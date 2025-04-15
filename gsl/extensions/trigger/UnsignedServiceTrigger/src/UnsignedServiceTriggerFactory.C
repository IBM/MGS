// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "UnsignedServiceTriggerFactory.h"

UnsignedServiceTriggerFactory _UnsignedServiceTriggerFactory;

#include "UnsignedServiceTriggerDescriptor.h"
#include "FactoryMap.h"
#include "TriggerType.h"
#include "NDPairList.h"

extern "C"
{
   TriggerType* UnsignedServiceTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new UnsignedServiceTriggerDescriptor(sim));
   }
}


UnsignedServiceTriggerFactory::UnsignedServiceTriggerFactory()
{
   //   std::cout<<"UnsignedServiceTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("UnsignedServiceTrigger", UnsignedServiceTriggerFactoryFunction);
}

