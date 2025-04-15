// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "UnsignedTriggerFactory.h"

UnsignedTriggerFactory _UnsignedTriggerFactory;

#include "UnsignedTriggerDescriptor.h"
#include "FactoryMap.h"
#include "TriggerType.h"
#include "NDPairList.h"

extern "C"
{
   TriggerType* UnsignedTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new UnsignedTriggerDescriptor(sim));
   }
}


UnsignedTriggerFactory::UnsignedTriggerFactory()
{
   //   std::cout<<"UnsignedTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("UnsignedTrigger", UnsignedTriggerFactoryFunction);
}

