// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FloatTriggerFactory.h"

FloatTriggerFactory _FloatTriggerFactory;

#include "FloatTriggerDescriptor.h"
#include "FactoryMap.h"
#include "TriggerType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   TriggerType* FloatTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new FloatTriggerDescriptor(sim));
   }
}


FloatTriggerFactory::FloatTriggerFactory()
{
   //   std::cout<<"FloatTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("FloatTrigger", FloatTriggerFactoryFunction);
}

