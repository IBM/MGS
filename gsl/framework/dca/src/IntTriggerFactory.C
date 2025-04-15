// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "IntTriggerFactory.h"

IntTriggerFactory _IntTriggerFactory;

#include "IntTriggerDescriptor.h"
#include "FactoryMap.h"
#include "TriggerType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   TriggerType* IntTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new IntTriggerDescriptor(sim));
   }
}


IntTriggerFactory::IntTriggerFactory()
{
   //   std::cout<<"IntTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("IntTrigger", IntTriggerFactoryFunction);
}

