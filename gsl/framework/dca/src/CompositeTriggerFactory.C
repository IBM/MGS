// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "CompositeTriggerFactory.h"
#include "NDPairList.h"
#include "CompositeTriggerDescriptor.h"

extern "C"
{
   TriggerType* CompositeTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new CompositeTriggerDescriptor(sim));
   }
}

#include "TypeClassifier.h"

CompositeTriggerFactory _CompositeTriggerFactory;
#include "FactoryMap.h"
#include "TriggerType.h"
#include <list>



CompositeTriggerFactory::CompositeTriggerFactory()
{
   //   std::cout<<"CompositeTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("CompositeTrigger", CompositeTriggerFactoryFunction);
}

