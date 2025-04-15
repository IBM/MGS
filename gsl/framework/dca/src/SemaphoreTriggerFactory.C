// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SemaphoreTriggerFactory.h"
#include "NDPairList.h"
#include "SemaphoreTriggerDescriptor.h"

extern "C"
{
   TriggerType* SemaphoreTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new SemaphoreTriggerDescriptor(sim));
   }
}

#include "TypeClassifier.h"

SemaphoreTriggerFactory _SemaphoreTriggerFactory;
#include "FactoryMap.h"
#include "TriggerType.h"
#include <list>



SemaphoreTriggerFactory::SemaphoreTriggerFactory()
{
   //   std::cout<<"SemaphoreTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("SemaphoreTrigger", SemaphoreTriggerFactoryFunction);
}

