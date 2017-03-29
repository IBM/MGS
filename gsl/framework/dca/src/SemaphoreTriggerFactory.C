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

