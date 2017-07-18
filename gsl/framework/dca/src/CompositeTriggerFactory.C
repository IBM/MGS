// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

