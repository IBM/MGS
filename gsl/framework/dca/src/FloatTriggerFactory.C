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

