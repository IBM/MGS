// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

