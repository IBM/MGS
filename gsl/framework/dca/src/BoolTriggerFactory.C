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

#include "BoolTriggerFactory.h"

BoolTriggerFactory _BoolTriggerFactory;

#include "BoolTriggerDescriptor.h"
#include "FactoryMap.h"
#include "TriggerType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   TriggerType* BoolTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new BoolTriggerDescriptor(sim));
   }
}


BoolTriggerFactory::BoolTriggerFactory()
{
   //   std::cout<<"BoolTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("BoolTrigger", BoolTriggerFactoryFunction);
}

