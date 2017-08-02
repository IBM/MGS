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

