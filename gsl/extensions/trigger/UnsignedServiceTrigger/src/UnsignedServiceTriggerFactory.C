// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "UnsignedServiceTriggerFactory.h"

UnsignedServiceTriggerFactory _UnsignedServiceTriggerFactory;

#include "UnsignedServiceTriggerDescriptor.h"
#include "FactoryMap.h"
#include "TriggerType.h"
#include "NDPairList.h"

extern "C"
{
   TriggerType* UnsignedServiceTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new UnsignedServiceTriggerDescriptor(sim));
   }
}


UnsignedServiceTriggerFactory::UnsignedServiceTriggerFactory()
{
   //   std::cout<<"UnsignedServiceTriggerFactory constructed."<<std::endl;
   FactoryMap<TriggerType>* tfm = FactoryMap<TriggerType>::getFactoryMap();
   tfm->addFactory("UnsignedServiceTrigger", UnsignedServiceTriggerFactoryFunction);
}

