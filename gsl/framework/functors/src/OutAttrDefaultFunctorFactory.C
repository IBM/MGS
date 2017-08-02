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

#include "OutAttrDefaultFunctorFactory.h"

OutAttrDefaultFunctorFactory _OutAttrDefaultFunctorFactory;

#include "OutAttrDefaultFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* OutAttrDefaultFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new OutAttrDefaultFunctorType();
   }
}


OutAttrDefaultFunctorFactory::OutAttrDefaultFunctorFactory()
{
   //   std::cout<<"OutAttrDefaultFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("OutAttrDefault", OutAttrDefaultFunctorFactoryFunction);
}

