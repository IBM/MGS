// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

