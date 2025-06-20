// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BindFrontFunctorFactory.h"

BindFrontFunctorFactory _BindFrontFunctorFactory;

#include "BindFrontFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* BindFrontFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new BindFrontFunctorType();
   }
}


BindFrontFunctorFactory::BindFrontFunctorFactory()
{
   //   std::cout<<"BindFrontFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("BindFront", BindFrontFunctorFactoryFunction);
}

