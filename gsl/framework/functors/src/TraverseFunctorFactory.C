// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TraverseFunctorFactory.h"

TraverseFunctorFactory _TraverseFunctorFactory;

#include "TraverseFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* TraverseFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new TraverseFunctorType();
   }
}


TraverseFunctorFactory::TraverseFunctorFactory()
{
   //   std::cout<<"TraverseFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("Traverse", TraverseFunctorFactoryFunction);
}

