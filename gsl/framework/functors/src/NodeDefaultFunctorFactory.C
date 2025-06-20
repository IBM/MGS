// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NodeDefaultFunctorFactory.h"

NodeDefaultFunctorFactory _NodeDefaultFunctorFactory;

#include "NodeDefaultFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* NodeDefaultFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new NodeDefaultFunctorType();
   }
}


NodeDefaultFunctorFactory::NodeDefaultFunctorFactory()
{
   //   std::cout<<"NodeDefaultFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("NodeDefault", NodeDefaultFunctorFactoryFunction);
}

