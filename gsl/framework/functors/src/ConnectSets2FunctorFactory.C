// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ConnectSets2FunctorFactory.h"

ConnectSets2FunctorFactory _ConnectSets2FunctorFactory;

#include "ConnectSets2FunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* ConnectSets2FunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new ConnectSets2FunctorType();
   }
}


ConnectSets2FunctorFactory::ConnectSets2FunctorFactory()
{
   //   std::cout<<"ConnectSets2FunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("ConnectSets2", ConnectSets2FunctorFactoryFunction);
}

