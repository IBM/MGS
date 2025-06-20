// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ManhattanRingFunctorFactory.h"

ManhattanRingFunctorFactory _ManhattanRingFunctorFactory;

#include "ManhattanRingFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* ManhattanRingFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new ManhattanRingFunctorType();
   }
}


ManhattanRingFunctorFactory::ManhattanRingFunctorFactory()
{
   //   std::cout<<"ManhattanRingFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("ManhattanRing", ManhattanRingFunctorFactoryFunction);
}

