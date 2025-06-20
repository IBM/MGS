// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ManhattanRing2FunctorFactory.h"

ManhattanRing2FunctorFactory _ManhattanRing2FunctorFactory;

#include "ManhattanRing2FunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* ManhattanRing2FunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new ManhattanRing2FunctorType();
   }
}


ManhattanRing2FunctorFactory::ManhattanRing2FunctorFactory()
{
   //   std::cout<<"ManhattanRing2FunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("ManhattanRing2", ManhattanRing2FunctorFactoryFunction);
}

