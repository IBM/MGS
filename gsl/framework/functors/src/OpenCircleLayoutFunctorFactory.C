// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "OpenCircleLayoutFunctorFactory.h"

OpenCircleLayoutFunctorFactory _OpenCircleLayoutFunctorFactory;

#include "OpenCircleLayoutFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* OpenCircleLayoutFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new OpenCircleLayoutFunctorType();
   }
}


OpenCircleLayoutFunctorFactory::OpenCircleLayoutFunctorFactory()
{
   //   std::cout<<"OpenCircleLayoutFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("OpenCircleLayout", OpenCircleLayoutFunctorFactoryFunction);
}

