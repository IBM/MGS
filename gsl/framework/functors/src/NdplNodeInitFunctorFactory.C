// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NdplNodeInitFunctorFactory.h"

NdplNodeInitFunctorFactory _NvplNodeInitFunctorFactory;

#include "NdplNodeInitFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* NdplNodeInitFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new NdplNodeInitFunctorType();
   }
}


NdplNodeInitFunctorFactory::NdplNodeInitFunctorFactory()
{
   //   std::cout<<"NdplNodeInitFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("NdplNodeInit", NdplNodeInitFunctorFactoryFunction);
}

