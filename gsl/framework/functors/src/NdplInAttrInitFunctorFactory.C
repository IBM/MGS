// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NdplInAttrInitFunctorFactory.h"

NdplInAttrInitFunctorFactory _NvplInAttrInitFunctorFactory;

#include "NdplInAttrInitFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* NdplInAttrInitFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new NdplInAttrInitFunctorType();
   }
}


NdplInAttrInitFunctorFactory::NdplInAttrInitFunctorFactory()
{
   //   std::cout<<"NdplInAttrInitFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("NdplInAttrInit", NdplInAttrInitFunctorFactoryFunction);
}

