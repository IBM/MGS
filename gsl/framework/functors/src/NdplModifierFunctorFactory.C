// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NdplModifierFunctorFactory.h"

NdplModifierFunctorFactory _NvplModifierFunctorFactory;

#include "NdplModifierFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* NdplModifierFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new NdplModifierFunctorType();
   }
}


NdplModifierFunctorFactory::NdplModifierFunctorFactory()
{
   //   std::cout<<"NdplModifierFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("NdplModifier", NdplModifierFunctorFactoryFunction);
}

