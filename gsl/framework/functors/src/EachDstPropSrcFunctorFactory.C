// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EachDstPropSrcFunctorFactory.h"

EachDstPropSrcFunctorFactory _EachDstPropSrcFunctorFactory;

#include "EachDstPropSrcFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* EachDstPropSrcFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new EachDstPropSrcFunctorType();
   }
}


EachDstPropSrcFunctorFactory::EachDstPropSrcFunctorFactory()
{
   //   std::cout<<"EachDstPropSrcFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("EachDstPropSrc", EachDstPropSrcFunctorFactoryFunction);
}

