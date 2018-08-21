// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "UniformLayoutFunctorFactory.h"

UniformLayoutFunctorFactory _UniformLayoutFunctorFactory;

#include "UniformLayoutFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* UniformLayoutFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new UniformLayoutFunctorType();
   }
}


UniformLayoutFunctorFactory::UniformLayoutFunctorFactory()
{
   //   std::cout<<"UniformLayoutFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("UniformLayout", UniformLayoutFunctorFactoryFunction);
}

