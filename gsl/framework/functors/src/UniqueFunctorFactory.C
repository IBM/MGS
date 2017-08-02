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

#include "UniqueFunctorFactory.h"

UniqueFunctorFactory _UniqueFunctorFactory;

#include "UniqueFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* UniqueFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new UniqueFunctorType();
   }
}


UniqueFunctorFactory::UniqueFunctorFactory()
{
   //   std::cout<<"UniqueFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("Unique", UniqueFunctorFactoryFunction);
}

