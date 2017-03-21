// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "RangePassThruFunctorFactory.h"

RangePassThruFunctorFactory _RangePassThruFunctorFactory;

#include "RangePassThruFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* RangePassThruFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new RangePassThruFunctorType();
   }
}


RangePassThruFunctorFactory::RangePassThruFunctorFactory()
{
   //   std::cout<<"RangePassThruFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("RangePassThru", RangePassThruFunctorFactoryFunction);
}

