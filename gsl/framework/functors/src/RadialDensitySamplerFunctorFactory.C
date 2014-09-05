// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "RadialDensitySamplerFunctorFactory.h"

RadialDensitySamplerFunctorFactory _RadialDensitySamplerFunctorFactory;

#include "RadialDensitySamplerFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* RadialDensitySamplerFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new RadialDensitySamplerFunctorType();
   }
}


RadialDensitySamplerFunctorFactory::RadialDensitySamplerFunctorFactory()
{
   //   std::cout<<"RadialDensitySamplerFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("RadialDensitySampler", RadialDensitySamplerFunctorFactoryFunction);
}

