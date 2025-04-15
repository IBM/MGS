// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FFTGranuleMapperFactory.h"

FFTGranuleMapperFactory _FFTGranuleMapperFactory;

#include "FFTGranuleMapperType.h"
#include "FactoryMap.h"
#include "GranuleMapperType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   GranuleMapperType* FFTGranuleMapperFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new FFTGranuleMapperType(sim));
   }
}


FFTGranuleMapperFactory::FFTGranuleMapperFactory()
{
   //   std::cout<<"FFTGranuleMapperFactory constructed."<<std::endl;
   FactoryMap<GranuleMapperType>* gmfm = FactoryMap<GranuleMapperType>::getFactoryMap();
   gmfm->addFactory("FFTGranuleMapper", FFTGranuleMapperFactoryFunction);
}

