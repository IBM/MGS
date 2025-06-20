// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "RankGranuleMapperFactory.h"

RankGranuleMapperFactory _RankGranuleMapperFactory;

#include "RankGranuleMapperType.h"
#include "FactoryMap.h"
#include "GranuleMapperType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   GranuleMapperType* RankGranuleMapperFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new RankGranuleMapperType(sim));
   }
}


RankGranuleMapperFactory::RankGranuleMapperFactory()
{
   //   std::cout<<"RankGranuleMapperFactory constructed."<<std::endl;
   FactoryMap<GranuleMapperType>* gmfm = FactoryMap<GranuleMapperType>::getFactoryMap();
   gmfm->addFactory("RankGranuleMapper", RankGranuleMapperFactoryFunction);
}

