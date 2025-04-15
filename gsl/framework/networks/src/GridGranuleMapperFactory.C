// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GridGranuleMapperFactory.h"

GridGranuleMapperFactory _GridGranuleMapperFactory;

#include "GridGranuleMapperType.h"
#include "FactoryMap.h"
#include "GranuleMapperType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   GranuleMapperType* GridGranuleMapperFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new GridGranuleMapperType(sim));
   }
}


GridGranuleMapperFactory::GridGranuleMapperFactory()
{
   //   std::cout<<"GridGranuleMapperFactory constructed."<<std::endl;
   FactoryMap<GranuleMapperType>* gmfm = FactoryMap<GranuleMapperType>::getFactoryMap();
   gmfm->addFactory("GridGranuleMapper", GridGranuleMapperFactoryFunction);
}

