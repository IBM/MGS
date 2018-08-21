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

