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

