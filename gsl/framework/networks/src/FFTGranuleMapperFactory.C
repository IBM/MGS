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

