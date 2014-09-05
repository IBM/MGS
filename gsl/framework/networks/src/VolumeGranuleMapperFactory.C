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

#include "VolumeGranuleMapperFactory.h"

VolumeGranuleMapperFactory _VolumeGranuleMapperFactory;

#include "VolumeGranuleMapperType.h"
#include "FactoryMap.h"
#include "GranuleMapperType.h"
#include "NDPairList.h"
#include <list>

extern "C"
{
   GranuleMapperType* VolumeGranuleMapperFactoryFunction(Simulation& sim, const NDPairList& ndpList) {
      return (new VolumeGranuleMapperType(sim));
   }
}


VolumeGranuleMapperFactory::VolumeGranuleMapperFactory()
{
   //   std::cout<<"VolumeGranuleMapperFactory constructed."<<std::endl;
   FactoryMap<GranuleMapperType>* gmfm = FactoryMap<GranuleMapperType>::getFactoryMap();
   gmfm->addFactory("VolumeGranuleMapper", VolumeGranuleMapperFactoryFunction);
}

