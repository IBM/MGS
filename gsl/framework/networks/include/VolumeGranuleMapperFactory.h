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

#ifndef VOLUMEGRANULEMAPPERFACTORY_H
#define VOLUMEGRANULEMAPPERFACTORY_H
#include "Copyright.h"

#include <list>

class DataItem;
class GranuleMapperType;
class Simulation;
class NDPairList;

extern "C"
{
   GranuleMapperType* VolumeGranuleMapperFactoryFunction(Simulation& sim, const NDPairList& ndpList);
};

class VolumeGranuleMapperFactory
{
   public:

      VolumeGranuleMapperFactory();
      ~VolumeGranuleMapperFactory() {}
};
#endif
