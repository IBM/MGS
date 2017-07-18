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

#ifndef GRIDGRANULEMAPPERFACTORY_H
#define GRIDGRANULEMAPPERFACTORY_H
#include "Copyright.h"

#include <list>

class DataItem;
class GranuleMapperType;
class Simulation;
class NDPairList;

extern "C"
{
   GranuleMapperType* GridGranuleMapperFactoryFunction(Simulation& sim, const NDPairList& ndpList);
};

class GridGranuleMapperFactory
{
   public:

      GridGranuleMapperFactory();
      ~GridGranuleMapperFactory() {}
};
#endif
