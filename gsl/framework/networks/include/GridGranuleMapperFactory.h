// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
