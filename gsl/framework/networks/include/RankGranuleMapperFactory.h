// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RANKGRANULEMAPPERFACTORY_H
#define RANKGRANULEMAPPERFACTORY_H
#include "Copyright.h"

#include <list>

class DataItem;
class GranuleMapperType;
class Simulation;
class NDPairList;

extern "C"
{
   GranuleMapperType* RankGranuleMapperFactoryFunction(Simulation& sim, const NDPairList& ndpList);
};

class RankGranuleMapperFactory
{
   public:

      RankGranuleMapperFactory();
      ~RankGranuleMapperFactory() {}
};
#endif
