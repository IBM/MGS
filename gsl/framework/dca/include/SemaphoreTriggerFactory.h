// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SEMAPHORETRIGGERFACTORY_H
#define SEMAPHORETRIGGERFACTORY_H
#include "Copyright.h"

#include <list>

class DataItem;
class TriggerType;
class Simulation;
class NDPairList;

extern "C"
{
   TriggerType* SemaphoreTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList);
};

class SemaphoreTriggerFactory
{
   public:

      SemaphoreTriggerFactory();
      ~SemaphoreTriggerFactory() {}
};
#endif
