// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef UNSIGNEDSERVICETRIGGERFACTORY_H
#define UNSIGNEDSERVICETRIGGERFACTORY_H
#include "Copyright.h"

#include <list>

class DataItem;
class TriggerType;
class Simulation;
class NDPairList;

extern "C"
{
   TriggerType* UnsignedServiceTriggerFactoryFunction(Simulation& sim, const NDPairList& ndpList);
};

class UnsignedServiceTriggerFactory
{
   public:

      UnsignedServiceTriggerFactory();
      ~UnsignedServiceTriggerFactory() {}
};
#endif
