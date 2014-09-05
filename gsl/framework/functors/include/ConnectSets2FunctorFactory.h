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

#ifndef CONNECTSETS2FUNCTORFACTORY_H
#define CONNECTSETS2FUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* ConnectSets2FunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class ConnectSets2FunctorFactory
{
   public:
      ConnectSets2FunctorFactory();
      ~ConnectSets2FunctorFactory(){};

};
#endif
