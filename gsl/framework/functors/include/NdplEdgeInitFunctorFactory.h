// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef NDPLEDGEINITFUNCTORFACTORY_H
#define NDPLEDGEINITFUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* NdplEdgeInitFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class NdplEdgeInitFunctorFactory
{
   public:
      NdplEdgeInitFunctorFactory();
      ~NdplEdgeInitFunctorFactory(){};

};
#endif
