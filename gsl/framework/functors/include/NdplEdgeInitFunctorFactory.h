// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
