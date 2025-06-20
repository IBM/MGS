// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BINDFRONTFUNCTORFACTORY_H
#define BINDFRONTFUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* BindFrontFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class BindFrontFunctorFactory
{
   public:
      BindFrontFunctorFactory();
      ~BindFrontFunctorFactory(){};

};
#endif
