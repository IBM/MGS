// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MANHATTANRING2FUNCTORFACTORY_H
#define MANHATTANRING2FUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* ManhattanRing2FunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class ManhattanRing2FunctorFactory
{
   public:
      ManhattanRing2FunctorFactory();
      ~ManhattanRing2FunctorFactory(){};

};
#endif
