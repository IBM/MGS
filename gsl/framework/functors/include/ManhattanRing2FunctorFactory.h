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
