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

#ifndef TRAVERSEFUNCTORFACTORY_H
#define TRAVERSEFUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* TraverseFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class TraverseFunctorFactory
{
   public:
      TraverseFunctorFactory();
      ~TraverseFunctorFactory(){};

};
#endif
