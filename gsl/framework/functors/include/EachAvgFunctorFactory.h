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

#ifndef EACHAVGFUNCTORFACTORY_H
#define EACHAVGFUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* EachAvgFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class EachAvgFunctorFactory
{
   public:
      EachAvgFunctorFactory();
      ~EachAvgFunctorFactory(){};

};
#endif
