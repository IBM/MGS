// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef SAMEFUNCTORFACTORY_H
#define SAMEFUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* SameFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class SameFunctorFactory
{
   public:
      SameFunctorFactory();
      ~SameFunctorFactory(){};

};
#endif
