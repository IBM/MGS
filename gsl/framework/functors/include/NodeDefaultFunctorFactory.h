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

#ifndef NODEDEFAULTFUNCTORFACTORY_H
#define NODEDEFAULTFUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* NodeDefaultFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class NodeDefaultFunctorFactory
{
   public:
      NodeDefaultFunctorFactory();
      ~NodeDefaultFunctorFactory(){};

};
#endif
