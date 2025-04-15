// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _EACHDSTPROPSRCFUNCTORFACTORY_H
#define _EACHDSTPROPSRCFUNCTORFACTORY_H
#include "Copyright.h"

class FunctorType;
class NDPairList;
class Simulation;

extern "C"
{
   FunctorType* EachDstPropSrcFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList);
}


class EachDstPropSrcFunctorFactory
{
   public:
      EachDstPropSrcFunctorFactory();
      ~EachDstPropSrcFunctorFactory(){};

};
#endif
