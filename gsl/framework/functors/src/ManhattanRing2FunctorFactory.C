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

#include "ManhattanRing2FunctorFactory.h"

ManhattanRing2FunctorFactory _ManhattanRing2FunctorFactory;

#include "ManhattanRing2FunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* ManhattanRing2FunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new ManhattanRing2FunctorType();
   }
}


ManhattanRing2FunctorFactory::ManhattanRing2FunctorFactory()
{
   //   std::cout<<"ManhattanRing2FunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("ManhattanRing2", ManhattanRing2FunctorFactoryFunction);
}

