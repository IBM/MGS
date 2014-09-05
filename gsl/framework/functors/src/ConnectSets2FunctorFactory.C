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

#include "ConnectSets2FunctorFactory.h"

ConnectSets2FunctorFactory _ConnectSets2FunctorFactory;

#include "ConnectSets2FunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* ConnectSets2FunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new ConnectSets2FunctorType();
   }
}


ConnectSets2FunctorFactory::ConnectSets2FunctorFactory()
{
   //   std::cout<<"ConnectSets2FunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("ConnectSets2", ConnectSets2FunctorFactoryFunction);
}

