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

#include "NdplNodeInitFunctorFactory.h"

NdplNodeInitFunctorFactory _NvplNodeInitFunctorFactory;

#include "NdplNodeInitFunctorType.h"
#include "FunctorType.h"
#include "FactoryMap.h"
#include "NDPairList.h"

extern "C"
{
   FunctorType* NdplNodeInitFunctorFactoryFunction(Simulation& s, const NDPairList& ndpList) {
      return new NdplNodeInitFunctorType();
   }
}


NdplNodeInitFunctorFactory::NdplNodeInitFunctorFactory()
{
   //   std::cout<<"NdplNodeInitFunctorFactory constructed."<<std::endl;
   FactoryMap<FunctorType>::getFactoryMap()->addFactory("NdplNodeInit", NdplNodeInitFunctorFactoryFunction);
}

