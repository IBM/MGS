// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodeFactory.h"
CG_LifeNodeFactory CG_LifeNodeFactory;
#include "CG_LifeNodeFactory.h"
#include "FactoryMap.h"
#include "LifeNodeCompCategory.h"
#include "NDPairList.h"
#include "NodeType.h"
#include "Simulation.h"

extern "C"
{
   NodeType* CG_LifeNodeFactoryFunction(Simulation& s, const NDPairList& ndpList) 
   {
      return new LifeNodeCompCategory(s, "LifeNode", ndpList);
   }
}

CG_LifeNodeFactory::CG_LifeNodeFactory() 
{
   FactoryMap<NodeType>::getFactoryMap()->addFactory("LifeNode", CG_LifeNodeFactoryFunction);
}

CG_LifeNodeFactory::~CG_LifeNodeFactory() 
{
}

