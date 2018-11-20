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
#include "CG_LifeNodeWorkUnitInstance.h"
#include "CG_LifeNodeCompCategory.h"
#include "NodePartitionItem.h"
#include "WorkUnit.h"
#include "rndm.h"

CG_LifeNodeWorkUnitInstance::CG_LifeNodeWorkUnitInstance(NodePartitionItem* arg, void (CG_LifeNodeCompCategory::*computeState) (NodePartitionItem*, CG_LifeNodeWorkUnitInstance*), CG_LifeNodeCompCategory* compCategory) 
   : WorkUnit(), _arg(arg), _compCategory(compCategory), _computeState(computeState)
{
   _rng.reSeed(urandom(_compCategory->getSimulation().getWorkUnitRandomSeedGenerator()), _compCategory->getSimulation().getRank());
}

void CG_LifeNodeWorkUnitInstance::execute() 
{
   (*_compCategory.*_computeState)(_arg, this);
}

CG_LifeNodeWorkUnitInstance::~CG_LifeNodeWorkUnitInstance() 
{
}

