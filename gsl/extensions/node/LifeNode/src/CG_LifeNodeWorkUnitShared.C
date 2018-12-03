// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodeWorkUnitShared.h"
#include "CG_LifeNodeCompCategory.h"
#include "LifeNodeCompCategory.h"
#include "WorkUnit.h"
#include "rndm.h"

CG_LifeNodeWorkUnitShared::CG_LifeNodeWorkUnitShared(void (LifeNodeCompCategory::*computeState) (RNG&), CG_LifeNodeCompCategory* compCategory) 
   : WorkUnit(), _computeState(computeState){
   _compCategory = static_cast<LifeNodeCompCategory*>(compCategory);
   _rng.reSeedShared(urandom(_compCategory->getSimulation().getWorkUnitRandomSeedGenerator()));
}

void CG_LifeNodeWorkUnitShared::execute() 
{
   (*_compCategory.*_computeState)(_rng);
}

CG_LifeNodeWorkUnitShared::~CG_LifeNodeWorkUnitShared() 
{
}

