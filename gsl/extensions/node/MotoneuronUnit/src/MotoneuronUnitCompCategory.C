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

#include "Lens.h"
#include "MotoneuronUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_MotoneuronUnitCompCategory.h"

MotoneuronUnitCompCategory::MotoneuronUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_MotoneuronUnitCompCategory(sim, modelName, ndpList)
{
}

void MotoneuronUnitCompCategory::initializeShared(RNG& rng) 
{
}

