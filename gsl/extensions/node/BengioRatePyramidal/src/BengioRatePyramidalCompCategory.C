// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "BengioRatePyramidalCompCategory.h"
#include "NDPairList.h"
#include "CG_BengioRatePyramidalCompCategory.h"

#define SHD getSharedMembers()

BengioRatePyramidalCompCategory::BengioRatePyramidalCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_BengioRatePyramidalCompCategory(sim, modelName, ndpList)
{
}

void BengioRatePyramidalCompCategory::outputWeightsShared(RNG& rng) 
{
}

void BengioRatePyramidalCompCategory::initialize(RNG& rng) 
{
   SHD.predictionFactor = SHD.g_B/(SHD.g_lk + SHD.g_B + SHD.g_A);
}
