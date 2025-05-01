// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "BengioRateInterneuronCompCategory.h"
#include "NDPairList.h"
#include "CG_BengioRateInterneuronCompCategory.h"

#define SHD getSharedMembers()

BengioRateInterneuronCompCategory::BengioRateInterneuronCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_BengioRateInterneuronCompCategory(sim, modelName, ndpList)
{
}

void BengioRateInterneuronCompCategory::outputWeightsShared(RNG& rng) 
{
}

void BengioRateInterneuronCompCategory::initialize(RNG& rng) 
{
   SHD.predictionFactor = SHD.g_D/(SHD.g_lk + SHD.g_D);
}

