#include "Lens.h"
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

