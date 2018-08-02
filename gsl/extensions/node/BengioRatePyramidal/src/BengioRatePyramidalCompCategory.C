#include "Lens.h"
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
