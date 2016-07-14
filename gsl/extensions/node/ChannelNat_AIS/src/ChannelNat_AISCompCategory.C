#include "Lens.h"
#include "ChannelNat_AISCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelNat_AISCompCategory.h"

ChannelNat_AISCompCategory::ChannelNat_AISCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelNat_AISCompCategory(sim, modelName, ndpList)
{
}

void ChannelNat_AISCompCategory::computeE(RNG& rng) 
{
}

