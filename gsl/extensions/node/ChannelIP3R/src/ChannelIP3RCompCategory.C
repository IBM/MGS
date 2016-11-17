#include "Lens.h"
#include "ChannelIP3RCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelIP3RCompCategory.h"

ChannelIP3RCompCategory::ChannelIP3RCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelIP3RCompCategory(sim, modelName, ndpList)
{
}

void ChannelIP3RCompCategory::computeTadj(RNG& rng) 
{
}

