#include "Lens.h"
#include "SingleChannelIP3RCompCategory.h"
#include "NDPairList.h"
#include "CG_SingleChannelIP3RCompCategory.h"

SingleChannelIP3RCompCategory::SingleChannelIP3RCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_SingleChannelIP3RCompCategory(sim, modelName, ndpList)
{
}

void SingleChannelIP3RCompCategory::computeTadj(RNG& rng) 
{
}

