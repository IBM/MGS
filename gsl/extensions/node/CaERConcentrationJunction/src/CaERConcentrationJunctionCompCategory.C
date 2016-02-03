#include "Lens.h"
#include "CaERConcentrationJunctionCompCategory.h"
#include "NDPairList.h"
#include "CG_CaERConcentrationJunctionCompCategory.h"

CaERConcentrationJunctionCompCategory::CaERConcentrationJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_CaERConcentrationJunctionCompCategory(sim, modelName, ndpList)
{
}

void CaERConcentrationJunctionCompCategory::deriveParameters(RNG& rng) 
{
}

