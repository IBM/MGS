#include "Lens.h"
#include "ToneUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_ToneUnitCompCategory.h"

ToneUnitCompCategory::ToneUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ToneUnitCompCategory(sim, modelName, ndpList)
{
}

void ToneUnitCompCategory::initializeShared(RNG& rng) 
{
}

