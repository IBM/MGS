#include "Lens.h"
#include "HtreeCompCategory.h"
#include "NDPairList.h"
#include "CG_HtreeCompCategory.h"

HtreeCompCategory::HtreeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_HtreeCompCategory(sim, modelName, ndpList)
{
}

