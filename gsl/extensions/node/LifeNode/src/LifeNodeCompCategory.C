#include "Lens.h"
#include "LifeNodeCompCategory.h"
#include "NDPairList.h"
#include "CG_LifeNodeCompCategory.h"

LifeNodeCompCategory::LifeNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_LifeNodeCompCategory(sim, modelName, ndpList)
{
}

