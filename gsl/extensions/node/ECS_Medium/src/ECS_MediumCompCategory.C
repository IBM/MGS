#include "Lens.h"
#include "ECS_MediumCompCategory.h"
#include "NDPairList.h"
#include "CG_ECS_MediumCompCategory.h"

ECS_MediumCompCategory::ECS_MediumCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ECS_MediumCompCategory(sim, modelName, ndpList)
{
}

