#include "Lens.h"
#include "LypCollectorCompCategory.h"
#include "NDPairList.h"
#include "CG_LypCollectorCompCategory.h"

LypCollectorCompCategory::LypCollectorCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_LypCollectorCompCategory(sim, modelName, ndpList)
{
}

