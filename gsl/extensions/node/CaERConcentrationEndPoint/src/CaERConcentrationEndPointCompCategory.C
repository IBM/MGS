#include "Lens.h"
#include "CaERConcentrationEndPointCompCategory.h"
#include "NDPairList.h"
#include "CG_CaERConcentrationEndPointCompCategory.h"

CaERConcentrationEndPointCompCategory::CaERConcentrationEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_CaERConcentrationEndPointCompCategory(sim, modelName, ndpList)
{
}

