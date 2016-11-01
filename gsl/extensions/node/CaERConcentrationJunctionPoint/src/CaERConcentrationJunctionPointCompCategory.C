#include "Lens.h"
#include "CaERConcentrationJunctionPointCompCategory.h"
#include "NDPairList.h"
#include "CG_CaERConcentrationJunctionPointCompCategory.h"

CaERConcentrationJunctionPointCompCategory::CaERConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_CaERConcentrationJunctionPointCompCategory(sim, modelName, ndpList)
{
}

