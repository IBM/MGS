#include "Lens.h"
#include "SynapticCleftCompCategory.h"
#include "NDPairList.h"
#include "CG_SynapticCleftCompCategory.h"

SynapticCleftCompCategory::SynapticCleftCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_SynapticCleftCompCategory(sim, modelName, ndpList)
{
}

