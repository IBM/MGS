#include "Lens.h"
#include "VoltageMegaSynapticSpaceCompCategory.h"
#include "NDPairList.h"
#include "CG_VoltageMegaSynapticSpaceCompCategory.h"

VoltageMegaSynapticSpaceCompCategory::VoltageMegaSynapticSpaceCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_VoltageMegaSynapticSpaceCompCategory(sim, modelName, ndpList)
{
}

