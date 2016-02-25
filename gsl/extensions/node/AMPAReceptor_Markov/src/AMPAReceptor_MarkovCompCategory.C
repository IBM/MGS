#include "Lens.h"
#include "AMPAReceptor_MarkovCompCategory.h"
#include "NDPairList.h"
#include "CG_AMPAReceptor_MarkovCompCategory.h"

AMPAReceptor_MarkovCompCategory::AMPAReceptor_MarkovCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_AMPAReceptor_MarkovCompCategory(sim, modelName, ndpList)
{
}

void AMPAReceptor_MarkovCompCategory::computeTadj(RNG& rng) 
{
}

