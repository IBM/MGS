#include "Lens.h"
#include "EpileptorNodeCompCategory.h"
#include "EpileptorNode.h" // to access paper dependant flags
#include "NDPairList.h"
#include "CG_EpileptorNodeCompCategory.h"

EpileptorNodeCompCategory::EpileptorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_EpileptorNodeCompCategory(sim, modelName, ndpList){
}

void EpileptorNodeCompCategory::initializeShared(RNG& rng) 
{
#ifdef Proix_et_al_2014
  SHD.y0=1.0;
#endif
}
