#include "Lens.h"
#include "EpileptorNodeCompCategory.h"
#include "NDPairList.h"
#include "CG_EpileptorNodeCompCategory.h"

EpileptorNodeCompCategory::EpileptorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_EpileptorNodeCompCategory(sim, modelName, ndpList){
}

