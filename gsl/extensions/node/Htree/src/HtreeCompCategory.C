#include "Lens.h"
#include "HtreeCompCategory.h"
#include "NDPairList.h"
#include "CG_HtreeCompCategory.h"
/*
@ University of Canterbury 2017-2018. All rights reserved.
*/

HtreeCompCategory::HtreeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_HtreeCompCategory(sim, modelName, ndpList)
{
}

