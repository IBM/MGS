// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
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
