// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
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

