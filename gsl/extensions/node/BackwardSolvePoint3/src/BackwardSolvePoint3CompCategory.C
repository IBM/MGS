// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "BackwardSolvePoint3CompCategory.h"
#include "NDPairList.h"
#include "CG_BackwardSolvePoint3CompCategory.h"

BackwardSolvePoint3CompCategory::BackwardSolvePoint3CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_BackwardSolvePoint3CompCategory(sim, modelName, ndpList)
{
}

