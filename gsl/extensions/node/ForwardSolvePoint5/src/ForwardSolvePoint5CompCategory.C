// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ForwardSolvePoint5CompCategory.h"
#include "NDPairList.h"
#include "CG_ForwardSolvePoint5CompCategory.h"

ForwardSolvePoint5CompCategory::ForwardSolvePoint5CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ForwardSolvePoint5CompCategory(sim, modelName, ndpList)
{
}

