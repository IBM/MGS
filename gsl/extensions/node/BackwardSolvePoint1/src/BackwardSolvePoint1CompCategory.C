// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "BackwardSolvePoint1CompCategory.h"
#include "NDPairList.h"
#include "CG_BackwardSolvePoint1CompCategory.h"

BackwardSolvePoint1CompCategory::BackwardSolvePoint1CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_BackwardSolvePoint1CompCategory(sim, modelName, ndpList)
{
}

