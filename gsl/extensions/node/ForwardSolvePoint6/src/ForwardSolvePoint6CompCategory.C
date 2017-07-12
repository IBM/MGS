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
#include "ForwardSolvePoint6CompCategory.h"
#include "NDPairList.h"
#include "CG_ForwardSolvePoint6CompCategory.h"

ForwardSolvePoint6CompCategory::ForwardSolvePoint6CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ForwardSolvePoint6CompCategory(sim, modelName, ndpList)
{
}

