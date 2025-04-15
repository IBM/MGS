// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "IP3ConcentrationJunctionPointCompCategory.h"
#include "NDPairList.h"
#include "CG_IP3ConcentrationJunctionPointCompCategory.h"

IP3ConcentrationJunctionPointCompCategory::IP3ConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_IP3ConcentrationJunctionPointCompCategory(sim, modelName, ndpList)
{
}
