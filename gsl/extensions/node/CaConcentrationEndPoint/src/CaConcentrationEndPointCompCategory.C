// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CaConcentrationEndPointCompCategory.h"
#include "NDPairList.h"
#include "CG_CaConcentrationEndPointCompCategory.h"

CaConcentrationEndPointCompCategory::CaConcentrationEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_CaConcentrationEndPointCompCategory(sim, modelName, ndpList)
{
}
