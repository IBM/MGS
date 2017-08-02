// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "IP3ConcentrationJunctionPointCompCategory.h"
#include "NDPairList.h"
#include "CG_IP3ConcentrationJunctionPointCompCategory.h"

IP3ConcentrationJunctionPointCompCategory::IP3ConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_IP3ConcentrationJunctionPointCompCategory(sim, modelName, ndpList)
{
}
