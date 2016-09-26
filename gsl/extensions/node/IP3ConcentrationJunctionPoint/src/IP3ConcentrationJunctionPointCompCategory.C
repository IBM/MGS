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
#include "IP3ConcentrationJunctionPointCompCategory.h"
#include "NDPairList.h"
#include "CG_IP3ConcentrationJunctionPointCompCategory.h"

IP3ConcentrationJunctionPointCompCategory::IP3ConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_IP3ConcentrationJunctionPointCompCategory(sim, modelName, ndpList)
{
}
