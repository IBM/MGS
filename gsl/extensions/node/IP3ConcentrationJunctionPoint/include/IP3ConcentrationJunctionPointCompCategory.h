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

#ifndef IP3ConcentrationJunctionPointCompCategory_H
#define IP3ConcentrationJunctionPointCompCategory_H

#include "Lens.h"
#include "CG_IP3ConcentrationJunctionPointCompCategory.h"

class NDPairList;

class IP3ConcentrationJunctionPointCompCategory : public CG_IP3ConcentrationJunctionPointCompCategory
{
   public:
      IP3ConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
