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

#ifndef IP3ConcentrationEndPointCompCategory_H
#define IP3ConcentrationEndPointCompCategory_H

#include "Lens.h"
#include "CG_IP3ConcentrationEndPointCompCategory.h"

class NDPairList;

class IP3ConcentrationEndPointCompCategory : public CG_IP3ConcentrationEndPointCompCategory
{
   public:
      IP3ConcentrationEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
