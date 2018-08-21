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

#ifndef CaConcentrationEndPointCompCategory_H
#define CaConcentrationEndPointCompCategory_H

#include "Lens.h"
#include "CG_CaConcentrationEndPointCompCategory.h"

class NDPairList;

class CaConcentrationEndPointCompCategory : public CG_CaConcentrationEndPointCompCategory
{
   public:
      CaConcentrationEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
