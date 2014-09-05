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
