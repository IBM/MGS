// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConcentrationEndPointCompCategory_H
#define CaConcentrationEndPointCompCategory_H

#include "Mgs.h"
#include "CG_CaConcentrationEndPointCompCategory.h"

class NDPairList;

class CaConcentrationEndPointCompCategory : public CG_CaConcentrationEndPointCompCategory
{
   public:
      CaConcentrationEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
