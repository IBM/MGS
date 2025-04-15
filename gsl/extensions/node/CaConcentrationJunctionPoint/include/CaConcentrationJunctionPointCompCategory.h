// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConcentrationJunctionPointCompCategory_H
#define CaConcentrationJunctionPointCompCategory_H

#include "Lens.h"
#include "CG_CaConcentrationJunctionPointCompCategory.h"

class NDPairList;

class CaConcentrationJunctionPointCompCategory : public CG_CaConcentrationJunctionPointCompCategory
{
   public:
      CaConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
