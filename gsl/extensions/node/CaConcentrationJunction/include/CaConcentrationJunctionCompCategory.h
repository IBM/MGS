// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConcentrationJunctionCompCategory_H
#define CaConcentrationJunctionCompCategory_H

#include "Mgs.h"
#include "CG_CaConcentrationJunctionCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class CaConcentrationJunctionCompCategory : public CG_CaConcentrationJunctionCompCategory, public CountableModel
{
   public:
      CaConcentrationJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void deriveParameters(RNG& rng);
      void count();
};

#endif
