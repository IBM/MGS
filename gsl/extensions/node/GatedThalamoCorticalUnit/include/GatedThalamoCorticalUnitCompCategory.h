// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef GatedThalamoCorticalUnitCompCategory_H
#define GatedThalamoCorticalUnitCompCategory_H

#include "Mgs.h"
#include "CG_GatedThalamoCorticalUnitCompCategory.h"

class NDPairList;

class GatedThalamoCorticalUnitCompCategory : public CG_GatedThalamoCorticalUnitCompCategory
{
   public:
      GatedThalamoCorticalUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
      void updateWhitMatrixShared(RNG& rng);
};

#endif
