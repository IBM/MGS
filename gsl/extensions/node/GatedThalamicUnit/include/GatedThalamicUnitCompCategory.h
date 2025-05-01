// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef GatedThalamicUnitCompCategory_H
#define GatedThalamicUnitCompCategory_H

#include "Mgs.h"
#include "CG_GatedThalamicUnitCompCategory.h"

class NDPairList;

class GatedThalamicUnitCompCategory : public CG_GatedThalamicUnitCompCategory
{
   public:
      GatedThalamicUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
};

#endif
