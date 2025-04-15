// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ZhengSORNExcUnitCompCategory_H
#define ZhengSORNExcUnitCompCategory_H

#include "Lens.h"
#include "CG_ZhengSORNExcUnitCompCategory.h"

class NDPairList;

class ZhengSORNExcUnitCompCategory : public CG_ZhengSORNExcUnitCompCategory
{
   public:
      ZhengSORNExcUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
};

#endif
