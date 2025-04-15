// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ZhengSORNInhUnitCompCategory_H
#define ZhengSORNInhUnitCompCategory_H

#include "Lens.h"
#include "CG_ZhengSORNInhUnitCompCategory.h"

class NDPairList;

class ZhengSORNInhUnitCompCategory : public CG_ZhengSORNInhUnitCompCategory
{
   public:
      ZhengSORNInhUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
      void outputWeightsShared(RNG& rng); 
};

#endif
