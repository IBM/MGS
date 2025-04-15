// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NazeSORNExcUnitCompCategory_H
#define NazeSORNExcUnitCompCategory_H

#include "Lens.h"
#include "CG_NazeSORNExcUnitCompCategory.h"

class NDPairList;

class NazeSORNExcUnitCompCategory : public CG_NazeSORNExcUnitCompCategory
{
   public:
      NazeSORNExcUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void saveInitParams(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void outputDelaysShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
};

#endif
