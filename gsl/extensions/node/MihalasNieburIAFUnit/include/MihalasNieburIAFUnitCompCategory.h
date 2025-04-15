// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MihalasNieburIAFUnitCompCategory_H
#define MihalasNieburIAFUnitCompCategory_H

#include "Lens.h"
#include "CG_MihalasNieburIAFUnitCompCategory.h"

class NDPairList;

class MihalasNieburIAFUnitCompCategory : public CG_MihalasNieburIAFUnitCompCategory
{
   public:
      MihalasNieburIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
};

#endif
