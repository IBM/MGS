// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LinskerInfomaxUnitCompCategory_H
#define LinskerInfomaxUnitCompCategory_H

#include "Mgs.h"
#include "CG_LinskerInfomaxUnitCompCategory.h"

class NDPairList;

class LinskerInfomaxUnitCompCategory : public CG_LinskerInfomaxUnitCompCategory
{
   public:
      LinskerInfomaxUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void invertQmatrixShared(RNG& rng);
};

#endif
