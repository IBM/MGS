// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RabinovichWinnerlessUnitCompCategory_H
#define RabinovichWinnerlessUnitCompCategory_H

#include "Mgs.h"
#include "CG_RabinovichWinnerlessUnitCompCategory.h"

class NDPairList;

class RabinovichWinnerlessUnitCompCategory : public CG_RabinovichWinnerlessUnitCompCategory
{
   public:
      RabinovichWinnerlessUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
};

#endif
