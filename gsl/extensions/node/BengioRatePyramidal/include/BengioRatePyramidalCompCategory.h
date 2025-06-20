// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef BengioRatePyramidalCompCategory_H
#define BengioRatePyramidalCompCategory_H

#include "Mgs.h"
#include "CG_BengioRatePyramidalCompCategory.h"

class NDPairList;

class BengioRatePyramidalCompCategory : public CG_BengioRatePyramidalCompCategory
{
   public:
      BengioRatePyramidalCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void outputWeightsShared(RNG& rng);
      void initialize(RNG& rng);
};

#endif
