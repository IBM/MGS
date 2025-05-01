// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef BengioRateInterneuronCompCategory_H
#define BengioRateInterneuronCompCategory_H

#include "Mgs.h"
#include "CG_BengioRateInterneuronCompCategory.h"

class NDPairList;

class BengioRateInterneuronCompCategory : public CG_BengioRateInterneuronCompCategory
{
   public:
      BengioRateInterneuronCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void outputWeightsShared(RNG& rng);
      void initialize(RNG& rng);
};

#endif
