// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayKtCompCategory_H
#define ChannelHayKtCompCategory_H

#include "Lens.h"
#include "CG_ChannelHayKtCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHayKtCompCategory : public CG_ChannelHayKtCompCategory, public CountableModel
{
   public:
      ChannelHayKtCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
