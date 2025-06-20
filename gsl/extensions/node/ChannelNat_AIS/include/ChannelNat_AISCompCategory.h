// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelNat_AISCompCategory_H
#define ChannelNat_AISCompCategory_H

#include "Mgs.h"
#include "CG_ChannelNat_AISCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelNat_AISCompCategory : public CG_ChannelNat_AISCompCategory,
                               public CountableModel
{
   public:
      ChannelNat_AISCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
