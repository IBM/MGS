// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelBKCompCategory_H
#define ChannelBKCompCategory_H

#include "Mgs.h"
#include "CG_ChannelBKCompCategory.h"

#include "CountableModel.h"

class NDPairList;

class ChannelBKCompCategory : public CG_ChannelBKCompCategory, public CountableModel
{
   public:
      ChannelBKCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
