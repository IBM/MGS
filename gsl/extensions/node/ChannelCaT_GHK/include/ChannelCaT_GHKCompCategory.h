// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelCaT_GHKCompCategory_H
#define ChannelCaT_GHKCompCategory_H

#include "Mgs.h"
#include "CG_ChannelCaT_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaT_GHKCompCategory : public CG_ChannelCaT_GHKCompCategory,
                               public CountableModel
{
  public:
  ChannelCaT_GHKCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
