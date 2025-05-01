// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelNasCompCategory_H
#define ChannelNasCompCategory_H

#include "CG_ChannelNasCompCategory.h"
#include "CountableModel.h"
#include "Mgs.h"

class NDPairList;

class ChannelNasCompCategory : public CG_ChannelNasCompCategory,
                               public CountableModel
{
  public:
  ChannelNasCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
